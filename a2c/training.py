"""
File: training
Date: 2019-07-25 
Author: Jon Deaton (jonpauldeaton@gmail.com)

Inspired by:
    - http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
    - https://github.com/MG2033/A2C/
    
"""

from a2c.hyperparameters import HyperParameters
from a2c.rollout import Rollout, transpose_batch
from a2c.async_coordinator import AsyncCoordinator

import os
from tqdm import tqdm
import numpy as np
import random
import logging
logger = logging.getLogger("root")
logger.propagate = False

from multiprocessing import Queue, Semaphore


def worker_target(wid: int, queue: Queue, sema: Semaphore,
                  get_env, model_directory, to_action, hyperams: HyperParameters):
    """ the task that each worker process performs: gather the complete
        roll-out of an episode using the latest model and send it back to the
        master process. """
    print(f"Worker {wid} started")

    env = get_env()

    # workers can't use GPU because TensorFlow...
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    from a2c.eager_models import make_model

    input_shape = (None,) + env.observation_space.shape
    model = make_model(hyperams.architecture,
                       hyperams.encoder_class,
                       input_shape,
                       hyperams.action_shape)

    sema.acquire()  # wait for master process to signal

    while True:
        model.load_weights(model_directory)
        rollout = get_rollout(model, env,
                              hyperams.agents_per_env,
                              hyperams.episode_length,
                              to_action,
                              progress_bar=True)
        queue.put(rollout.as_batch(cache=False))
        del rollout  # saves some memory


def get_rollout(model, env, agents_per_env, episode_length, to_action,
                record=True, progress_bar=True) -> Rollout:
    """ performs a roll-out """
    import tensorflow as tf
    rollout = Rollout() if record else None

    observations = env.reset()
    dones = [False] * agents_per_env

    if model.recurrent:
        model.reset_states()

    iter = tqdm(range(episode_length)) if progress_bar else range(episode_length)
    for _ in iter:
        if all(dones): break

        obs_tensor = tf.convert_to_tensor(observations)

        # need to add time time dimension for recurrent models only
        if model.recurrent:
                obs_tensor = tf.expand_dims(obs_tensor, axis=1)

        action_logits, est_values = model(obs_tensor)

        # sample the action
        if model.recurrent:
            action_logits = tf.squeeze(action_logits, axis=1)
        actions = tf.squeeze(tf.random.categorical(action_logits, 1), axis=-1).numpy()

        # reshape the critic's value estimations
        # todo: im only like 90% sure that this is part is corect...
        squeeze_axes = [1, 2] if model.recurrent else 1
        values = list(tf.squeeze(est_values, axis=squeeze_axes).numpy())

        next_obs, rewards, next_dones, _ = env.step(list(map(to_action, actions)))

        if record:
            rollout.record(observations, actions, rewards, values, dones)
        dones = next_dones
        observations = next_obs

    return rollout


class Trainer:

    def __init__(self, get_env, hyperams: HyperParameters, to_action, test_env=None, training_dir=None):
        self.get_env = get_env
        self.hyperams = hyperams
        self.to_action = to_action
        self.test_env = test_env
        self.num_envs = hyperams.num_envs
        self.training_dir = training_dir

    def train(self, asynchronous=False):
        """ trains a model """
        if asynchronous:
            self._train_async()
        else:
            self._train_sync()

    def _train_sync(self):
        """ trains a model sequentially with a single environment """
        env = self.get_env()

        import tensorflow as tf
        from a2c.eager_models import make_model
        input_shape = (None,) + env.observation_space.shape
        model = make_model(self.hyperams.architecture,
                           self.hyperams.encoder_class,
                           input_shape,
                           self.hyperams.action_shape)

        self.optimizer = tf.keras.optimizers.Adam(lr=self.hyperams.learning_rate, clipnorm=1.0)

        summary_writer = None
        if self.training_dir is not None:
            model_directory = os.path.join(self.training_dir, "model")
            summary_writer = tf.summary.create_file_writer(self.training_dir)

        for ep in range(self.hyperams.num_episodes):
            logger.info(f"Episode {ep}")
            episode_length = self.hyperams.episode_length
            rollout = get_rollout(model, env,
                                  self.hyperams.agents_per_env,
                                  episode_length,
                                  self.to_action)

            losses = self._update_with_rollout(model, rollout.as_batch())
            self._log_rollout(summary_writer, ep, rollout.as_batch(), losses)

            if self.training_dir is not None and ep % self.hyperams.save_frequency == 0:
                logger.info("Checkpointing model...")
                model.save_weights(model_directory)

    def _train_async(self):
        """ trains a model asynchronously """

        model_directory = os.path.join(self.training_dir, "model")

        args = self.get_env, model_directory, self.to_action, self.hyperams
        coordinator = AsyncCoordinator(self.hyperams.num_envs,
                                       worker_target, args)

        with coordinator:
            import tensorflow as tf
            from a2c.eager_models import make_model

            summary_writer = tf.summary.create_file_writer(self.training_dir)

            input_shape = (None,) + self.get_env().observation_space.shape
            model = make_model(self.hyperams.architecture,
                               self.hyperams.encoder_class,
                               input_shape,
                               self.hyperams.action_shape)

            self.optimizer = tf.keras.optimizers.Adam(lr=self.hyperams.learning_rate, clipnorm=1.0)

            logger.info("Saving initial model...")
            model.save_weights(model_directory)
            logger.info("Initial model saved.")

            coordinator.start()  # start the worker processes

            for episode in range(self.hyperams.num_episodes):
                rollout_batch = coordinator.pop()

                losses = self._update_with_rollout(model, rollout_batch)
                self._log_rollout(summary_writer, episode, rollout_batch, losses=losses)

                model.save_weights(model_directory)

    def _train_async_shared(self):
        """ trains asynchronously with a single shared model
        todo: this is interesting but ultimately garbage... try make better someday?
        """
        # from a2c.shared_async_coordinator import AsyncCoordinatorShared
        # import threading
        # lock = threading.Lock()
        #
        # def get_actions(observations, carry):
        #     obs_in = tf.convert_to_tensor(observations)
        #     with lock:
        #         actions_t, values_t, carry = self.model.cell.action_value(obs_in, carry)
        #
        #     # convert Tensors to list of action-indices and values
        #     actions = actions_t.numpy()
        #     values = list(values_t.numpy())
        #
        #     return actions, values, carry
        #
        # def initialize():
        #     with lock:
        #         return self.model.cell.get_initial_state(batch_size=self.hyperams.agents_per_env,
        #                                            dtype=tf.float32)
        #
        # coordinator = AsyncCoordinatorShared(self.hyperams.num_envs, self.get_env,
        #                                self.hyperams.episode_length,
        #                                initialize, get_actions, self.to_action)
        #
        # with coordinator:
        #     for ep in range(self.hyperams.num_episodes):
        #         rollout = coordinator.await_rollout()
        #         self._log_rollout(rollout, ep)
        #         self._update_with_rollout(rollout)

    def _update_with_rollout(self, model, rollout_batch):
        """ updates the network using a roll-out """

        from a2c.losses import a2c_loss, get_loss_variables, batch_loss_variables

        loss_vars = get_loss_variables(rollout_batch, self.hyperams.gamma, model.recurrent)

        for loss_vars_batch in batch_loss_variables(loss_vars, self.hyperams.batch_size):
            losses, grads = a2c_loss(model, self.hyperams.entropy_weight, *loss_vars_batch)
            self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return losses

    def _log_rollout(self, summary_writer, episode, rollout_batch, losses=None):
        """ logs the performance of the roll-out """
        episode_length = len(rollout_batch[2][0])
        logger.info(f"Episode {episode}, length: {episode_length}")
        if losses is not None:
            logger.info(f"Actor loss: {losses[0]:.3f}, Critic loss: {losses[1]:.3f}")

        returns = []
        max_masses = []
        average_masses = []
        efficiencies = []

        for rewards in rollout_batch[2]:
            episode_return = rewards.sum()
            mass = 10 + rewards.cumsum()
            eff = get_efficiency(rewards, episode_length, self.hyperams)

            returns.append(episode_return)
            max_masses.append(mass.max())
            average_masses.append(mass.mean())
            efficiencies.append(eff)

        print(f"Average Ep Return:\t{np.mean(returns):.2f}")
        print(f"Average Max mass:\t{np.mean(max_masses):.2f}")
        print(f"Average Avg mass:\t{np.mean(average_masses):.2f}")
        print(f"Average efficiency:\t{np.mean(efficiencies):.2f}")

        if summary_writer is not None:  # in debug mode theres no directory to write to
            import tensorflow as tf
            with summary_writer.as_default():
                tf.summary.scalar('train/efficiency', np.mean(efficiencies), step=episode)
                if losses is not None:
                    tf.summary.scalar('loss/actor', losses[0], step=episode)
                    tf.summary.scalar('loss/critic', losses[1], step=episode)

    def _test(self, model, summary_writer=None, episode_length=None):
        logger.info(f"Testing performance...")
        episode_length = episode_length or self.hyperams.episode_length
        rollout = get_rollout(model, self.test_env, self.hyperams.agents_per_env,
                              episode_length, self.to_action)

        # todo: pass the real summary writer to log results to TensorBoard.
        self._log_rollout(summary_writer, "test", rollout.as_batch(), episode_length)


def get_efficiency(rewards, episode_length, hyperams):
    """ calculates the "agario mass efficiency", which is a quantity that i invented lol
    It is supposed to capture the rate at which mass was accumulated relative to
    the density of pellets in the arena. In this way, performance can be
    compared across episodes of different lengths, arenas of different sizes
    and with different numbers of pellets.
    """
    G = rewards.sum()
    pellet_density = hyperams.num_pellets / pow(hyperams.arena_size, 2)
    efficiency = G / (episode_length * pellet_density)
    return efficiency
