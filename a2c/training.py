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


def make_model(input_shape, architecture, encoder_class, action_shape):
    """ build the actor-critic network """
    from a2c.eager_models import get_encoder_type

    Encoder = get_encoder_type(encoder_class)

    if architecture == 'LSTM':
        from a2c.eager_models import LSTMAC
        model = LSTMAC(input_shape, action_shape, Encoder)

    elif architecture == 'ConvLSTM':
        from a2c.eager_models import ConvLSTMAC
        model = ConvLSTMAC(action_shape)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return model


def worker_target(wid: int, queue: Queue, sema: Semaphore,
                  get_env, model_directory, to_action, hyperams: HyperParameters):
    """ the task that each worker process performs: gather the complete
        roll-out of an episode using the latest model and send it back to the
        master process. """
    print(f"Worker {wid} started")

    # workers can't use GPU because TensorFlow...
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    env = get_env()

    input_shape = (None,) + env.observation_space.shape
    model = make_model(input_shape,
                       hyperams.architecture,
                       hyperams.encoder_class,
                       hyperams.action_shape)

    sema.acquire()  # wait for master process to signal

    while True:
        model.load_weights(model_directory)

        episode_length = random.randint(32, hyperams.episode_length)

        rollout = get_rollout(model, env,
                              hyperams.agents_per_env,
                              episode_length,
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

    model.reset_states()

    iter = tqdm(range(episode_length)) if progress_bar else range(episode_length)
    for _ in iter:
        if all(dones): break
        obs_tensor = tf.expand_dims(tf.convert_to_tensor(observations), axis=1)
        action_logits, est_values = model(obs_tensor)

        # sample the action
        actions = tf.squeeze(tf.random.categorical(tf.squeeze(action_logits, axis=1), 1), axis=-1).numpy()
        values = list(tf.squeeze(est_values, axis=[1, 2]).numpy())

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
        raise NotImplementedError()
        env = self.get_env()
        model = None # todo
        for ep in range(self.hyperams.num_episodes):
            episode_length = min(10 + ep, self.hyperams.episode_length)
            rollout = get_rollout(model, env,
                                  self.hyperams.agents_per_env,
                                  episode_length,
                                  self.to_action)
            self._log_rollout(rollout, ep, episode_length)
            self._update_with_rollout(model, rollout)

    def _train_async(self):
        """ trains a model asynchronously """

        model_directory = os.path.join(self.training_dir, "model")

        args = self.get_env, model_directory, self.to_action, self.hyperams
        coordinator = AsyncCoordinator(self.hyperams.num_envs,
                                       worker_target, args)

        with coordinator:
            import tensorflow as tf

            summary_writer = tf.summary.create_file_writer(self.training_dir)

            input_shape = (None,) + self.get_env().observation_space.shape
            model = make_model(input_shape,
                               self.hyperams.architecture,
                               self.hyperams.encoder_class,
                               self.hyperams.action_shape)

            self.optimizer = tf.keras.optimizers.Adam(lr=self.hyperams.learning_rate, clipnorm=1.0)

            logger.info("Saving initial model...")
            model.save_weights(model_directory)
            logger.info("Initial model saved.")

            coordinator.start()  # start the worker processes

            for episode in range(self.hyperams.num_episodes):
                rollout_batch = coordinator.pop()

                losses = self._update_with_rollout(model, rollout_batch)
                self._log_rollout(summary_writer, episode, rollout_batch, losses)

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
        import tensorflow as tf
        from a2c.losses import a2c_loss, make_returns_batch

        observations, actions, rewards, values, dones = rollout_batch

        returns = make_returns_batch(rewards, self.hyperams.gamma)

        obs_batch = np.array(observations)
        act_batch = np.array(actions)
        ret_batch = np.array(returns)
        val_batch = np.array(values)
        adv_batch = ret_batch - val_batch
        mask = tf.convert_to_tensor(np.logical_not(np.array(dones)), dtype=tf.bool)

        loss_vars = obs_batch, mask, act_batch, adv_batch, ret_batch
        (a_loss_val, c_loss_val), grads = a2c_loss(model, *loss_vars)
        logger.info(f"Actor loss: {a_loss_val:.3f}, Critic loss: {c_loss_val:.3f}")

        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return a_loss_val, c_loss_val

    def _log_rollout(self, episode, summary_writer, rollout_batch, losses):
        """ logs the performance of the roll-out """
        episode_length = len(rollout_batch[2][0])
        logger.info(f"Episode {episode}, length: {episode_length}")

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

        import tensorflow as tf
        with summary_writer.as_default():
            tf.summary.scalar('Average efficiency', np.mean(efficiencies), step=episode)
            tf.summary.scalar('Actor loss', losses[0], step=episode)
            tf.summary.scalar('Critic loss', losses[1], step=episode)

    def test(self, model, episode_length=None):
        return
        logger.info(f"Testing performance...")
        episode_length = episode_length or self.hyperams.episode_length
        rollout = get_rollout(model, self.test_env,
                              self.hyperams.agents_per_env,
                              episode_length, self.to_action)

        self._log_rollout(rollout, "test", episode_length)


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
