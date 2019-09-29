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
                  get_env, model_directory, to_action,
                  hyperams: HyperParameters):
    """ the task that each worker process performs: gather the complete
         roll-out of an episode using the latest model and send it back to the
         master process. """
    env = get_env()

    input_shape = (None,) + env.observation_space.shape
    model = make_model(input_shape,
                       hyperams.architecture,
                       hyperams.encoder_class,
                       hyperams.action_shape)

    sema.acquire()  # wait for master process to signal

    while True:
        model.load_weights(model_directory)

        print(f"Worker {wid} starting episode...")
        rollout = get_rollout(model, env,
                              hyperams.agents_per_env,
                              hyperams.episode_length,
                              to_action)
        print(f"Worker {wid} episode finished")
        queue.put(rollout)


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
        env = self.get_env()
        for ep in range(self.hyperams.num_episodes):
            episode_length = min(10 + ep, self.hyperams.episode_length)
            rollout = get_rollout(self.model, env,
                                  self.hyperams.agents_per_env,
                                  episode_length,
                                  self.to_action)
            self._log_rollout(rollout, ep, episode_length)
            self._update_with_rollout(rollout)

    def _train_async(self):
        """ trains a model asynchronously """

        model_directory = os.path.join(self.training_dir, "model")

        args = self.get_env, model_directory, self.to_action, self.hyperams
        coordinator = AsyncCoordinator(self.hyperams.num_envs,
                                       worker_target, args)

        with coordinator:
            import tensorflow as tf

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

            for ep in range(self.hyperams.num_episodes):
                rollout = coordinator.pop()
                self._log_rollout(rollout, ep, self.hyperams.episode_length)
                self._update_with_rollout(rollout)
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

    def _update_with_rollout(self, rollout):
        """ updates the network using a roll-out """
        import tensorflow as tf
        from a2c.losses import a2c_loss, make_returns_batch

        observations, actions, rewards, values, dones = rollout.as_batch()

        returns = make_returns_batch(rewards, self.hyperams.gamma)

        obs_batch = np.array(observations)
        act_batch = np.array(actions)
        ret_batch = np.array(returns)
        val_batch = np.array(values)
        adv_batch = ret_batch - val_batch
        mask = tf.convert_to_tensor(np.logical_not(np.array(dones)), dtype=tf.bool)

        loss_vars = obs_batch, mask, act_batch, adv_batch, ret_batch
        (a_loss_val, c_loss_val), grads = a2c_loss(self.model, *loss_vars)
        logger.info(f"Actor loss: {a_loss_val:.3f}, Critic loss: {c_loss_val:.3f}")

        logger.info(f"Applying gradients...")
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


    def _log_rollout(self, rollout, episode, episode_length):
        """ logs the performance of the roll-out """
        logger.info(f"Episode {episode}")

        for rewards in transpose_batch(rollout.rewards):
            G = rewards.sum()
            mass = rewards.cumsum() + 10
            eff = get_efficiency(rewards, episode_length, self.hyperams)
            print(f"Return:\t{G:.0f}, max mass:\t{mass.max():.0f}, avg. mass:\t{mass.mean():.1f}, efficiency:\t{eff:.1f}")


    def test(self, episode_length=None):
        return
        logger.info(f"Testing performance...")
        episode_length = episode_length or self.hyperams.episode_length
        rollout = get_rollout(self.model, self.test_env,
                              self.hyperams.agents_per_env,
                              episode_length, self.to_action)

        self._log_rollout(rollout, "test", episode_length)


def get_efficiency(rewards, episode_length, hyperams):
    """ calculates the agario efficiency, which is a made up quantity """
    G = rewards.sum()
    pellet_density = hyperams.num_pellets / pow(hyperams.arena_size, 2)
    efficiency = G / (episode_length * pellet_density)
    return efficiency