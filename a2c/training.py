"""
File: training
Date: 2019-07-25 
Author: Jon Deaton (jdeaton@stanford.edu)

Inspired by:
    - http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
    - https://github.com/MG2033/A2C/
    
"""

from a2c.Model import ActorCritic
from a2c.hyperparameters import HyperParameters
from a2c.Coordinator import Coordinator

import logging
from datetime import datetime, timedelta
from tqdm import tqdm
import multiprocessing as mp

import numpy as np
import tensorflow as tf
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

logger = logging.getLogger("root")
logger.propagate = False


class Trainer:

    def __init__(self, get_env, hyperams: HyperParameters, training_dir=None):
        self.hyperams = hyperams
        self.set_seed(hyperams.seed)

        self.get_env = get_env

        self.num_envs = hyperams.num_envs
        self.envs = None

        self.model = ActorCritic(hyperams.action_shape)
        self.model.compile(optimizer=ko.RMSprop(lr=hyperams.learning_rate),
                           loss=[self._actor_loss, self._critic_loss])

        self.time_steps = 0
        self.episodes = 0
        self.gradient_steps = 0

        self.last_save = datetime.now()
        self.save_freq = timedelta(minutes=5)

        self.tensorboard = None
        if training_dir is not None:
            self.summary_writer = tf.summary.create_file_writer(training_dir)
            self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir=training_dir,
                                                              histogram_freq=1)

    def _critic_loss(self, returns, value):
        mse = kls.mean_squared_error(returns, value)
        return self.hyperams.params_value * mse

    def _actor_loss(self, acts_and_advs, logits):
        actions, advantages = tf.split(acts_and_advs, 2, axis=1)
        actions = tf.cast(actions, tf.int32)

        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)

        # entropy loss can be calculated via CE over itself
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        loss = policy_loss - self.hyperams.entropy_weight * entropy_loss

        return loss

    class Rollout:
        def __init__(self):
            self.observations = list()
            self.actions = list()
            self.rewards = list()
            self.values = list()
            self.dones = list()

        def to_batch(self):
            obs_batch = self.tranpose_batch(self.observations)
            action_batch = self.tranpose_batch(self.actions)
            reward_batch = self.tranpose_batch(self.rewards)
            value_batch = self.tranpose_batch(self.values)
            return obs_batch, action_batch, reward_batch, value_batch

        @staticmethod
        def tranpose_batch(collection):
            transposed = zip(*collection)
            to_arr = lambda rollout: np.array(list(filter(lambda x: x is not None, rollout)))
            return list(map(to_arr, transposed))

    def train(self, num_episodes=None):
        with Coordinator(self.get_env, self.num_envs) as self.envs:
            for _ in range(num_episodes or self.hyperams.num_episodes):
                rollout = self._rollout()
                o, a, r, v = rollout.to_batch()

                avg_reward = np.mean([rewards.sum() for rewards in r])
                std_reward = np.std([rewards.sum() for rewards in r])
                logger.info(f"Episode returns: {avg_reward:.2f} +/- {std_reward:.2f}")

                returns = self._make_returns(r, self.hyperams.gamma)

                obs_batch = np.concatenate(o)
                act_batch = np.concatenate(a)
                ret_batch = np.concatenate(returns)
                val_batch = np.concatenate(v)

                adv_batch = ret_batch - np.squeeze(val_batch)
                acts_and_advs = np.concatenate([act_batch[:, None], adv_batch[:, None]], axis=1)

                self.model.fit(obs_batch, [acts_and_advs, ret_batch],
                               batch_size=self.hyperams.batch_size,
                               shuffle=True, epochs=1)

    def _rollout(self):

        rollout = Trainer.Rollout()
        obs = self.envs.reset()
        done = [False] * self.num_envs

        for _ in tqdm(range(self.hyperams.episode_length)):
            obs_in = np.array(list(filter(lambda o: o is not None, obs)))
            actions_t, values_t= self.model.action_value(obs_in)

            actions = [None if d else self.to_action(a) for a, d in zip(actions_t, done)]

            obs, rs, done, _ = self.envs.step(actions)

            rollout.observations.append(obs)
            rollout.actions.append(actions_t)
            rollout.rewards.append(rs)
            rollout.values.append(values_t)
            rollout.dones.append(done)

            if all(done): break

        return rollout

    def to_action(self, index):
        """ converts a raw action index into an action shape """
        indices = np.unravel_index(index, self.hyperams.action_shape)
        theta = (2 * np.pi * indices[0]) / self.hyperams.action_shape[0]
        mag = 1 - indices[1] / self.hyperams.action_shape[1]
        act = indices[2]
        x = np.cos(theta) * mag
        y = np.sin(theta) * mag
        return x, y, act

    def _make_returns(self, reward_batch, gamma):
        returns_batch = list()

        for reward_rollout in reward_batch:
            return_rollout = np.zeros_like(reward_rollout)
            return_rollout[-1] = reward_rollout[-1]
            for i in reversed(range(len(reward_rollout) - 1)):
                return_rollout[i] = reward_rollout[i] + gamma * return_rollout[i + 1]
            returns_batch.append(return_rollout)

        return returns_batch

    def set_seed(self, seed):
        pass
