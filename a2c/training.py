"""
File: training
Date: 2019-07-25 
Author: Jon Deaton (jdeaton@stanford.edu)

Inspired by:
    - http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
    - https://github.com/MG2033/A2C/
    
"""

def is_None(x):
    return x is None

def is_not_None(x):
    return x is not None

def none_filter(l):
    return filter(is_not_None, l)

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

    def __init__(self, get_env, hyperams: HyperParameters, to_action, training_dir=None):
        self.get_env = get_env
        self.hyperams = hyperams
        self.to_action = to_action

        self.num_envs = hyperams.num_envs
        # self.envs = None

        self.env = get_env()

        self.set_seed(hyperams.seed)
        self.model = ActorCritic(hyperams.action_shape, hyperams.EncoderClass)
        self.model.compile(optimizer=ko.Adam(lr=hyperams.learning_rate, clipnorm=1),
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
        return kls.mean_squared_error(returns, value)

    def _actor_loss(self, acts_and_advs, logits):
        actions, advantages = tf.split(acts_and_advs, 2, axis=1)
        actions = tf.cast(actions, tf.int32)

        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)

        # entropy loss can be calculated via CE over itself
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        return policy_loss - self.hyperams.entropy_weight * entropy_loss

    class Rollout:
        def __init__(self):
            self.observations = list()
            self.actions = list()
            self.rewards = list()
            self.values = list()

        def record(self, observations, actions, rewards, values):
            self.observations.append(observations)
            self.actions.append(actions)
            self.rewards.append(rewards)
            self.values.append(values)

        def to_batch(self):
            obs_batch = self.tranpose_batch(self.observations)
            action_batch = self.tranpose_batch(self.actions)
            reward_batch = self.tranpose_batch(self.rewards)
            value_batch = self.tranpose_batch(self.values)
            return obs_batch, action_batch, reward_batch, value_batch

        @staticmethod
        def tranpose_batch(collection):
            transposed = zip(*collection)
            to_arr = lambda rollout: np.array(list(none_filter(rollout)))
            return list(map(to_arr, transposed))

    def train(self, num_episodes=None):
        # with Coordinator(self.get_env, self.num_envs) as self.envs:
        for _ in range(num_episodes or self.hyperams.num_episodes):
            self.rollout_train()

    def rollout_train(self):
        rollout = self.rollout_(self.hyperams.episode_length)
        observations, actions, r, values = rollout.to_batch()

        avg_reward = np.mean([rewards.sum() for rewards in r])
        std_reward = np.std([rewards.sum() for rewards in r])
        logger.info(f"Episode returns: {avg_reward:.2f} +/- {std_reward:.2f}")

        returns = self.make_returns(r, self.hyperams.gamma)

        obs_batch = np.concatenate(observations)
        act_batch = np.concatenate(actions)
        ret_batch = np.concatenate(returns)
        val_batch = np.concatenate(values)

        adv_batch = ret_batch - np.squeeze(val_batch)
        act_adv_batch = np.concatenate([act_batch[:, None], adv_batch[:, None]], axis=1)

        self.model.fit(obs_batch, [act_adv_batch, ret_batch],
                       shuffle=True, batch_size=self.hyperams.batch_size)


    def rollout_(self, episode_length):
        # performs a rollout and trains the model on it
        rollout = Trainer.Rollout()

        observations = self.env.reset()

        dones = [False] * self.hyperams.agents_per_env
        for _ in tqdm(range(episode_length)):
            obs_in = np.array(list(filter(is_not_None, observations)))
            actions_t, values_t = self.model.action_value(obs_in)

            actions = list()
            values = list()
            j = 0
            for i in range(self.hyperams.agents_per_env):
                if not dones[i]:
                    actions.append(actions_t[j])
                    values.append(values_t[j])
                    j += 1
                else:
                    actions.append(None)
                    values.append(None)

            next_observations, rewards, dones, _ = self.env.step(list(map(self.to_action, actions)))

            rollout.record(observations, actions, rewards, values)
            observations = next_observations
            if all(dones): break

        return rollout

    @staticmethod
    def make_returns(reward_batch, gamma):
        returns_batch = list()

        for reward_rollout in reward_batch:
            return_rollout = np.zeros_like(reward_rollout)
            return_rollout[-1] = reward_rollout[-1]
            for i in reversed(range(len(reward_rollout) - 1)):
                return_rollout[i] = reward_rollout[i] + gamma * return_rollout[i + 1]
            returns_batch.append(return_rollout)

        return returns_batch

    def set_seed(self, seed):
        pass # todo :(
