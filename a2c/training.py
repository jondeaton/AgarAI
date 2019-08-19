"""
File: training
Date: 2019-07-25 
Author: Jon Deaton (jonpauldeaton@gmail.com)

Inspired by:
    - http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
    - https://github.com/MG2033/A2C/
    
"""

from a2c.Model import ActorCritic
from a2c.hyperparameters import HyperParameters
from a2c.Coordinator import Coordinator
from utils import *

import logging
from datetime import datetime, timedelta
from tqdm import tqdm

import numpy as np

import tensorflow as tf
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

logger = logging.getLogger("root")
logger.propagate = False

class Trainer:

    def __init__(self, get_env, hyperams: HyperParameters, to_action,
                 test_env=None,
                 training_dir=None):
        self.get_env = get_env
        self.hyperams = hyperams
        self.to_action = to_action
        self.test_env = test_env

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

        self.test()  # begin with benchmark

        num_episodes = num_episodes or self.hyperams.num_episodes
        for ep in range(num_episodes):
            self.rollout_train(ep)

            if ep and ep % 10 == 0:
                self.test()

    def rollout_train(self, episode):
        rollout = self.rollout_(episode, self.hyperams.episode_length)
        observations, actions, rewards, values = rollout.to_batch()

        Gs = np.array([rs.sum() for rs in rewards])
        print(f"Episode {episode} returns: {Gs.min():.0f} min. {Gs.mean():.1f} ± {Gs.std():.0f} avg. {Gs.max():.0f} max.")

        returns = self.make_returns(rewards, self.hyperams.gamma)

        # flatten list of lists into single batch
        obs_batch = np.concatenate(observations)
        act_batch = np.concatenate(actions)
        ret_batch = np.concatenate(returns)
        val_batch = np.concatenate(values)

        adv_batch = ret_batch - np.squeeze(val_batch)
        act_adv_batch = np.concatenate([act_batch[:, None], adv_batch[:, None]], axis=1)

        n, *_ = obs_batch.shape

        self.model.fit(obs_batch, [act_adv_batch, ret_batch],
                       shuffle=True, batch_size=self.hyperams.batch_size)

    def rollout_(self, episode, episode_length):
        # performs a rollout and trains the model on it
        rollout = Trainer.Rollout()

        observations = self.env.reset()

        dones = [False] * self.hyperams.agents_per_env
        for t in tqdm(range(episode_length), desc=f"Episode {episode}"):
            if not all(dones):
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

    def test(self):
        logger.info(f"Testing performance...")
        o = self.test_env.reset()
        rewards = list()
        done = False
        for t in tqdm(range(self.hyperams.episode_length)):
            if not done:
                a = self.model.action(np.expand_dims(o, axis=0))
                o, r, done, _ = self.test_env.step(self.to_action(a))
                rewards.append(r)

        episode_len = len(rewards)
        rewards = np.array(rewards)
        G = rewards.sum()
        mass = rewards.cumsum() + 10

        pellet_density = self.hyperams.num_pellets / pow(self.hyperams.arena_size, 2)
        efficiency = G / (episode_len * pellet_density)

        print(f"Return: {G:.0f}, max mass: {mass.max():.0f}, avg. mass: {mass.mean():.1f}, efficiency: {efficiency:.1f}")

    def set_seed(self, seed):
        self.env.seed(seed)
