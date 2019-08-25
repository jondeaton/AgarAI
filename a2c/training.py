"""
File: training
Date: 2019-07-25 
Author: Jon Deaton (jonpauldeaton@gmail.com)

Inspired by:
    - http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
    - https://github.com/MG2033/A2C/
    
"""

from a2c.Model import ActorCriticCell, ActorCritic
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

        # make the model
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
        returns = tf.reshape(returns, [-1])
        value = tf.reshape(value, [-1])
        return kls.mean_squared_error(returns, value)

    def _actor_loss(self, acts_and_advs, logits):
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        actions = tf.cast(tf.squeeze(actions), tf.int32)

        # advantages = tf.reshape(advantages, [-1])
        # actions = tf.reshape(actions, [-1])
        # logits = tf.reshape(logits, (None, 64))

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
            self.dones = list()

        def record(self, observations, actions, rewards, values, dones):
            self.observations.append(observations)
            self.actions.append(actions)
            self.rewards.append(rewards)
            self.values.append(values)
            self.dones.append(dones)

        def to_batch(self):
            obs_batch = self.transpose_batch(self.observations)
            action_batch = self.transpose_batch(self.actions)
            reward_batch = self.transpose_batch(self.rewards)
            value_batch = self.transpose_batch(self.values)
            dones = self.transpose_batch(self.dones)
            return obs_batch, action_batch, reward_batch, value_batch, dones

        @staticmethod
        def transpose_batch(collection):
            transposed = zip(*collection)
            to_arr = lambda rollout: np.array(list(none_filter(rollout)))
            return list(map(to_arr, transposed))

    def train(self, num_episodes=None):
        # with Coordinator(self.get_env, self.num_envs) as self.envs:

        # self.test()  # begin with benchmark

        num_episodes = num_episodes or self.hyperams.num_episodes
        for ep in range(num_episodes):
            self.rollout_train(ep)

            if ep and ep % 10 == 0:
                self.test()

    def rollout_train(self, episode: int):
        """ performs a single rollout of an episode followed
        by training self.model using the data from that rollout
        :param episode: the episode number
        :return: None
        """
        rollout = self._rollout(episode, self.hyperams.episode_length)
        observations, actions, rewards, values, dones = rollout.to_batch()

        Gs = np.array([rs.sum() for rs in rewards])  # episode returns
        print(f"Episode {episode} returns: {Gs.min():.0f} min. {Gs.mean():.1f} ± {Gs.std():.0f} avg. {Gs.max():.0f} max.")

        returns = self.make_returns(rewards, self.hyperams.gamma)

        # flatten list of lists into single batch for training
        # obs_batch = np.concatenate(observations)
        # act_batch = np.concatenate(actions)
        # ret_batch = np.concatenate(returns)
        # val_batch = np.concatenate(values)
        # mask = np.logical_not(np.concatenate(dones))

        # adv_batch = ret_batch - np.squeeze(val_batch)
        # act_adv_batch = np.concatenate([act_batch[:, None], adv_batch[:, None]], axis=1)
        # self.model.fit(obs_batch, [act_adv_batch, ret_batch],
        #                batch_size=self.hyperams.batch_size)

        obs_batch = np.array(observations)
        act_batch = np.array(actions)
        ret_batch = np.array(returns)
        val_batch = np.array(values)
        adv_batch = ret_batch - val_batch
        mask = np.logical_not(np.array(dones))

        act_adv_batch = np.stack([act_batch, adv_batch], axis=-1)

        inputs = (obs_batch, mask)
        self.model.train_on_batch(obs_batch, [act_adv_batch, ret_batch])

    def _rollout(self, episode, episode_length) -> Rollout:
        # performs a rollout and trains the model on it
        rollout = Trainer.Rollout()

        observations = self.env.reset()

        dones = [False] * self.hyperams.agents_per_env
        hc = self.model.cell.get_initial_state(batch_size=self.hyperams.agents_per_env,
                                               dtype=tf.float32)

        for t in tqdm(range(episode_length), desc=f"Episode {episode}"):
            if not all(dones):
                obs_in = tf.convert_to_tensor(observations)
                actions_t, values_t, hc = self.model.cell.action_value(obs_in, hc)

                # convert Tensors to list of action-indices and values
                actions = actions_t.numpy()
                act_in = list(map(self.to_action, actions))
                values = list(values_t.numpy())

                next_obs, rewards, dones, _ = self.env.step(act_in)

                rollout.record(observations, actions, rewards, values, dones)
                observations = next_obs

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
        return  # yikes
        logger.info(f"Testing performance...")
        o = self.test_env.reset()
        rewards = list()
        done = False
        lstm_state = np.zeros(32)
        for t in tqdm(range(self.hyperams.episode_length)):
            if not done:
                a, lstm_state = self.model.action(np.expand_dims(o, axis=0), lstm_state)
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
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.env.seed(seed)
