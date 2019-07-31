"""
File: training
Date: 2019-07-25 
Author: Jon Deaton (jdeaton@stanford.edu)

Inspired by:
    - http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
    - https://github.com/MG2033/A2C/
    
"""


from a2c.hyperparameters import HyperParameters
from a2c.Coordinator import Coordinator

import multiprocessing as mp

from datetime import datetime, timedelta

import tensorflow as tf
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

import numpy as np

from tqdm import tqdm


class Trainer:

    def __init__(self, get_env, model, hyperams: HyperParameters):
        self.hyperams = hyperams

        self.model = model
        self.model.compile(optimizer=ko.Adam(lr=hyperams.lr),
                           loss=[self._actor_loss, self._critic_loss])

        self.time_steps = 0
        self.episodes = 0
        self.gradient_steps = 0

        self.last_save = datetime.now()
        self.save_freq = timedelta(minutes=5)

        self.set_seed(hyperams.seed)

        self.num_envs = mp.cpu_count()
        self.envs = Coordinator(get_env, self.num_envs)

    def _critic_loss(self, returns, value):
        mse = kls.mean_squared_error(returns, value)
        return self.hyperams.params_value * mse

    def _actor_loss(self, acts_and_advs, logits):
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)

        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)

        # entropy loss can be calculated via CE over itself
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)

        return policy_loss - self.hyperams.entropy_weight * entropy_loss

    def train(self, num_batches):

        observations = list()
        actions = list()
        rewards = list()
        values = list()
        dones = list()

        all_done = False

        for _ in tqdm(range(num_batches)):

            obs = self.envs.reset()
            while not all_done:
                actions_t, values_t = self.model.action_value(obs)
                a = actions_t.numpy()
                obs, rs, done, _ = self.envs.step(a)
                all_done = all(done)

                observations.append(obs)
                actions.append(a)
                rewards.append(rs)
                values.append(values_t.numpy())
                dones.append(done)

                # todo: transpose first two dimensions of these
                # todo: get advantages/returns
                acts_and_advs = None
                returns = None
                losses = self.model.train_on_batch(observations, [acts_and_advs, returns])

    def _returns_advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.hyperams.gamma * returns[t+1] * (1-dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages

    def set_seed(self, seed):
        pass