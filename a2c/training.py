"""
File: training
Date: 2019-07-25 
Author: Jon Deaton (jonpauldeaton@gmail.com)

Inspired by:
    - http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
    - https://github.com/MG2033/A2C/
    
"""

from a2c.hyperparameters import HyperParameters
from a2c.rollout import Rollout
from a2c.coordinator import Coordinator
from a2c.async_coordinator import AsyncCoordinator
from a2c.eager_models import ActorCritic

import logging
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko


logger = logging.getLogger("root")
logger.propagate = False


def make_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """ Calculates the discounted future returns for a single rollout
    :param rewards: numpy array containing rewards
    :param gamma: discount factor 0 < gamma < 1
    :return: numpy array containing discounted future returns
    """
    returns = np.zeros_like(rewards)

    ret = 0.0
    for i in reversed(range(len(rewards))):
        returns[i] = ret = rewards[i] + gamma * ret

    return returns


def make_returns_batch(reward_batch: List[np.ndarray], gamma: float) -> List[np.ndarray]:
    """ Calculates discounted episodes returns
    :param reward_batch: list of numpy arrays. Each numpy array is
    the episode rewards for a single episode in the batch
    :param gamma: discount factor 0 < gamma < 1
    :return: list of numpy arrays representing the returns
    """
    return [make_returns(rewards, gamma) for rewards in reward_batch]


def critic_loss(values_pred, returns):
    """ Critic loss defined as MSE for value estimation
    :param returns: tensor of discounted returns
    :param value: model-estimated future returns
    :return: tensor equal to MSE of
    """
    returns = tf.reshape(returns, [-1])
    values_pred = tf.reshape(values_pred, [-1])
    return kls.mean_squared_error(returns, values_pred)


def actor_loss(actions, advantages, action_logits):
    """ Standard advantage Policy gradient loss for Actor """

    # actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
    actions = tf.cast(tf.squeeze(actions), tf.int32)

    weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
    policy_loss = weighted_sparse_ce(actions, action_logits, sample_weight=advantages)

    # entropy loss may be calculated via CE over itself
    entropy_loss = kls.categorical_crossentropy(action_logits, action_logits, from_logits=True)
    return tf.reduce_mean(policy_loss - 1e-4 * entropy_loss)


def a2c_loss(model, observations, mask, actions, advantages, returns):
    """ computes the A2C loss and gradients of the given model w.r.t. that loss """
    with tf.GradientTape() as tape:
        action_logits, values_pred = model(observations, mask=mask, training=True)

        actor_loss_value = actor_loss(actions, advantages, action_logits)
        critic_loss_value = critic_loss(values_pred, returns)

        loss_value = [actor_loss_value, critic_loss_value]

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


class Trainer:

    def __init__(self, get_env, hyperams: HyperParameters, to_action, test_env=None, training_dir=None):
        self.get_env = get_env
        self.hyperams = hyperams
        self.to_action = to_action
        self.test_env = test_env

        self.num_envs = hyperams.num_envs
        # self.envs = None

        self.env = get_env()

        self.set_seed(hyperams.seed)

        # build the actor-critic network
        self.model = ActorCritic(hyperams.action_shape, hyperams.EncoderClass)

        self.optimizer = ko.Adam(lr=hyperams.learning_rate, clipnorm=1.0)

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

    def train(self, num_episodes=None):
        # with Coordinator(self.get_env, self.num_envs) as self.envs:

        # self.test()  # begin with benchmark

        num_episodes = num_episodes or self.hyperams.num_episodes
        for ep in range(num_episodes):
            rollout = self._rollout(ep, self.hyperams.episode_length)
            self.update_with_rollout(rollout)

            if ep and ep % 10 == 0:
                self.test()

    def train_async(self):

        import threading
        lock = threading.Lock()

        def get_actions(observations, carry):
            obs_in = tf.convert_to_tensor(observations)
            with lock:
                actions_t, values_t, carry = self.model.cell.action_value(obs_in, carry)

            # convert Tensors to list of action-indices and values
            actions = actions_t.numpy()
            values = list(values_t.numpy())

            return actions, values, carry

        def initialize():
            with lock:
                return self.model.cell.get_initial_state(batch_size=self.hyperams.agents_per_env,
                                                   dtype=tf.float32)

        coordinator = AsyncCoordinator(self.hyperams.num_envs, self.get_env,
                                       self.hyperams.episode_length,
                                       initialize, get_actions, self.to_action)

        with coordinator:
            for ep in range(self.hyperams.num_episodes):
                rollout = coordinator.await_rollout()
                self._log_rollout(rollout, ep)
                self.update_with_rollout(rollout)

    def update_with_rollout(self, rollout):
        """ updates the network using a rollout """
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

    def _rollout(self, episode, episode_length) -> Rollout:
        # performs a rollout and then trains the model on it
        rollout = Rollout()

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

    def _log_rollout(self, rollout, episode):
        """ logs the performance of the rollout """
        _, _, rewards, _, _ = rollout.as_batch()
        Gs = np.array([rs.sum() for rs in rewards])  # episode returns
        print(f"Episode {episode} returns: {Gs.min():.0f} min. {Gs.mean():.1f} ± {Gs.std():.0f} avg. {Gs.max():.0f} max.")

    def test(self):
        return
        logger.info(f"Testing performance...")
        o = self.test_env.reset()
        rewards = []
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
