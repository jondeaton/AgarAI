"""
File: actor_critic
Date: 2019-08-12 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os, sys
import random
import gym
import numpy as np
from tqdm import tqdm

import logging, argparse
logger = logging.getLogger("root")

import tensorflow as tf

import tensorflow.keras.losses as kls
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko


class ActorCritic(tf.keras.Model):
    def __init__(self, action_size):
        super(ActorCritic, self).__init__()

        self.dense1 = kl.Dense(128, activation=tf.nn.leaky_relu)
        self.dense2 = kl.Dense(128, activation=tf.nn.leaky_relu)
        self.dense3 = kl.Dense(128, activation=tf.nn.leaky_relu)
        self.dropout = kl.Dropout(0.1)

        self.value_layer = kl.Dense(1, activation=None, name="value")
        self.action_layer = kl.Dense(action_size, activation=None, name="action")

    def call(self, x):
        x = self.dropout(self.dense1(x))
        x = self.dropout(self.dense2(x))
        # x = self.dropout(self.dense3(x))

        values = self.value_layer(x)
        action_logits = self.action_layer(x)
        return action_logits, values

    def action_value(self, state: np.ndarray):
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        logits, value = self.predict(state)
        action = tf.random.categorical(logits, 1)
        return tf.squeeze(action, axis=-1), np.squeeze(value, axis=1)


def critic_loss(returns, value):
    return kls.mean_squared_error(returns, value)

def actor_loss(acts_and_advs, logits):
    actions, advantages = tf.split(acts_and_advs, 2, axis=1)
    actions = tf.cast(actions, tf.int32)

    weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
    policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)

    # entropy loss can be calculated via CE over itself
    entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
    return policy_loss - 1e-4 * entropy_loss


def main():
    args = parse_args()

    np.random.seed(42)
    random.seed(42)

    env = gym.make(args.env_name)
    env.seed(42)

    state_n,  = env.observation_space.shape
    action_n = env.action_space.n

    lr = 0.001
    gamma = 0.99

    ep_returns = np.zeros(args.num_episodes)

    model = ActorCritic(action_n)
    model.compile(optimizer=ko.Adam(lr=lr),
                  loss=[actor_loss, critic_loss])

    episode_iterator = tqdm(range(args.num_episodes), unit="Episode")
    for ep in episode_iterator:
        s = env.reset()

        states = [s]
        rewards = list()
        actions = list()
        values = list()

        G = 0  # return
        done = False
        while not done:
            action, value = model.action_value(s)
            a = int(action)
            v = int(value)
            s, r, done, info = env.step(a)
            G += r

            if not done:
                states.append(s)
            rewards.append(r)
            actions.append(a)
            values.append(v)

        ep_returns[ep] = G
        episode_iterator.set_description(f"return: {G}")

        states, rewards, actions, values = map(np.array,
                                               (states, rewards, actions, values))

        # make returns
        returns = np.zeros_like(rewards)
        returns[-1] = rewards[-1]
        for t in reversed(range(len(rewards) - 1)):
            returns[t] = rewards[t] + gamma * returns[t + 1]

        advs = returns - values
        acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=1)

        model.train_on_batch(states, [acts_and_advs, returns])


def parse_args():
    parser = argparse.ArgumentParser(description="Train Actor Critic Agent")

    env_options = parser.add_argument_group("Environment")
    env_options.add_argument("--env", default="CartPole-v1",
                             dest="env_name",
                             choices=["CartPole-v1"])

    hyperams_options = parser.add_argument_group("HyperParameters")
    hyperams_options.add_argument("--episodes", type=int, default=4000,
                                  dest="num_episodes",
                                  help="Number of epochs to train")

    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument('--log', dest="log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                               default="DEBUG", help="Logging level")
    args = parser.parse_args()

    # Setup the logger

    # Logging level configuration
    log_level = getattr(logging, args.log_level.upper())
    log_formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')

    # For the console
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    logger.setLevel(log_level)

    return args


if __name__ == "__main__":
    main()
