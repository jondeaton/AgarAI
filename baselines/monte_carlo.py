"""
File: monte_carlo
Date: 2019-08-07
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import gym
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


import sys, argparse, logging
logger = logging.getLogger("root")


def to_obs(s: np.ndarray) -> np.ndarray:
    theta = s[2]
    theta_dot = s[3]

    y = np.sin(theta)
    x = np.cos(theta)
    dy = np.sin(theta_dot)
    dx = np.cos(theta_dot)

    o = np.append(s, [x, y, dx, dy])
    return np.append(o, np.outer(o, o).reshape(-1))


def main():
    args = parse_args()

    np.random.seed(42)
    random.seed(42)

    env = gym.make(args.env_name)
    env.seed(42)

    state_n,  = env.observation_space.shape
    action_n = env.action_space.n

    obs_n = ((state_n + 4) + 1) * (state_n + 4)

    w_shape = (action_n, obs_n)
    w = np.random.randn(np.prod(w_shape)).reshape(w_shape)
    eps_base = 0.5
    lr_base = 0.00025

    gamma = 0.95

    ep_returns = np.zeros(args.num_episodes)

    episode_iterator = tqdm(range(args.num_episodes), unit="Episode")
    for ep in episode_iterator:
        epsilon = eps_base * (1 / (1 + ep / 500))
        lr = lr_base + 0.007 * np.exp(- ep / 1000.0)

        s = env.reset()

        rewards = list()
        actions = list()
        states = list()

        G = 0  # return
        done = False
        while not done:
            o = to_obs(s)

            if np.random.rand() <= epsilon:
                a = env.action_space.sample()
            else:
                a = np.argmax(np.matmul(w, o))

            s, r, done, info = env.step(a)
            G += r

            rewards.append(r)
            actions.append(a)
            states.append(o)

        episode_iterator.set_description(f"return: {G}, epsilon: {epsilon:.3f}")
        ep_returns[ep] = G

        # make returns
        returns = np.zeros(len(rewards))
        returns[-1] = rewards[-1]
        for i in reversed(range(len(rewards) - 1)):
            returns[i] = rewards[i] + gamma * returns[i + 1]

        # update
        for i in range(len(returns)):
            a = actions[i]
            G = returns[i]
            s = states[i]
            w[a] += lr * ((G - np.matmul(w, s)[a]) * s - 0.002 * w[a])

    plt.figure()
    plt.plot(ep_returns, 'blue')
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Train Q-Learning Agent")

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
