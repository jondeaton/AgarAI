"""
File: sarsa.py
Date: 2019-08-09
Author: Jon Deaton (jonpdeaton@gmail.com))
"""

import gym
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


import sys, argparse, logging
logger = logging.getLogger("root")


def to_obs(s: np.ndarray) -> np.ndarray:
    x, dx, t, dt = s
    v = np.array([x, dx, t, dt, np.sin(t), np.cos(t), np.sin(dt), np.cos(dt)])
    return np.concatenate(v, np.outer(v, v).reshape(-1))


def evaluate(pi, N=1000):
    returns = np.zeros(N)
    
    env = gym.make("CartPole-v1")
    for ep in range(N):
        s = env.reset()
        done = False
        G = 0
        while not done:
            o = to_obs(s)
            a = pi(o)
            s, r, done, _ = env.step(a)
            G += r
        returns[ep] = G

    logger.info(f"returns: {returns.mean():.2f} +/- {returns.std():.2f}")


def main():
    args = parse_args()

    np.random.seed(42)
    random.seed(42)

    env = gym.make(args.env_name)
    env.seed(42)

    state_n,  = env.observation_space.shape
    action_n = env.action_space.n

    obs_n = 8 + 8 * 8

    w_shape = (action_n, obs_n)
    w = np.zeros(w_shape)
    #w = np.random.randn(np.prod(w_shape)).reshape(w_shape)
    eps_base = 0.5
    lr_base = 0.00025

    gamma = 0.99

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

        # lambda: 0 => one-step SARSA (more bias, less variance)
        # lambda: 1 => Monte Carlo (less bias, more variance)
        lambd = 0.95

        H = len(actions)
        qs = np.zeros(H)
        for t in range(H - 1):
            qs[t] = np.matmul(w, states[t])[actions[t]]

        qs = np.append(qs[1:], 0)
        rewards = np.array(rewards)
        delta = rewards + gamma * (1 - lambd) * qs

        q_l = np.zeros_like(delta)
        q_l[-1] = delta[-1]

        for t in reversed(range(H - 1)):
            q_l[t] = delta[t] + gamma * lambd * q_l[t + 1]

        for t in reversed(range(H)):
            a = actions[t]
            s = states[t]
            Qsa = np.matmul(w, s)[a]
            w[a] += lr * ( (q_l[t] - Qsa) * s - 0.002 * w[a])

    pi = lambda o: np.argmax(np.matmul(w, o))
    evaluate(pi)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SARSA Agent")

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
