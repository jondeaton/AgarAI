"""
File: q_learning
Date: 2019-08-07 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import gym
import numpy as np
from tqdm import tqdm
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

    env = gym.make(args.env_name)
    np.random.seed(42)
    env.seed(42)

    state_n,  = env.observation_space.shape
    action_n = env.action_space.n

    obs_n = (state_n + 4) + (state_n + 4) * (state_n + 4)

    w_shape = (action_n, obs_n)
    w1 = np.random.randn(np.prod(w_shape)).reshape(w_shape)
    if args.double:
        w2 = np.copy(w1)

    epsilon = 0.5
    lr = 0.001
    gamma = 1.0

    returns = np.zeros(args.num_episodes)

    episode_iterator = tqdm(range(args.num_episodes), unit="Episode")
    for ep in episode_iterator:

        s = env.reset()
        o = to_obs(s)
        G = 0  # return

        done = False
        while not done:
            if np.random.rand() <= epsilon:
                a = env.action_space.sample()
            else:
                if args.double:
                    qsa = np.matmul(w1, o) + np.matmul(w2, o)
                else:
                    qsa = np.matmul(w1, o)

                a = np.argmax(qsa)

            sp, r, done, info = env.step(a)
            op = to_obs(sp)

            G += r

            # W update
            if args.double:
                if np.random.rand() < 0.5:
                    w1, w2 = w2, w1

                target = r
                if not done:
                    ap = np.argmax(np.matmul(w1, op))
                    target += gamma * np.matmul(w2, op)[ap]

                dw = o
                dwa = (target - np.matmul(w1, o)[a]) * dw
                w1[a] += lr * (dwa - 0.002 * w1[a])

            else:
                target = r + (0 if done else gamma * np.max(np.matmul(w1, op)))
                dw = o
                w1[a] += lr *((target - np.matmul(w1, o)[a]) * dw - 0.002 * w1[a])

            o = op  # advance

        epsilon = 0.9 * (1 / (1 + ep / 500))
        # lr *= 0.999

        episode_iterator.set_description(f"return: {G}")
        returns[ep] = G

def parse_args():
    parser = argparse.ArgumentParser(description="Train Q-Learning Agent")

    env_options = parser.add_argument_group("Environment")
    env_options.add_argument("--env", default="CartPole-v1",
                             dest="env_name",
                             choices=["CartPole-v1"])

    hyperams_options = parser.add_argument_group("HyperParameters")
    hyperams_options.add_argument("--double", action='store_true', help="Double Q learning")
    hyperams_options.add_argument("--episodes", type=int, default=4000,
                                  dest="num_episodes",
                                  help="Number of epochs to train")

    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument('--log', dest="log_level",
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
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
