"""
File: train.py
Date: 5/6/19 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os, sys
import argparse, logging

logger = logging.getLogger()
from config import config
from log import tensorboard

import numpy as np

import gym
import gym_agario

from dqn.training import Trainer
from dqn.qn import QN, DQN
from dqn import HyperParameters

from features.extractors import FeatureExtractor
import torch

def main():
    args = parse_args()

    hyperams = HyperParameters()
    hyperams.override(args)

    output_dir = args.output
    os.makedirs(args.output, exist_ok=True)

    training_dir = get_training_dir(output_dir, args.name)
    os.makedirs(training_dir, exist_ok=True)
    logger.info(f"Model directory: {training_dir}")

    tensorboard.set_directory(training_dir)

    hp_file = os.path.join(training_dir, "hp.json")
    logger.debug(f"Saving hyper-parameters to: {hp_file}")
    hyperams.save(hp_file)

    logger.info("Creating Agar.io gym environment...")
    env = gym.make("agario-full-v0")

    extractor = FeatureExtractor(num_pellet=1, num_virus=0, num_food=0, num_other=0, num_cell=1)
    state_size = extractor.size
    action_size = np.prod(hyperams.action_shape)

    logger.info("Creating Q network...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q = DQN(state_size, action_size, p_dropout=hyperams.p_dropout, device=device)
    target_q = DQN(state_size, action_size, p_dropout=hyperams.p_dropout, device=device)

    logger.info("Training...")
    trainer = Trainer(env, q, target_q, hyperams=hyperams, extractor=extractor)
    trainer.train(num_episodes=hyperams.num_episodes, training_dir=training_dir)
    logger.info("Exiting.")


def get_training_dir(output_dir, name):
    """
    finds a suitable subdirectory within `output_dir` to
    save files from this run named `name`.
    :param output_dir: global output directory
    :param name: name of this run
    :return: path to file of the form /path/to/output/name-X
    """
    base = os.path.join(output_dir, name)
    i = 0
    while os.path.exists("%s-%03d" % (base, i)):
        i += 1
    return "%s-%03d" % (base, i)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN Model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    output_options = parser.add_argument_group("Output")
    output_options.add_argument("--output", default="model_outputs", help="Output directory")
    output_options.add_argument("--name", default="dqn",
                                help="Experiment or run name")

    hyperams_options = parser.add_argument_group("HyperParameters")
    # note: make sure that the "dest" value is exactly the same as the variable name in "Hyperparameters"
    # in order for over-riding to work correctly.
    hyperams_options.add_argument("-episodes", "--episodes", dest="num_episodes", type=int,
                                  help="Number of epochs to train")

    training_options = parser.add_argument_group("Training")
    training_options.add_argument("-gpu", "--gpu", action='store_true', help="Enable GPU")

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
