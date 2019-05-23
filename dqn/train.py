"""
File: train.py
Date: 5/6/19 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os, sys
import argparse, logging

logger = logging.getLogger()
from log import tensorboard

import numpy as np

import gym
import gym_agario

from dqn.training import Trainer
from dqn.qn import DQN, DuelingDQN, StateEncoder, ConvEncoder
from dqn import HyperParameters
from dqn.hyperparameters import  FullEnvHyperparameters, ScreenEnvHyperparameters

import torch


def make_q_networks(hyperams: HyperParameters, state_shape):
    """ creates an online and target Q network
    :param hyperams: hyper-parameters
    :param state_shape: shape of the state in put into the networks
    :return: tuple containing the online and target Q networks
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if hyperams.encoder_type == 'linear':
        encoder        = StateEncoder(state_shape, hyperams.layer_sizes, p_dropout=hyperams.p_dropout, device=device)
        target_encoder = StateEncoder(state_shape, hyperams.layer_sizes, p_dropout=hyperams.p_dropout, device=device)

    elif hyperams.encoder_type == 'cnn':
        encoder        = ConvEncoder(state_shape, device=device)
        target_encoder = ConvEncoder(state_shape, device=device)

    else:
        raise ValueError(f"Unknown encoder type: {hyperams.encoder_type}")

    network = DuelingDQN if hyperams.dueling_dqn else DQN
    action_size = np.prod(hyperams.action_shape)

    # make the Q networks on top of the encoder
    q        = network(encoder,        action_size, device=device)
    target_q = network(target_encoder, action_size, device=device)

    return q, target_q


def get_feature_extractor(hyperams: HyperParameters):
    """ creates a feature extractor object for the given environment
    :param hyperams: hyper-parameters object
    :return: a feature extractor to extract feature vectors from states
    """
    if hyperams.extractor_type == "full":
        assert isinstance(hyperams, FullEnvHyperparameters)
        from features.extractors import FeatureExtractor
        extractor = FeatureExtractor(num_pellet = hyperams.num_pellets_features,
                                     num_virus  = hyperams.num_viruses_features,
                                     num_food   = hyperams.num_food_features,
                                     num_other  = hyperams.num_other_features,
                                     num_cell   = hyperams.num_cell_features)

    elif hyperams.extractor_type == "grid":
        assert isinstance(hyperams, FullEnvHyperparameters)
        from features.extractors import GridFeatureExtractor
        extractor = GridFeatureExtractor(hyperams.ft_extractor_view_size,
                                         hyperams.ft_extractor_grid_size,
                                         hyperams.arena_size,
                                         grid_shaped=hyperams.ft_grid_shaped)

    elif hyperams.extractor_type == "screen":
        assert isinstance(hyperams, ScreenEnvHyperparameters)
        from features.extractors import ScreenFeatureExtractor
        extractor = ScreenFeatureExtractor(hyperams.frames_per_step, hyperams.screen_len)

    elif hyperams.extractor_type is None:
        extractor = None

    else:
        raise ValueError(f"Unknown extractor type: {hyperams.extractor_type}")

    return extractor


def main():
    args = parse_args()

    hyperams = FullEnvHyperparameters() if args.env_type == "full" else ScreenEnvHyperparameters()
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

    env_config =  {
            'frames_per_step': hyperams.frames_per_step,
            'arena_size':      hyperams.arena_size,
            'num_pellets':     hyperams.num_pellets,
            'num_viruses':     hyperams.num_viruses,
            'num_bots':        hyperams.num_bots,
            'pellet_regen':    hyperams.pellet_regen,
        }
    if args.env_type == "screen":
        env_config["screen_len"] = hyperams.screen_len

    logger.info(f"Creating Agar.io gym environment of type: {hyperams.env_name}")
    env = gym.make(hyperams.env_name, **env_config)

    extractor = get_feature_extractor(hyperams)
    state_shape = extractor.shape

    logger.info("Creating Q network...")
    q, target_q = make_q_networks(hyperams, state_shape)

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

    env_options = parser.add_argument_group("Environment")
    env_options.add_argument("--env", default="full", choices=["full", "screen"], dest="env_type",
                             help="Environment type")

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
