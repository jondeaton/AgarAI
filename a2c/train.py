"""
File: train
Date: 2019-07-25 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os, sys
import argparse, logging
import gym, gym_agario

import a2c
from a2c.hyperparameters import *


logger = logging.getLogger()

def make_environment(env_type, hyperams):
    """ makes and configures the specified OpenAI gym environment """
    env_config =  {
            'frames_per_step': hyperams.frames_per_step,
            'arena_size':      hyperams.arena_size,
            'num_pellets':     hyperams.num_pellets,
            'num_viruses':     hyperams.num_viruses,
            'num_bots':        hyperams.num_bots,
            'pellet_regen':    hyperams.pellet_regen,
        }

    if env_type == "screen":
        env_config["screen_len"] = hyperams.screen_len

    elif env_type == "grid":
        env_config.update({
            "grid_size":       hyperams.grid_size,
            "observe_cells":   hyperams.observe_cells,
            "observe_others":  hyperams.observe_others,
            "observe_viruses": hyperams.observe_viruses,
            "observe_pellets": hyperams.observe_pellets
        })

    logger.info(f"Creating Agar.io gym environment of type: {hyperams.env_name}")
    env = gym.make(hyperams.env_name, **env_config)
    return env


def main():
    args = parse_args()

    if args.env_type == "full":
        hyperams = FullEnvHyperparameters()
    elif args.env_type == "screen":
        hyperams = ScreenEnvHyperparameters()
    elif args.env_type == "grid":
        hyperams = GridEnvHyperparameters()
    else:
        raise ValueError(args.env_type)

    hyperams.override(args)

    output_dir = args.output
    os.makedirs(args.output, exist_ok=True)

    training_dir = get_training_dir(output_dir, args.name)
    os.makedirs(training_dir, exist_ok=True)
    logger.info(f"Model directory: {training_dir}")

    hp_file = os.path.join(training_dir, "hp.json")
    logger.debug(f"Saving hyper-parameters to: {hp_file}")
    hyperams.save(hp_file)

    env = make_environment(args.env_type, hyperams)

    trainer = a2c.Trainer(env, hyperams)

    trainer.train(25)


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
    env_options.add_argument("--env", default="full", choices=["full", "screen", "grid"], dest="env_type",
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
