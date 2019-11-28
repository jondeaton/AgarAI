"""
File: train
Date: 2019-07-25 
Author: Jon Deaton (jonpauldeaton@gmail.com)
"""
import os, sys
import argparse, logging
import gym, gym_agario
import numpy as np

from a2c.training import Trainer

logger = logging.getLogger("root")
logger.propagate = False


def make_test_env(env_name, hyperams):
    """ creates an environment for testing """
    return gym.make(env_name, **{
        'num_agents': 8,
        'difficulty': 'normal',
        'ticks_per_step': hyperams.ticks_per_step,
        'arena_size': 500,
        'num_pellets': 1000,
        'num_viruses': 25,
        'num_bots': 25,
        'pellet_regen': True,

        "grid_size": hyperams.grid_size,
        "observe_cells": hyperams.observe_cells,
        "observe_others": hyperams.observe_others,
        "observe_viruses": hyperams.observe_viruses,
        "observe_pellets": hyperams.observe_pellets
    })


def make_environment(env_name, hyperams):
    """ makes and configures the specified OpenAI gym environment """

    env_config = dict()

    if env_name == "agario-grid-v0":
        env_config = {
                'num_agents':      hyperams.agents_per_env,
                'difficulty':      hyperams.difficulty,
                'ticks_per_step':  hyperams.ticks_per_step,
                'arena_size':      hyperams.arena_size,
                'num_pellets':     hyperams.num_pellets,
                'num_viruses':     hyperams.num_viruses,
                'num_bots':        hyperams.num_bots,
                'pellet_regen':    hyperams.pellet_regen,
            }

        # observation parameters
        env_config.update({
            "grid_size":       hyperams.grid_size,
            "observe_cells":   hyperams.observe_cells,
            "observe_others":  hyperams.observe_others,
            "observe_viruses": hyperams.observe_viruses,
            "observe_pellets": hyperams.observe_pellets
        })

    env = gym.make(env_name, **env_config)
    return env


def agario_to_action(index, action_shape):
    """ converts a raw action index into an Agario action """
    if index is None:
        return None
    if type(index) is not int:
        index = int(index)
    indices = np.unravel_index(index, action_shape)
    theta = (2 * np.pi * indices[0]) / action_shape[0]
    mag = 1 - indices[1] / action_shape[1]
    act = int(indices[2])
    x = np.cos(theta) * mag
    y = np.sin(theta) * mag
    return np.array([x, y]), act


def main():
    args = parse_args()
    setup_logger(args, logger)

    if args.env == "CartPole-v1":
        from a2c.hyperparameters import CartPoleHyperparameters
        hyperams = CartPoleHyperparameters()
        to_action = lambda index: index
    elif args.env == "agario-grid-v0":
        from a2c.hyperparameters import GridEnvHyperparameters
        hyperams = GridEnvHyperparameters()
        to_action = lambda i: agario_to_action(i, hyperams.action_shape)
    else:
        raise ValueError(args.env)

    hyperams.override(args)
    logger.debug(f"Environment: {args.env}")

    if args.debug:
        logger.warning(f"Debug mode on. Model will not be saved")
        training_dir = None
    else:
        output_dir = args.output
        os.makedirs(args.output, exist_ok=True)

        training_dir = get_training_dir(output_dir, args.name)
        os.makedirs(training_dir, exist_ok=True)
        logger.info(f"Model directory: {training_dir}")

        hp_file = os.path.join(training_dir, "hp.json")
        logger.debug(f"Saving hyper-parameters to: {hp_file}")
        hyperams.save(hp_file)

    get_env = lambda: make_environment(args.env, hyperams)

    test_env = make_test_env(args.env, hyperams)

    trainer = Trainer(get_env, hyperams, to_action, test_env=test_env, training_dir=training_dir)

    trainer.train(asynchronous=hyperams.asynchronous)

    logger.debug("Exiting.")


def get_training_dir(output_dir, name):
    """
    finds a suitable subdirectory within `output_dir` to
    save files from this run named `name`.
    :param output_dir: global output directory
    :param name: name of this run
    :return: path to file of the form /path/to/output/name-X
    """
    if output_dir is None:
        return None
    base = os.path.join(output_dir, name)
    i = 0
    while os.path.exists("%s-%03d" % (base, i)):
        i += 1
    return "%s-%03d" % (base, i)


def parse_args():
    parser = argparse.ArgumentParser(description="Train A2C Agent")

    env_options = parser.add_argument_group("Environment")
    env_options.add_argument("--env", default="agario-grid-v0",
                             choices=["agario-grid-v0", "CartPole-v1",
                                      "RoboschoolHalfCheetah-v1"])

    output_options = parser.add_argument_group("Output")
    output_options.add_argument("--output", default="model_outputs", help="Output directory")
    output_options.add_argument("--name", default="a2c", help="Experiment or run name")
    output_options.add_argument("--debug", action="store_true", help="Debug mode")

    hyperams_options = parser.add_argument_group("HyperParameters")
    # note: make sure that the "dest" value is exactly the same as the 
    # variable name in "Hyperparameters" in order for over-riding to work correctly.
    hyperams_options.add_argument("-episodes", "--episodes", dest="num_episodes", type=int,
                                  help="Number of epochs to train")
    hyperams_options.add_argument('-async', '--asynchronous', dest='asynchronous',
                                  action='store_true', help="")

    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument('--log', dest="log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                               default="DEBUG", help="Logging level")
    args = parser.parse_args()
    return args


def setup_logger(args, logger):
    """ configures the global logger """
    if not hasattr(args, "log_level"):
        raise ValueError(f"parsed argumens expected")
    log_level = getattr(logging, args.log_level.upper())
    log_formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')

    # For the console
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    logger.setLevel(log_level)


if __name__ == "__main__":
    main()
