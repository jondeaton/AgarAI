
import os, sys
import argparse, logging
from typing import *

import gym, gym_agario
import configuration

import jax
from jax import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # If its not an error then fuck off.
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")  # no GPU for U, bitch.

# Limit operations to single thread when on CPU.
# todo: Do I need this???
# os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


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
            "num_frames":      1,
            "observe_cells":   hyperams.observe_cells,
            "observe_others":  hyperams.observe_others,
            "observe_viruses": hyperams.observe_viruses,
            "observe_pellets": hyperams.observe_pellets
        })

    env = gym.make(env_name, **env_config)
    return env


def agario_to_action(index, action_shape):
    """ Converts a raw action index into an Agario action """
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


def action_size(action_config: configuration.Action) -> int:
    num_acts = int(action_config.allow_splitting) + int(action_config.allow_feeing)
    move_size = action_config.num_directions * action_config.num_magnitudes
    return 1 + move_size + action_config.num_directions * num_acts


def get_action_converter(action_config: configuration.Action) -> Callable:

    def index_to_action(indexg: int):
        return np.array([0, 0]), 0
        # if index == 0:
        #     return np.array([0, 0]), 0
        #
        #  move_size = action_config.num_directions * action_config.num_magnitudes
        #  if index + 1 < move_size:
        #      ...  # ugh
        #
        # indices = np.unravel_index(index, action_config.num_directions)
        # theta = (2 * np.pi * indices[0]) / action_shape[0]
        # mag = 1 - indices[1] / action_shape[1]
        # act = int(indices[2])
        # x = np.cos(theta) * mag
        # y = np.sin(theta) * mag
        # return np.array([x, y]), act

    return index_to_action




def train(env: gym.Env, config: configuration.Config,
          test_env: Optional[gym.Env] = None,
          training_dir: Optional[str] = None):
    """Trains agent."""

    input_shape = (1,) + env.observation_space.shape

    model_def = CNN.partial(action_shape=np.prod(self.hp.action_shape))
    _, model = model_def.create_by_shape(self.key, [(input_shape, dtype)])
    optimizer = optim.Adam(learning_rate=self.hp.learning_rate).create(model)

    if training_dir is not None:
        model_directory = os.path.join(self.training_dir, "model")
        summary_writer = tf.summary.create_file_writer(self.training_dir)
    else:
        summary_writer = None
    
    for ep in range(self.hp.num_episodes):
        logging.info(f"Episode {ep}")

        # Perform a rollout.
        rollout = Rollout()
        observations = env.reset()
        dones = [False] * self.hp.agents_per_env
        for _ in tqdm(range(self.hp.episode_length)):
            if all(dones): break
            action_logits, values_estimate = model(observations)
            action_indices = jax.random.categorical(self.key, action_logits)  # Choose actions.
            log_pas = take_along(nn.log_softmax(action_logits), action_indices)
            values = list(jnp.squeeze(values_estimate))
            actions = [self.to_action(i) for i in action_indices]
            next_obs, rewards, next_dones, _ = env.step(actions)

            rollout.record({
                "observations": observations,
                "actions": action_indices,
                "rewards": rewards,
                "values": values,
                "log_pas": log_pas,
                "dones": dones
            })
            dones = next_dones
            observations = next_obs

        # Optimize the proximal policy.
        observations = jnp.array(rollout.batched('observations'))
        actions = jnp.array(rollout.batched('actions'))
        log_pas = jnp.array(rollout.batched('log_pas'))
        dones = jnp.array(rollout.batched('dones'))
        values = jnp.array(rollout.batched('values'))

        rewards = rollout.batched('rewards')
        returns = jnp.array(make_returns_batch(rewards, self.hp.gamma))
        advantages = returns - values

        for step in range(self.hp.num_sgd_steps):
            dones = combine_dims(dones)

            observations = combine_dims(observations)[jnp.logical_not(dones)]
            actions = combine_dims(actions)[jnp.logical_not(dones)]
            advantages = combine_dims(advantages)[jnp.logical_not(dones)]
            returns = combine_dims(returns)[jnp.logical_not(dones)]
            log_pas = combine_dims(log_pas)[jnp.logical_not(dones)]

            optimizer = train_step(optimizer, observations, actions, advantages, returns, log_pas)

def main():
    args = parse_args()

    if args.env == "agario-grid-v0":
        from ppo.hyperparameters import GridEnvHyperparameters
        hyperams = GridEnvHyperparameters()
        to_action = lambda i: agario_to_action(i, hyperams.action_shape)
    else:
        raise ValueError(args.env)

    hyperams.override(args)
    logging.debug(f"Environment: {args.env}")

    if args.debug:
        logging.warning(f"Debug mode on. Model will not be saved")
        training_dir = None
    else:
        output_dir = args.output
        os.makedirs(args.output, exist_ok=True)

        training_dir = get_training_dir(output_dir, args.name)
        os.makedirs(training_dir, exist_ok=True)
        logging.info(f"Model directory: {training_dir}")


    config = configuration.load_config(args.config)

    # Train Environment.
    train_env = gym.make(
        configuration.gym_env_name(config.environment),
        **configuration.gym_env_config(config.environment))

    # Test Environment
    test_env = gym.make(
        configuration.gym_env_name(config.test_environment),
        **configuration.gym_env_config(config.test_environment))

    train(train_env, config, test_env=test_env, training_dir=training_dir)
    logging.debug("Exiting.")


def get_training_dir(output_dir: Optional[str], name: str) -> Optional[str]:
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
    while os.path.exists(f"{base}-{i:03d}"):
        i += 1
    return f"{base}-{i:03d}"


def parse_args():
    parser = argparse.ArgumentParser(description="Train an agent.")

    parser.add_argument('--config', default=None, required=True, help="Configuration name.")

    env_options = parser.add_argument_group("Environment")
    env_options.add_argument("--env", default="agario-grid-v0", choices=["agario-grid-v0"])

    output_options = parser.add_argument_group("Output")
    output_options.add_argument("--output", default="model_outputs", help="Output directory")
    output_options.add_argument("--name", default="ppo-debug", help="Experiment or run name")
    output_options.add_argument("--debug", action="store_true", help="Debug mode")

    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument('--log', dest="log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                               default="DEBUG", help="Logging level")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
