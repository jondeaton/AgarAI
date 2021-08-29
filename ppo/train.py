import functools
import os, sys
import argparse, logging
from typing import *

import gym, gym_agario
import configuration
from configuration import config_pb2, environment_pb2

import rollout
from ppo import utils

import jax
from jax import numpy as np
import numpy as onp

from jax.experimental import stax
from jax.experimental import optimizers

from tqdm import tqdm


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # If its not an error then fuck off.
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")  # no GPU for U, bitch.

# Limit operations to single thread when on CPU.
# todo: Do I need this???
# os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


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


def action_size(action_config: environment_pb2.Action) -> int:
    # todo: fix this
    num_acts = int(action_config.allow_splitting) + int(action_config.allow_feeing)
    move_size = action_config.num_directions * action_config.num_magnitudes
    return 1 + move_size + action_config.num_directions * num_acts


def get_action_converter(action_config: environment_pb2.Action) -> Callable:
    """Creates function to turn what comes out of the model into an
    action that can go into the environment.
    """
    # todo: fix this
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


def make_feature_extractor(output_size: int, mode: str) -> Tuple[Callable, Callable]:
    SiLu = stax.elementwise(lambda x: x * jax.nn.sigmoid(x))
    return stax.serial(
        stax.Conv(32, filter_shape=(5, 5), padding='SAME'), stax.BatchNorm(), SiLu,
        stax.Conv(32, filter_shape=(5, 5), padding='SAME'), stax.BatchNorm(), SiLu,
        stax.Conv(10, filter_shape=(3, 3), padding='SAME'), stax.BatchNorm(), SiLu, stax.Dropout(0.3, mode=mode),
        stax.Conv(10, filter_shape=(3, 3), padding='SAME'), SiLu, stax.Dropout(0.3, mode=mode),
        stax.Flatten,
        stax.Dense(output_size))


def make_actor_critic(model_config, num_actions, mode: str):
    num_features = model_config.feature_extractor.num_features

    fe_init, fe_fn = make_feature_extractor(num_features, mode='train')
    actor_init, actor_fn = stax.Dense(num_actions)
    critic_init, critic_fn = stax.Dense(1)

    def init_fn(key, input_shape):
        return (
            fe_init(key, input_shape),
            actor_init(key, (num_features, )),
            critic_init(key, (num_features, ))
        )

    def apply_fn(params, observations):
        fe_params, actor_params, critic_params = params
        features = fe_fn(fe_params, observations)
        return actor_fn(features), critic_fn(features)

    return init_fn, apply_fn


def train(env: gym.Env, config: configuration.Config,
          test_env: Optional[gym.Env] = None,
          training_dir: Optional[str] = None):
    """Trains agent."""
    hps = config.hyperparameters

    input_shape = (1,) + env.observation_space.shape
    num_actions = action_size(config.environment.action)
    to_action = get_action_converter(config.environment.action)

    init_fn, apply_fn = make_actor_critic(config.model, num_actions, 'train')

    key = jax.random.PRNGKey(hps.seed)
    _, params = init_fn(key, input_shape)

    adam_init, adam_update, get_params = optimizers.adam(hps.lr)
    adam_state = adam_init(params)

    step = 0
    for ep in range(hps.num_episodes):
        logging.info(f"Episode {ep}")

        # Perform a rollout.
        roll = rollout.Rollout()
        observations = env.reset()
        dones = [False] * hps.agents_per_env
        for _ in tqdm(range(hps.episode_length)):
            if all(dones): break
            action_logits, values_estimate = apply_fn(observations)
            action_indices = jax.random.categorical(key, action_logits)  # Choose actions.
            log_pas = utils.take_along(jax.nn.log_softmax(action_logits), action_indices)
            values = list(np.squeeze(values_estimate))
            actions = [to_action(i) for i in action_indices]
            next_obs, rewards, next_dones, _ = env.step(actions)

            roll.record({
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
        observations = np.array(roll.batched('observations'))
        actions = np.array(roll.batched('actions'))
        log_pas = np.array(roll.batched('log_pas'))
        dones = np.array(roll.batched('dones'))
        values = np.array(roll.batched('values'))

        rewards = roll.batched('rewards')
        returns = np.array(utils.make_returns_batch(rewards, hps.gamma))
        advantages = returns - values

        for _ in range(hps.num_sgd_steps):
            dones = utils.combine_dims(dones)
            observations = utils.combine_dims(observations)[np.logical_not(dones)]
            actions = utils.combine_dims(actions)[np.logical_not(dones)]
            advantages = utils.combine_dims(advantages)[np.logical_not(dones)]
            returns = utils.combine_dims(returns)[np.logical_not(dones)]
            log_pas = utils.combine_dims(log_pas)[np.logical_not(dones)]

            params = get_params()
            dparams = jax.grad(
                functools.partial(utils.loss, apply_fn)
            )(params, observations, actions, advantages, returns, log_pas)

            adam_state = adam_update(step, dparams, adam_state)
            step += 1

def main():
    args = parse_args()

    if args.debug:
        logging.warning(f"Debug mode on. Model will not be saved")
        training_dir = None
    else:
        output_dir = args.output
        os.makedirs(args.output, exist_ok=True)

        training_dir = get_training_dir(output_dir, args.name)
        os.makedirs(training_dir, exist_ok=True)
        logging.info(f"Model directory: {training_dir}")

    config = configuration.load(args.config)
    print(f'Training config: {config}')

    # Train Environment.
    train_env = gym.make(
        configuration.gym_env_name(config.environment),
        **configuration.gym_env_config(config.environment))

    # Test Environment
    test_environment_config = (
      config.test_environment if config.HasField('test_environment')
      else config.environment
    )
    test_env = gym.make(
        configuration.gym_env_name(test_environment_config),
        **configuration.gym_env_config(test_environment_config))

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

    output_options = parser.add_argument_group("Output")
    output_options.add_argument("--output", default="model_outputs", help="Output directory")
    output_options.add_argument("--name", default="ppo", help="Experiment or run name")
    output_options.add_argument("--debug", action="store_true", help="Debug mode.")

    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument('--log', dest="log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                               default="DEBUG", help="Logging level")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
