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

def action_size(action_config: environment_pb2.Action) -> int:
    num_acts = int(action_config.allow_splitting) + int(action_config.allow_feeding)
    move_size = action_config.num_directions * action_config.num_magnitudes
    return 1 + move_size + action_config.num_directions * num_acts


AgarioAction = Tuple[np.ndarray, int]


def make_action_converter(action_config: environment_pb2.Action):
    """Creates function to turn what comes out of the model into an
    action that can go into the environment.

    Actions have the following form

    index 0: do nothing
    next (num_directions * num_magnitudes) indexes: just move
    next (num_directions * num_actions): act in specified direction
    """
    num_moves = action_config.num_directions * action_config.num_magnitudes

    @jax.jit
    def direction(magnitude: float, theta: float) -> np.ndarray:
        x = np.cos(theta) * magnitude
        y = np.sin(theta) * magnitude
        return np.array([x, y])

    @jax.jit
    def move(index: int) -> AgarioAction:
        theta_index = index % action_config.num_directions
        mag_index = index // action_config.num_directions

        theta = 2 * np.pi * theta_index / action_config.num_directions
        mag = 1 - mag_index / (1 + action_config.num_magnitudes)
        return direction(mag, theta), 0

    @jax.jit
    def act(index: int) -> AgarioAction:
        theta_index = index % action_config.num_directions
        theta = 2 * np.pi * theta_index / action_config.num_directions

        if action_config.allow_splitting and action_config.allow_feeding:
            act = 1 + index / action_config.num_directions
        elif action_config.allow_splitting:
            act = 1
        elif action_config.allow_feeding:
            act = 2
        else:
            act = 0

        return direction(1.0, theta), act

    @jax.jit
    def do_something(index: int) -> AgarioAction:
        return jax.lax.cond(
            index < num_moves,
            lambda i: move(i),
            lambda i: act(i - num_moves),
            operand=index
        )

    @jax.jit
    def action_converter(index: int) -> AgarioAction:
        do_nothing = np.array([0, 0], dtype=np.float32), 0
        return jax.lax.cond(
            index == 0,
            lambda _: do_nothing,
            lambda i: do_something(i - 1),
            operand=index)

    return action_converter


def make_actor_critic(model_config: config_pb2.Model,
                      num_actions: int, mode: str):
    """Makes Actor-Critic model.

    Args:
        model_config: Model configuration.
        num_actions: The number of actions.
        mode: 'train' or 'test'.
    Returns:
        Initialization and apply function.
    """
    num_features = model_config.feature_extractor.num_features

    def make_feature_extractor(output_size: int, mode: str) -> Tuple[Callable, Callable]:
        SiLu = stax.elementwise(lambda x: x * jax.nn.sigmoid(x))
        return stax.serial(
            stax.Conv(32, filter_shape=(5, 5), padding='SAME'), stax.BatchNorm(), SiLu,
            stax.Conv(32, filter_shape=(5, 5), padding='SAME'), stax.BatchNorm(), SiLu,
            stax.Conv(10, filter_shape=(3, 3), padding='SAME'), stax.BatchNorm(), SiLu, stax.Dropout(0.3, mode=mode),
            stax.Conv(10, filter_shape=(3, 3), padding='SAME'), SiLu, stax.Dropout(0.3, mode=mode),
            stax.Flatten,
            stax.Dense(output_size))

    fe_init, fe_fn = make_feature_extractor(num_features, mode='train')
    actor_init, actor_fn = stax.Dense(num_actions)
    critic_init, critic_fn = stax.Dense(1)

    def init_fn(key, input_shape):
        input_shape, fe_params = fe_init(key, input_shape)
        _, actor_params = actor_init(key, (num_features, ))
        _, critic_params = critic_init(key, (num_features, ))
        params = fe_params, actor_params, critic_params
        return input_shape, params

    @jax.jit
    def apply_fn(params, observations, rng: Optional[jax.random.PRNGKey] = None):
        observations = observations.astype(np.float32)
        fe_params, actor_params, critic_params = params
        features = fe_fn(fe_params, observations, rng=rng)
        actor_output = actor_fn(actor_params, features, rng=rng)
        critic_output = np.squeeze(critic_fn(critic_params, features, rng=rng))
        return actor_output, critic_output

    return init_fn, apply_fn


def train(env: gym.Env, config: configuration.Config,
          test_env: Optional[gym.Env] = None,
          training_dir: Optional[str] = None):
    """Trains agent."""
    hps = config.hyperparameters

    input_shape = (1,) + env.observation_space.shape
    num_actions = action_size(config.environment.action)
    action_converter = make_action_converter(config.environment.action)

    init_fn, apply_fn = make_actor_critic(config.model, num_actions, 'train')

    key = jax.random.PRNGKey(hps.seed)
    _, params = init_fn(key, input_shape)

    adam_init, adam_update, get_params = optimizers.adam(hps.learning_rate)
    adam_state = adam_init(params)

    step = 0
    for ep in range(hps.num_episodes):
        logging.info(f"Episode {ep}")

        params = get_params(adam_state)

        # Perform a rollout.
        roll = rollout.Rollout()
        observations = env.reset()
        dones = [False] * config.environment.num_agents
        for _ in tqdm(range(hps.episode_length)):
            _, key = jax.random.split(key)
            if all(dones): break
            action_logits, values_estimate = apply_fn(params, np.array(observations), rng=key)
            action_indices = jax.random.categorical(key, action_logits)  # Choose actions.
            # todo: take along isn't right here...
            log_pas = utils.take_along(jax.nn.log_softmax(action_logits), action_indices)
            values = values_estimate
            actions = list(zip(*jax.vmap(action_converter)(action_indices)))
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

        observations = np.array(roll.batched('observations'))
        actions = np.array(roll.batched('actions'))
        log_pas = np.array(roll.batched('log_pas'))
        dones = np.array(roll.batched('dones'))
        values = np.array(roll.batched('values'))

        rewards = roll.batched('rewards')
        g = np.array(rewards).sum(axis=1)
        print(f'total rewards: {g}')

        returns = np.array(utils.make_returns_batch(rewards, hps.gamma))
        advantages = returns - values

        dones = utils.combine_dims(dones)
        observations = utils.combine_dims(observations)[np.logical_not(dones)]
        actions = utils.combine_dims(actions)[np.logical_not(dones)]
        advantages = utils.combine_dims(advantages)[np.logical_not(dones)]
        returns = utils.combine_dims(returns)[np.logical_not(dones)]
        log_pas = utils.combine_dims(log_pas)[np.logical_not(dones)]

        dparams_fn = jax.jit(
            jax.grad(
                functools.partial(utils.loss, apply_fn)
            )
        )
        for _ in tqdm(range(hps.num_sgd_steps)):
            _, key = jax.random.split(key)
            params = get_params(adam_state)
            dparams = dparams_fn(params, observations, actions, advantages, returns, log_pas, rng=key)
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
