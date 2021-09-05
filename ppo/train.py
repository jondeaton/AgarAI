import functools
import os, sys
import argparse, logging
from typing import *

import gym, gym_agario
import configuration
from configuration import config_pb2, environment_pb2

import rollout
from ppo import utils
from ppo import action_conversion

import jax
from jax import numpy as np
import numpy as onp

from jax.experimental import stax
from jax.experimental import optimizers

from tqdm import tqdm


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # If its not an error then fuck off.
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")  # no GPU for U, bitch.


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
            stax.Conv(8, filter_shape=(5, 5), padding='SAME'), stax.BatchNorm(), SiLu,
            stax.Conv(8, filter_shape=(5, 5), padding='SAME'), stax.BatchNorm(), SiLu,
            stax.Conv(4, filter_shape=(3, 3), padding='SAME'), stax.BatchNorm(), SiLu, stax.Dropout(0.3, mode=mode),
            stax.Conv(4, filter_shape=(3, 3), padding='SAME'), SiLu, stax.Dropout(0.3, mode=mode),
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


def get_rollouts(config, env, model, key) -> rollout.MultiRollout:
    apply_fn, params = model
    hps = config.hyperparameters
    action_converter = action_conversion.make_action_converter(config.environments[0].action)

    roll = rollout.MultiRollout()

    observations = env.reset()

    num_agents = len(observations)
    dones = [False] * num_agents

    for i in tqdm(range(hps.episode_length)):
        if all(dones): break

        _, key = jax.random.split(key)
        action_logits, values_estimate = apply_fn(params, np.array(observations), rng=key)
        action_indices = jax.random.categorical(key, action_logits)  # Choose actions.
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

    # Need to record the final value estimate for all the agents that didn't finish.
    # that didn't finish. This means that agents which didn't finish will have one additional
    # observation
    if not all(dones):
        _, value_estimates = apply_fn(params, np.array(observations), rng=key)
        value_estimates = onp.array(value_estimates)
        value_estimates[dones] = 0.0
        for agent in range(num_agents):
            roll.record({
                "values": value_estimates
            })

    return roll




def train(envs: List[gym.Env], config: configuration.Config,
          test_env: Optional[gym.Env] = None,
          training_dir: Optional[str] = None):
    """Trains agent."""
    hps = config.hyperparameters

    input_shape = (1,) + envs[0].observation_space.shape
    num_actions = action_conversion.action_size(config.environments[0].action)
    action_converter = action_conversion.make_action_converter(config.environments[0].action)

    init_fn, apply_fn = make_actor_critic(config.model, num_actions, 'train')

    key = jax.random.PRNGKey(hps.seed)
    _, params = init_fn(key, input_shape)

    adam_init, adam_update, get_params = optimizers.adam(hps.learning_rate)
    adam_state = adam_init(params)

    step = 0
    for episode in range(hps.num_episodes):
        logging.info(f"Episode {episode}")

        _, key = jax.random.split(key)

        params = get_params(adam_state)
        model = (apply_fn, params)

        multi_obs = []
        multi_actions = []
        multi_values = []
        multi_returns = []
        multi_log_pas = []
        multi_advantages = []
        for env in envs:
            roll = get_rollouts(config, env, model, key=key)

            # Now these are arrays shaped (num_agents, steps, ...)
            observations = np.array(roll.batched('observations'))
            actions = np.array(roll.batched('actions'))
            log_pas = np.array(roll.batched('log_pas'))
            dones = np.array(roll.batched('dones'))

            # Make the returns estimate.
            # todo: using Monte-Carlo here. Use n-step return or GAE if this doesn't work.
            rewards = roll.batched('rewards')
            values = roll.batched('values')
            returns = []
            for r, v in zip(rewards, values):
                g = utils.make_returns(r, hps.gamma, end_value=v[-1])
                returns.append(g)

            rewards = np.array(rewards)
            values = np.array(values)[:, :-1]  # clip the last value estimate.
            returns = np.array(returns)

            dones = utils.combine_dims(dones)
            observations = utils.combine_dims(observations)[~dones]
            actions = utils.combine_dims(actions)[~dones]
            values = utils.combine_dims(values)[~dones]
            returns = utils.combine_dims(returns)[~dones]
            log_pas = utils.combine_dims(log_pas)[~dones]
            advantages = returns - values


            multi_obs.append(observations)
            multi_actions.append(actions)
            multi_values.append(values)
            multi_returns.append(returns)
            multi_log_pas.append(log_pas)
            multi_advantages.append(advantages)


            g = list(rewards.sum(axis=1))
            print(f'total rewards: {g}, mean: {onp.mean(g)}')

            tf.summary.histogram(f'returns',  g, step=step)
            tf.summary.scalar(f'returns/average', onp.mean(g), step=step)
            tf.summary.scalar(f'returns/max', max(g), step=step)

        # combine from multiple environments.
        observations = np.vstack(multi_obs)
        actions = np.vstack(multi_actions)
        returns = np.vstack(multi_returns)
        log_pas = np.vstack(multi_log_pas)
        advantages = np.vstack(multi_advantages)

        get_loss = jax.jit(functools.partial(utils.loss, apply_fn))
        get_loss_components = jax.jit(functools.partial(utils.loss_components, apply_fn))

        @jax.jit
        def dparams_fn(*args, **kwargs):
            dparams = jax.grad(get_loss)(*args, **kwargs)
            return jax.tree_map(functools.partial(np.clip, a_min=-0.1, a_max=0.1), dparams)

        num_samples = observations.shape[0]
        batch_size = 256
        for _ in tqdm(range(hps.num_sgd_steps)):
            _, key = jax.random.split(key)
            params = get_params(adam_state)

            al, cl = get_loss_components(params, observations, actions, advantages, returns, log_pas, rng=key)
            tf.summary.scalar(f'loss/actor', al, step=step)
            tf.summary.scalar(f'loss/critic', cl, step=step)

            i = jax.random.choice(key, np.arange(num_samples), shape=(batch_size, ), replace=False)
            dparams = dparams_fn(params, observations[i], actions[i], advantages[i],
                                 returns[i], log_pas[i], rng=key)
            adam_state = adam_update(step, dparams, adam_state)
            step += 1


def main():
    args = parse_args()

    if args.debug:
        logging.warning(f"Debug mode on. Model will not be saved")
        training_dir = None
        tb_writer = tf.summary.create_noop_writer()
    else:
        output_dir = args.output
        os.makedirs(args.output, exist_ok=True)

        training_dir = get_training_dir(output_dir, args.name)
        os.makedirs(training_dir, exist_ok=True)
        tb_writer = tf.summary.create_file_writer(training_dir)
        logging.info(f"Model directory: {training_dir}")

    config = configuration.load(args.config)

    # Train Environment.
    train_envs = [
        gym.make(
            configuration.gym_env_name(environment),
            **configuration.gym_env_config(environment))
        for environment in config.environments
    ]

    # Test Environment
    if config.HasField('test_environment'):
        test_env = gym.make(
            configuration.gym_env_name(config.test_environment),
            **configuration.gym_env_config(config.test_environment))
    else:
        test_env = None

    with tb_writer.as_default():
        train(train_envs, config, test_env=test_env, training_dir=training_dir)

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
