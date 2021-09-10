
import os
import functools
import argparse, logging
from typing import *

import gym, gym_agario
import configuration
from configuration import config_pb2, environment_pb2

from utils import get_training_dir

import rollout
from ppo import utils
from ppo import action_conversion


import jax
from jax import numpy as np
import numpy as onp

from jax.experimental import stax
from jax.experimental import optimizers

from tqdm import tqdm


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # If its not an error then I dont care.
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")  # no GPU for U.


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
            stax.Conv(16, filter_shape=(5, 5)), SiLu,
            stax.Conv(16, filter_shape=(5, 5)), SiLu,
            stax.Conv(5, filter_shape=(3, 3)), SiLu, stax.Dropout(0.95, mode=mode),
            stax.Conv(5, filter_shape=(3, 3)), SiLu, stax.Dropout(0.95, mode=mode),
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
        critic_output = critic_fn(critic_params, features, rng=rng)[:, 0]
        return actor_output, critic_output

    return init_fn, apply_fn


def get_rollouts(config, env, model, key) -> rollout.MultiRollout:
    apply_fn, params = model
    hps = config.hyperparameters
    action_converter = action_conversion.make_action_converter(config.environment.action)

    dones = [False] * config.environment.num_agents
    roll = rollout.MultiRollout()

    observations = env.reset()
    for i in tqdm(range(hps.episode_length)):
        if all(dones): break

        _, key = jax.random.split(key)
        action_logits, values_estimate = apply_fn(params, np.array(observations), rng=key)
        action_indices = jax.random.categorical(key, action_logits)  # Choose actions.
        values = values_estimate
        actions = list(zip(*jax.vmap(action_converter)(action_indices)))
        next_obs, rewards, next_dones, _ = env.step(actions)

        roll.record({
            "observations": observations,
            "actions": action_indices,
            "rewards": rewards,
            "values": values,
            "action_logits": action_logits,
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
        roll.record({
            "values": value_estimates
        })

    return roll


def train(env: gym.Env, config: configuration.Config,
          test_env: Optional[gym.Env] = None,
          training_dir: Optional[str] = None):
    """Trains agent."""
    hps = config.hyperparameters

    input_shape = (1,) + env.observation_space.shape
    num_actions = action_conversion.action_size(config.environment.action)
    action_converter = action_conversion.make_action_converter(config.environment.action)

    init_fn, apply_fn = make_actor_critic(config.model, num_actions, 'train')

    key = jax.random.PRNGKey(hps.seed)
    _, params = init_fn(key, input_shape)

    learning_rate = lambda step: 0.001 * np.exp(- step / 500) + 0.00001

    adam_init, adam_update, get_params = optimizers.adam(learning_rate)
    adam_state = adam_init(params)

    step = 0
    for episode in range(hps.num_episodes):
        logging.info(f"Episode {episode}")

        _, key = jax.random.split(key)

        params = get_params(adam_state)
        model = (apply_fn, params)
        roll = get_rollouts(config, env, model, key=key)

        # Now these are arrays shaped (num_agents, steps, ...)
        observations = np.array(roll.batched('observations'))
        actions = np.array(roll.batched('actions'))
        action_logits = np.array(roll.batched('action_logits'))
        dones = np.array(roll.batched('dones'))

        # Make the returns / advantage estimate.
        rewards = roll.batched('rewards')
        values = roll.batched('values')
        advantages = []
        returns = []
        for r, v, a in zip(rewards, values, actions):
            g = utils.make_returns(r, hps.gamma, end_value=v[-1])
            returns.append(g)

            adv = utils.gae(r, v, gamma=hps.gamma, lam=hps.gae.lam)
            advantages.append(adv)

        rewards = np.array(rewards)
        values = np.array(values)[:, :-1]  # clip the last value estimate.
        returns = np.array(returns)
        advantages = np.array(advantages)

        dones = utils.combine_dims(dones)
        observations = utils.combine_dims(observations)[~dones]
        actions = utils.combine_dims(actions)[~dones]
        values = utils.combine_dims(values)[~dones]
        returns = utils.combine_dims(returns)[~dones]
        action_logits = utils.combine_dims(action_logits)[~dones]
        advantages = utils.combine_dims(advantages)[~dones]

        tf.summary.histogram('value_estimates', values, step=step)

        g = list(rewards.sum(axis=1))
        print(f'total rewards: {g}, mean: {onp.mean(g)}')
        max_sizes = np.cumsum(rewards, axis=1).max(axis=1)
        print(f'max sizes: {max_sizes}')

        tf.summary.histogram('returns',  g, step=step)
        tf.summary.histogram('maxsize', max_sizes, step=step)
        tf.summary.scalar('returns/avg_max_size', max_sizes.mean(), step=step)
        tf.summary.scalar('returns/median_max_size', np.median(max_sizes), step=step)
        tf.summary.scalar('returns/min_max_size', np.min(max_sizes), step=step)
        tf.summary.scalar('returns/average', onp.mean(g), step=step)
        tf.summary.scalar('returns/max', max(g), step=step)
        tf.summary.scalar('learning_rate', learning_rate(step), step=step)

        get_loss = jax.jit(functools.partial(utils.loss, apply_fn))
        get_loss_components = jax.jit(functools.partial(utils.loss_components, apply_fn))

        dparams_fn = jax.jit(jax.grad(get_loss))

        num_samples = observations.shape[0]
        for _ in tqdm(range(hps.num_sgd_steps)):
            _, key = jax.random.split(key)
            params = get_params(adam_state)

            al, cl, entropy = get_loss_components(params, observations, actions, advantages, returns, action_logits, rng=key)
            tf.summary.scalar('loss/actor', al, step=step)
            tf.summary.scalar('loss/critic', cl, step=step)
            tf.summary.scalar('loss/entropy', np.mean(entropy), step=step)
            tf.summary.histogram('entropy', entropy, step=step)

            batch_indices = jax.random.choice(
                key, np.arange(num_samples), shape=(hps.batch_size, ), replace=False)
            dparams = dparams_fn(params,
                                 observations[batch_indices], actions[batch_indices], advantages[batch_indices],
                                 returns[batch_indices], action_logits[batch_indices],
                                 rng=key)

            tf.summary.scalar('gradnorm', optimizers.l2_norm(dparams), step=step)
            dparams = optimizers.clip_grads(dparams, 1)

            adam_state = adam_update(step, dparams, adam_state)
            step += 1


def main():
    args = parse_args()

    if args.debug:
        logging.debug('Debug mode.')

    if args.debug or args.output is None:
        logging.warning("Model will not be saved.")
        training_dir = None
        tb_writer = tf.summary.create_noop_writer()
    else:
        output_dir = args.output
        os.makedirs(args.output, exist_ok=True)

        training_dir = get_training_dir(output_dir, args.name)
        os.makedirs(training_dir, exist_ok=True)
        tb_writer = tf.summary.create_file_writer(training_dir)
        logging.info(f"Training directory: {training_dir}")

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

    with tb_writer.as_default():
        train(train_env, config, test_env=test_env, training_dir=training_dir)
    logging.debug("Exiting.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an agent.")

    parser.add_argument('--config', default=None, required=True, help="Configuration name.")

    output_options = parser.add_argument_group("Output")
    output_options.add_argument("--output", default="model_outputs", help="Output directory")
    output_options.add_argument("--name", default="ppo", help="Experiment or run name")
    output_options.add_argument("--debug", action="store_true", help="Debug mode.")

    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument('--log', dest="log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                               default="INFO", help="Logging level")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
