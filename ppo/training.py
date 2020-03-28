"""
File: training
Date: 2/1/20 
Author: Jon Deaton (jonpauldeaton@gmail.com)
"""

from ppo.hyperparameters import HyperParameters
from a2c.rollout import Rollout, transpose_batch

import os
from tqdm import tqdm
import numpy as np
import logging
logger = logging.getLogger("root")

from flax import nn
from flax import optim

import jax
import jax.numpy as jnp
from jax import lax

from ppo.models import CNN
import tensorflow as tf

from a2c.losses import make_returns_batch

# Limit operations to single thread when on CPU.
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


CLIP_EPSILON = 0.01

dtype=jnp.float32


@jax.jit
def take_along(x: jnp.ndarray, i: jnp.ndarray) -> jnp.ndarray:
    return x.take(i + np.arange(x.shape[0]) * x.shape[1])


@jax.jit
def one_hot(labels, num_classes, on_value=1.0, off_value=0.0):
  x = (labels[..., None] == jnp.arange(num_classes)[None])
  x = lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
  return x.astype(jnp.float32)


@jax.jit
def actor_loss(model, observations, actions, advantages, log_pa):
    action_logits, values = model(observations)

    new_log_pas = nn.log_softmax(action_logits) * one_hot(actions, action_logits.shape[1])
    new_log_pa = take_along(new_log_pas, actions)
    ratio = jnp.exp(new_log_pa - log_pa)
    loss = jnp.minimum(
        advantages * ratio,
        advantages * jnp.clip(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)).mean()
    return loss


@jax.jit
def critic_loss(model, observations, returns):
    _, values_pred = model(observations)
    return jnp.square(values_pred - returns).mean()


@jax.jit
def combine_dims(x):
    return x.reshape((-1, ) + x.shape[2:])


@jax.jit
def train_step(optimizer, observations, actions, advantages, returns, log_pa):
    """Performs a single Proximal Policy Optimization training step."""

    def ppo_loss(model):
        loss_actor = actor_loss(model, observations, actions, advantages, log_pa)
        loss_critic = critic_loss(model, observations, returns)
        return loss_actor + loss_critic

    optimizer, _ = optimizer.optimize(ppo_loss)
    return optimizer


class Trainer:
    def __init__(self, get_env, hyperams: HyperParameters, to_action, test_env=None, training_dir=None):
        self.get_env = get_env
        self.hp = hyperams
        self.to_action = to_action
        self.test_env = test_env
        self.num_envs = hyperams.num_envs
        self.training_dir = training_dir

        self.key = jax.random.PRNGKey(0)

    def train(self):
        env = self.get_env()

        input_shape = (1, ) + env.observation_space.shape

        model_def = CNN.partial(action_shape=np.prod(self.hp.action_shape))
        _, model = model_def.create_by_shape(self.key, [(input_shape, dtype)])
        optimizer = optim.Adam(learning_rate=self.hp.learning_rate).create(model)

        summary_writer = None
        if self.training_dir is not None:
            model_directory = os.path.join(self.training_dir, "model")
            summary_writer = tf.summary.create_file_writer(self.training_dir)

        for ep in range(self.hp.num_episodes):
            logger.info(f"Episode {ep}")

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

            # self._log_rollout(summary_writer, ep, rewards)

    def _log_rollout(self, summary_writer, episode, rollout_batch, losses=None):
        """ logs the performance of the roll-out """
        episode_length = len(rollout_batch[2][0])
        logger.info(f"Episode {episode}, length: {episode_length}")
        if losses is not None:
            logger.info(f"Actor loss: {losses[0]:.3f}, Critic loss: {losses[1]:.3f}")

        returns = []
        max_masses = []
        average_masses = []
        efficiencies = []

        for rewards in rollout_batch[2]:
            episode_return = rewards.sum()
            mass = 10 + rewards.cumsum()
            eff = get_efficiency(rewards, episode_length, self.hp)

            returns.append(episode_return)
            max_masses.append(mass.max())
            average_masses.append(mass.mean())
            efficiencies.append(eff)

        print(f"Average Ep Return:\t{np.mean(returns):.2f}")
        print(f"Average Max mass:\t{np.mean(max_masses):.2f}")
        print(f"Average Avg mass:\t{np.mean(average_masses):.2f}")
        print(f"Average efficiency:\t{np.mean(efficiencies):.2f}")

        if summary_writer is not None:  # in debug mode theres no directory to write to
            import tensorflow as tf
            with summary_writer.as_default():
                tf.summary.scalar('train/efficiency', np.mean(efficiencies), step=episode)
                if losses is not None:
                    tf.summary.scalar('loss/actor', losses[0], step=episode)
                    tf.summary.scalar('loss/critic', losses[1], step=episode)

    def _test(self, model, summary_writer=None, episode_length=None):
        logger.info(f"Testing performance...")
        episode_length = episode_length or self.hp.episode_length

        rollout = Rollout()
        observations = self.test_env.reset()
        dones = [False] * self.hp.agents_per_env
        for _ in tqdm(range(self.hp.episode_length)):
            if all(dones): break
            action_logits, values_estimate = model(observations)
            action_indices = jax.random.categorical(self.key, action_logits)  # Choose actions.
            log_pas = take_along(nn.log_softmax(action_logits), action_indices)
            values = list(values_estimate)
            actions = [self.to_action(i) for i in action_indices]
            next_obs, rewards, next_dones, _ = self.test_env.step(actions)

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

        # todo: pass the real summary writer to log results to TensorBoard.
        self._log_rollout(summary_writer, "test", rollout.as_batch(), episode_length)


def get_efficiency(rewards, episode_length, hyperams):
    """ calculates the "agario mass efficiency", which is a quantity that i invented lol
    It is supposed to capture the rate at which mass was accumulated relative to
    the density of pellets in the arena. In this way, performance can be
    compared across episodes of different lengths, arenas of different sizes
    and with different numbers of pellets.
    """
    G = rewards.sum()
    pellet_density = hyperams.num_pellets / pow(hyperams.arena_size, 2)
    efficiency = G / (episode_length * pellet_density)
    return efficiency