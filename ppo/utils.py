

import jax
from jax import numpy as np


@jax.jit
def take_along(x: np.ndarray, i: np.ndarray) -> np.ndarray:
    return x.take(i + np.arange(x.shape[0]) * x.shape[1])


@jax.jit
def one_hot(labels, num_classes, on_value=1.0, off_value=0.0):
  x = (labels[..., None] == np.arange(num_classes)[None])
  x = jax.lax.select(x, np.full(x.shape, on_value), np.full(x.shape, off_value))
  return x.astype(np.float32)


@jax.jit
def actor_loss(
        model, observations, actions, advantages, log_pa,
        clip_eps: float = 0.01):
    """PPO Actor Loss."""
    action_logits, values = model(observations)

    new_log_pas = jax.nn.log_softmax(action_logits) * one_hot(actions, action_logits.shape[1])
    new_log_pa = take_along(new_log_pas, actions)
    ratio = np.exp(new_log_pa - log_pa)
    loss = np.minimum(
        advantages * ratio,
        advantages * np.clip(ratio, 1 - clip_eps, 1 + clip_eps)).mean()
    return loss


@jax.jit
def critic_loss(model, observations, returns):
    _, values_pred = model(observations)
    return np.square(values_pred - returns).mean()


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


@jax.jit
def get_efficiency(rewards: np.ndarray,
                   episode_length: int,
                   num_pellets: int, arena_size: int):
    """Agario mass efficiency.

    Calculates the "agario mass efficiency", which is a quantity that i invented lol
    It is supposed to capture the rate at which mass was accumulated relative to
    the density of pellets in the arena. In this way, performance can be
    compared across episodes of different lengths, arenas of different sizes
    and with different numbers of pellets.
    """
    G = rewards.sum()
    pellet_density = num_pellets / pow(arena_size, 2)
    efficiency = G / (episode_length * pellet_density)
    return efficiency