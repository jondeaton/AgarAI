"""Reinforcement Learning Utilities."""

import jax
from jax import numpy as np
import numpy as onp

from jax import ops

from typing import *


@jax.jit
def ppo_actor_loss(
        action_logits: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        old_action_logits: np.ndarray,
        clip_eps: float) -> float:
    """Proximal Policy Optimization Actor Loss.

    As per https://arxiv.org/abs/1707.06347.

    Args:
        action_logits: (batch_size, num_actions)
        actions: (batch_size, )
        advantages: (batch_size, )
        old_action_logits: (batch_size, num_actions)
    Returns:
        Scalar loss for the batch.
    """
    advantages = jax.lax.stop_gradient(advantages)

    new_log_pa = np.take_along_axis(
        jax.nn.log_softmax(action_logits, axis=1),
        actions[:, np.newaxis], axis=1)[:, 0]
    old_log_pa = np.take_along_axis(
        jax.nn.log_softmax(old_action_logits, axis=1),
        actions[:, np.newaxis], axis=1)[:, 0]

    ratio = np.exp(new_log_pa - old_log_pa)

    l = - np.minimum(
        advantages * ratio,
        advantages * np.clip(ratio, 1 - clip_eps, 1 + clip_eps))

    return l.mean(axis=0)


@jax.jit
def critic_loss(values, returns):
    return huber(values - returns).mean()


def huber(a, delta: float = 1.0):
    return np.where(
        np.abs(a) < delta, np.square(a) / 2, delta * (np.abs(a) - delta / 2)
    )

def actor_loss(
        action_logits: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray) -> np.ndarray:
    """Standard Actor loss.

    Args:
        action_logits: (batch_size, num_actions)
        actions: (batch_size, )
        advantages: (batch_size, )
    Returns:
        Scalar loss for the batch.
    """
    _, num_actions = action_logits.shape
    action_one_hot = jax.nn.one_hot(actions, num_actions)
    adv = jax.lax.stop_gradient(advantages)
    p = adv[:, np.newaxis] * action_one_hot
    l = fl(p, action_logits)
    return l.mean(axis=0)  # average over batch.


@jax.jit
def xs(p: np.ndarray, q_logits: np.ndarray) -> np.ndarray:
    """Cross Entropy."""
    log_q = jax.nn.log_softmax(q_logits, axis=1)
    return np.where(p == 0, 0, - p * log_q).sum(axis=1)


def fl(p, q_logits, gamma: float = 1.0):
    """Focal Loss."""
    q = jax.nn.softmax(q_logits, axis=1)
    mf = np.power(1 - q, gamma)
    return xs(mf * p, q_logits)


@jax.jit
def gae(rewards: np.ndarray,
         values: np.ndarray,
         gamma: float,
         lam: float) -> np.ndarray:
    """Generalized Advantage Estimation.

    Lambda zero makes the advantage equal to Temporal Difference residuals
    (low variance)

        r[t] + gamma * V[t] - V[t - 1]

    and lam: 1.0 makes it equivalent to Monte-Carlo advantage estimation
    for the full rollout.

    Args:
        rewards: Shape (T, )
        values: Shape (T + 1, ) state value estimates. This has
          shape one larger than returns so that the last step
          that is part of the estimate can have a next-step-value
          estimate. If the agent "finished" the episode then this
          final value should be zero, otherwise it should be the actual
          value-estimate for the final state reached which is technically
          not part of the episode.
        gamma: Future time-discount. Between (inclusive) 0 and 1.
        lam: Hyper-parameter between 0 and 1 inclusive  which scales the
          variance of the advantage estimation.
    Returns:
        Advantage estimates of the same shape as returns: (T, ).
    """
    T, = rewards.shape
    v_next = values[1:]
    v_current = values[:-1]
    delta = rewards + gamma * v_next - v_current

    # Convert to real time.
    t = lambda i: T - 1 - i

    return jax.lax.fori_loop(
        0, T,
        lambda i, adv: jax.ops.index_update(
            adv, t(i), delta[t(i)] + (gamma * lam) * adv[t(i) + 1]
        ),
        np.zeros(T))


@jax.jit
def make_returns(rewards: np.ndarray, gamma: float, end_value: float = 0.0) -> np.ndarray:
    """ Calculates the discounted future returns for a single rollout
    :param rewards: numpy array containing rewards
    :param gamma: discount factor 0 < gamma < 1
    :return: numpy array containing discounted future returns
    """
    T, = rewards.shape
    t = lambda i: T - 1 - i
    return jax.lax.fori_loop(
        0, T,
        lambda i, ret: jax.ops.index_update(
            ret, t(i),
            rewards[t(i)] + gamma * jax.lax.cond(
                i == 0,
                lambda _: end_value,
                lambda _: ret[t(i) + 1],
                None)
        ),
        np.zeros(T))


@jax.jit
def agario_efficiency(rewards: np.ndarray,
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
