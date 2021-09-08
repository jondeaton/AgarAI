

import jax
from jax import numpy as np
import numpy as onp

from typing import *


@jax.jit
def take_along(x: np.ndarray, i: np.ndarray) -> np.ndarray:
    return x.take(i + np.arange(x.shape[0]) * x.shape[1])


def ppo_actor_loss(
    action_logits, values, actions, advantages, log_pa,
        clip_eps: float = 0.01):
    """PPO Actor Loss."""
    one_hot_acitons = jax.nn.one_hot(actions, action_logits.shape[1])
    new_log_pas = jax.nn.log_softmax(action_logits) * one_hot_acitons
    new_log_pa = take_along(new_log_pas, actions)
    ratio = np.exp(new_log_pa - log_pa)
    loss = np.minimum(
        advantages * ratio,
        advantages * np.clip(ratio, 1 - clip_eps, 1 + clip_eps)).mean()
    return loss


@jax.jit
def critic_loss(values, returns):
    return np.square(values - returns).mean()


@jax.jit
def combine_dims(x):
    return x.reshape((-1, ) + x.shape[2:])


def loss(apply_fn,
         params, observations, actions, advantages, returns, log_pa,
         rng: Optional[jax.random.PRNGKey] = None):
    loss_actor, loss_critic = loss_components(
        apply_fn,
        params, observations, actions, advantages, returns, log_pa, rng=rng)
    return loss_actor + loss_critic


def loss_components(apply_fn,
         params, observations, actions, advantages, returns, log_pa,
         rng: Optional[jax.random.PRNGKey] = None):
    action_logits, values = apply_fn(params, observations, rng=rng)
    loss_actor = actor_loss(action_logits, values, actions, advantages, log_pa)
    loss_critic = critic_loss(values, returns)
    return loss_actor, loss_critic


def actor_loss(
        action_logits: np.ndarray,
        values: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        log_pa: np.ndarray) -> np.ndarray:
    """Standard Actor loss.

    Args:
        action_logits: (batch_size, num_actions)
        values: (batch_size, )
        actions: (batch_size, )
        advantages: (batch_size, )
        log_pa: (batch_size, num_actions)
    Returns:
        Scalar loss for the batch.
    """
    action_one_hot = jax.nn.one_hot(actions, action_logits.shape[1])
    a = jax.lax.stop_gradient(advantages)
    p = a[0] * action_one_hot
    l = xs(p, action_logits)
    return np.mean(l, axis=0)  # average over batch.


@jax.jit
def xs(p: np.ndarray, q_logits: np.ndarray) -> np.ndarray:
    log_q = jax.nn.log_softmax(q_logits, axis=1)
    return np.where(p == 0, 0, - p * log_q).sum(axis=1)


def fl(p, q_logits):
    q = jax.nn.softmax(q_logits, axis=1)
    mf = np.power(1 - q, 1.0)
    return xs(mf * p, q_logits)


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


def n_step_return(rewards: onp.ndarray,
                  value_estiamtes: onp.ndarray,
                  gamma: float,
                  n: int) -> onp.ndarray:
    returns = onp.zeros_like(rewards)
    for t in range(len(rewards)):

        Rt = value_estiamtes[t + n]
        for l in reversed(range(n - 1)):
            Rt = gamma * Rt + rewards[t + l]

        returns[t] = Rt

    return returns


def gae(returns: onp.ndarray,
        values: onp.ndarray,
        gamma: float, lam: float) -> onp.ndarray:
    """Generalized Advantage Estimation.

    Zero makes the advantage equal to Temporal Difference residuals
    (low variance)

        r[t] + gamma * V[t] - V[t - 1])

    and lam: 1.0 makes it equivalent to Monte-Carlo advantage estimation
    for the full rollout

    Args:
        returns: Shape (T, )
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
    delta = returns + gamma * values[1:] - values[:-1]  # TD residual.
    adv = onp.zeros_like(returns)

    adv[-1] = delta[-1]
    for t in reversed(range(len(returns) - 1)):
        adv[t] = delta[t] + (gamma * lam) * adv[t + 1]
    return adv


def make_returns(rewards: onp.ndarray, gamma: float, end_value: float = 0.0) -> onp.ndarray:
    """ Calculates the discounted future returns for a single rollout
    :param rewards: numpy array containing rewards
    :param gamma: discount factor 0 < gamma < 1
    :return: numpy array containing discounted future returns
    """
    returns = onp.zeros_like(rewards)
    ret = end_value
    for i in reversed(range(len(rewards))):
        ret = rewards[i] + gamma * ret
        returns[i] = ret

    return returns


def make_returns_batch(reward_batch: List[onp.ndarray], gamma: float) -> List[onp.ndarray]:
    """ Calculates discounted episodes returns
    :param reward_batch: list of numpy arrays. Each numpy array is
    the episode rewards for a single episode in the batch
    :param gamma: discount factor 0 < gamma < 1
    :return: list of numpy arrays representing the returns
    """
    return [make_returns(rewards, gamma) for rewards in reward_batch]

