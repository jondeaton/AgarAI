

import jax
from jax import numpy as np
import numpy as onp

from typing import *

import rl


@jax.jit
def combine_dims(x):
    return x.reshape((-1, ) + x.shape[2:])


def loss(apply_fn,
         params, observations, actions, advantages, returns, action_logits,
         rng: Optional[jax.random.PRNGKey] = None):
    loss_actor, loss_critic, entropy = loss_components(
        apply_fn,
        params, observations, actions, advantages, returns, action_logits, rng=rng)
    return loss_actor + loss_critic - (entropy.mean() / 100)


def loss_components(
        apply_fn,
        params, observations, actions, advantages, returns, old_action_logits,
        rng: Optional[jax.random.PRNGKey] = None):
    action_logits, values = apply_fn(params, observations, rng=rng)
    loss_actor = rl.ppo_actor_loss(
        action_logits, actions, advantages, old_action_logits, clip_eps = 0.2)
    loss_critic = rl.critic_loss(values, returns)

    entropy = rl.xs(jax.nn.softmax(action_logits, axis=1), action_logits)

    return loss_actor, loss_critic, entropy