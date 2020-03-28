"""
File: losses
Date: 9/26/19 
Author: Jon Deaton (jonpauldeaton@gmail.com)
"""

import numpy as np
import tensorflow as tf
from typing import List


def get_loss_variables(rollout_batch, gamma, recurrent):
    """ converts a batched roll-out of experience into a
    set of variables suitable for using for a loss funciton
    :param rollout_batch: batched roll-out returned from rollout.as_batch()
    :param gamma: discount factor
    :param recurrent: boolean, whether the model to be trained is recurrent or not
    :return: set of variables to pass to a2c_loss
    """
    observations, actions, rewards, values, dones = rollout_batch

    returns = make_returns_batch(rewards, gamma)

    # convert lists of numpy arrays into full numpy arrays
    obs_batch = np.array(observations)
    act_batch = np.array(actions)
    ret_batch = np.array(returns)
    val_batch = np.array(values)
    adv_batch = ret_batch - val_batch

    not_done_mask = np.logical_not(np.array(dones))
    if recurrent:  # for recurrent models, we pass the
        mask = tf.convert_to_tensor(not_done_mask, dtype=tf.bool)
    else:  # for non-recurrent models, we explicitly select examples with mask
        obs_batch = obs_batch[not_done_mask]
        act_batch = act_batch[not_done_mask]
        adv_batch = adv_batch[not_done_mask]
        ret_batch = ret_batch[not_done_mask]
        mask = None

    loss_vars = obs_batch, mask, act_batch, adv_batch, ret_batch
    return loss_vars


def div_round_up(n, d):
    """ ceil(n / d) """
    return int((n + d - 1) / d)


def batch_loss_variables(loss_vars, batch_size):
    """ splits up los variables into batches """
    obs, mask, act, adv, ret = loss_vars

    num_examples = obs.shape[0]
    num_batches = div_round_up(num_examples, batch_size)

    for b in range(num_batches):
        i = b * batch_size
        yield obs[i:i + batch_size], \
              mask[i:i + batch_size] if mask is not None else mask, \
              act[i:i+batch_size], \
              adv[i:i + batch_size], \
              ret[i:i + batch_size]


def make_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """ Calculates the discounted future returns for a single rollout
    :param rewards: numpy array containing rewards
    :param gamma: discount factor 0 < gamma < 1
    :return: numpy array containing discounted future returns
    """
    returns = np.zeros_like(rewards)

    ret = 0.0
    for i in reversed(range(len(rewards))):
        returns[i] = ret = rewards[i] + gamma * ret

    return returns


def make_returns_batch(reward_batch: List[np.ndarray], gamma: float) -> List[np.ndarray]:
    """ Calculates discounted episodes returns
    :param reward_batch: list of numpy arrays. Each numpy array is
    the episode rewards for a single episode in the batch
    :param gamma: discount factor 0 < gamma < 1
    :return: list of numpy arrays representing the returns
    """
    return [make_returns(rewards, gamma) for rewards in reward_batch]


def critic_loss(values_pred, returns):
    """ Critic loss defined as MSE for value estimation
    :param values_pred: model-estimated future returns
    :param returns: tensor of discounted returns
    :return: tensor equal to MSE of
    """
    returns = tf.reshape(returns, [-1])
    values_pred = tf.reshape(values_pred, [-1])
    return tf.keras.losses.mean_squared_error(returns, values_pred)


def actor_loss(actions, advantages, action_logits, entropy_weight):
    """ Standard advantage Policy gradient loss for Actor """
    actions = tf.cast(tf.squeeze(actions), tf.int32)

    weighted_sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    policy_loss = weighted_sparse_ce(actions, action_logits, sample_weight=advantages)

    # entropy loss may be calculated via CE over itself
    entropy_loss = tf.keras.losses.categorical_crossentropy(action_logits, action_logits, from_logits=True)
    return tf.reduce_mean(policy_loss - entropy_weight * entropy_loss)


def a2c_loss(model, entropy_weight, observations, mask, actions, advantages, returns):
    """ computes the A2C loss and gradients of the given model w.r.t. that loss """
    with tf.GradientTape() as tape:
        action_logits, values_pred = model(observations, mask=mask, training=True)

        actor_loss_value = actor_loss(actions, advantages, action_logits, entropy_weight)
        critic_loss_value = critic_loss(values_pred, returns)

        loss_value = [actor_loss_value, critic_loss_value]

    return loss_value, tape.gradient(loss_value, model.trainable_variables)
