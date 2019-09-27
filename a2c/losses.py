"""
File: losses
Date: 9/26/19 
Author: Jon Deaton (jonpauldeaton@gmail.com)
"""

import numpy as np
import tensorflow as tf
from typing import List


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
    :param returns: tensor of discounted returns
    :param value: model-estimated future returns
    :return: tensor equal to MSE of
    """
    returns = tf.reshape(returns, [-1])
    values_pred = tf.reshape(values_pred, [-1])
    return tf.keras.losses.mean_squared_error(returns, values_pred)


def actor_loss(actions, advantages, action_logits):
    """ Standard advantage Policy gradient loss for Actor """

    # actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
    actions = tf.cast(tf.squeeze(actions), tf.int32)

    weighted_sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    policy_loss = weighted_sparse_ce(actions, action_logits, sample_weight=advantages)

    # entropy loss may be calculated via CE over itself
    entropy_loss = tf.keras.losses.categorical_crossentropy(action_logits, action_logits, from_logits=True)
    return tf.reduce_mean(policy_loss - 1e-4 * entropy_loss)


def a2c_loss(model, observations, mask, actions, advantages, returns):
    """ computes the A2C loss and gradients of the given model w.r.t. that loss """
    with tf.GradientTape() as tape:
        action_logits, values_pred = model(observations, mask=mask, training=True)

        actor_loss_value = actor_loss(actions, advantages, action_logits)
        critic_loss_value = critic_loss(values_pred, returns)

        loss_value = [actor_loss_value, critic_loss_value]

    return loss_value, tape.gradient(loss_value, model.trainable_variables)
