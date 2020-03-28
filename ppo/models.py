from flax import nn
from flax import optim

import ppo
from ppo import random

import ppo.numpy as jnp

import numpy as onp


class CNN(nn.Module):

  def __init__(self, action_shape):
    self._action_shape = action_shape

  def apply(self, x):
    x = nn.Conv(x, features=32, kernel_size=(3, 3))
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(x, features=64, kernel_size=(3, 3))
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(x, features=256)
    x = nn.relu(x)
    a = nn.Dense(x, features=self._action_shape)
    v = nn.Dense(x, features=1)
    return a, v
