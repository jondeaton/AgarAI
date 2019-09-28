"""
File: rollout
Date: 9/18/19 
Author: Jon Deaton (jonpauldeaton@gmail.com)
"""

import numpy as np
from utils import none_filter
from typing import List


def transpose_batch(collection: List[np.ndarray]) -> List[np.ndarray]:
    """ performs a transpose of a batch of roll-out
    :param collection: List[np.ndarray]
    :return: List[np.ndarray]
    """
    transposed = zip(*collection)
    return [np.array(list(none_filter(rollout))) for rollout in transposed]


class Rollout:
    """ This class represents a batch of "roll-outs". Each roll-out is
    a complete history of an agent's interactions with it's environment.
    """
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self._batch = None

    def record(self, observations, actions, rewards, values, dones):
        """ records a single step forwards for each agent in in the batch
        """
        self.observations.append(observations)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.values.append(values)
        self.dones.append(dones)

    def as_batch(self):
        """ returns the roll-out as a batch of experiences (will cache) """
        if self._batch is not None:
            return self._batch
        self._batch = self._to_batch()
        return self.as_batch()

    def _to_batch(self):
        """ converts the recorded roll-out to a batch of experiences """
        obs_batch    = transpose_batch(self.observations)
        action_batch = transpose_batch(self.actions)
        reward_batch = transpose_batch(self.rewards)
        value_batch  = transpose_batch(self.values)
        dones        = transpose_batch(self.dones)
        return obs_batch, action_batch, reward_batch, value_batch, dones
