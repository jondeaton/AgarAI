"""
File: rollout
Date: 9/18/19 
Author: Jon Deaton (jonpauldeaton@gmail.com)
"""

import numpy as np
from utils import none_filter
from typing import List

from collections import defaultdict


def transpose_batch(collection: List[np.ndarray]) -> List[np.ndarray]:
    """ Performs a transpose of a batch of roll-outs.
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
        self._records = defaultdict(list)

        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self._batch = None

    def __getitem__(self, item):
        return self._records[item]

    def record(self, item_dict):
        for item, value in item_dict.items():
            self[item].append(value)

    def batched(self, item):
        return transpose_batch(self[item])

    # The old API is below
    def as_batch(self, cache=True):
        """ Returns the roll-out as a batch of experiences (optionally cache) """
        if self._batch is not None:
            return self._batch
        if cache:
            self._batch = self._to_batch()
            return self._batch
        else:
            return self._to_batch()

    def record_step(self, observations=None, actions=None, rewards=None, values=None, dones=None):
        """ Records a single step forwards for each agent in in the batch """
        self.observations.append(observations)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.values.append(values)
        self.dones.append(dones)

    def _to_batch(self):
        """ Converts the recorded roll-out to a batch of experiences """
        obs_batch    = transpose_batch(self.observations)
        action_batch = transpose_batch(self.actions)
        reward_batch = transpose_batch(self.rewards)
        value_batch  = transpose_batch(self.values)
        dones        = transpose_batch(self.dones)
        return obs_batch, action_batch, reward_batch, value_batch, dones
