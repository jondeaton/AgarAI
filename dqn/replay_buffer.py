"""
File: replay_buffer
Date: 5/6/19 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import numpy as np
import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):
    """ Replay Memory for Deep Q Learning
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()
        self.index = 0

    def push(self, state, action, next_state, reward):
        """ Saves a transition. """
        transition = Transition(state, action, next_state, reward)
        if not self.full:
            self.memory.append(transition)
        else:
            self.memory[self.index] = transition
            self.index = (self.index + 1) % self.capacity

    def sample_pop(self, num_examples):
        indices = self._sample_indices(num_examples)
        examples = [self.memory[i] for i in indices]
        for i in reversed(sorted(indices)):
            del self.memory[i]
        return examples

    def sample(self, num_examples):
        indices = self._sample_indices(num_examples)
        return [self.memory[i] for i in indices]

    def _sample_indices(self, num_examples):
        return random.choices(list(range(len(self.memory))), k=num_examples)

    @property
    def full(self):
        return len(self) == self.capacity

    def __len__(self):
        return len(self.memory)