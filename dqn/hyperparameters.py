"""
File: hyperparameters
Date: 5/6/19 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import json


class HyperParameters(object):
    """ Simple class for storing model hyper-parameters """

    def __init__(self):
        self.seed = 42

        self.gamma = 0.99

        self.num_episodes = 1000

        self.p_dropout = 0.05

        self.action_shape = (64, 64, 3)

        # DQN parameters
        self.batch_size = 16
        self.replay_buffer_capacity = 1000
        self.epsilon_base = 0.1
        self.epsilon_decay = 0.1
        self.target_update_freq = 50

        # Adam Optimization parameters
        self.lr = 0.1
        self.adam_betas = (0.9, 0.999)
        self.adam_eps = 1e-8
        self.grad_clip_norm = None

    def override(self, params):
        """
        Overrides attributes of this object with those of "params".
        All attributes of "params" which are also attributes of this object will be set
        to the values found in "params". This is particularly useful for over-riding
        hyperparamers from command-line arguments
        :param settings: Object with attributes to override in this object
        :return: None
        """
        for attr in vars(params):
            if hasattr(self, attr) and getattr(params, attr) is not None:
                value = getattr(params, attr)
                setattr(self, attr, value)

    def save(self, file):
        """ save the hyper-parameters to file in JSON format """
        with open(file, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @staticmethod
    def restore(file):
        """ restore hyper-parameters from JSON file """
        with open(file, 'r') as f:
            data = json.load(f)
        hp = HyperParameters()
        hp.__dict__.update(data)
        return hp