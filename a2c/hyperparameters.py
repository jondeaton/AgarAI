"""
File: hyperparameters
Date: 2019-07-25 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import json
import math


class HyperParameters(object):
    """ Simple class for storing model hyper-parameters """

    def __init__(self):
        self.seed = 42

        # to fill in by sub_classes
        self.env_name = None
        self.encoder_type = None
        self.extractor_type = None

        # optimizer
        self.learning_rate = 0.0007
        self.max_gradient_norm = 0.5

        # loss
        self.params_value = 1.0
        self.entropy_weight = 1e-4

        self.num_envs = 16

        # Agario Game parameters
        self.ticks_per_step = 4  # set equal to 1 => bug
        self.arena_size = 16
        self.num_pellets = 1
        self.num_viruses = 0
        self.num_bots = 0
        self.pellet_regen = True

        self.action_shape = (16, 2, 1)

        self.episode_length = 2000
        self.num_episodes = 64
        self.gamma = 0.99

        self.batch_size = 256

    def override(self, params):
        """
        Overrides attributes of this object with those of "params".
        All attributes of "params" which are also attributes of this object will be set
        to the values found in "params". This is particularly useful for over-riding
        hyper-parameters from command-line arguments
        :param params: Object with attributes to override in this object
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


class FullEnvHyperparameters(HyperParameters):
    def __init__(self):
        super(FullEnvHyperparameters, self).__init__()
        self.env_name = "agario-full-v0"


class ScreenEnvHyperparameters(HyperParameters):
    def __init__(self):
        super(ScreenEnvHyperparameters, self).__init__()
        self.env_name = "agario-screen-v0"


class GridEnvHyperparameters(HyperParameters):
    def __init__(self):
        super(GridEnvHyperparameters, self).__init__()

        self.env_name = "agario-grid-v0"

        self.num_frames = 1
        self.grid_size = 5
        self.observe_pellets = True
        self.observe_viruses = False
        self.observe_cells = False
        self.observe_others = False

        self.feature_extractor = None
