"""
File: hyperparameters
Date: 2019-07-25 
Author: Jon Deaton (jonpauldeaton@gmail.com)
"""

import json
import gym

def make_test_env(env_name, hyperams):
    """ creates an environment for testing """
    return gym.make(env_name, **{
        'num_agents': 8,
        'difficulty': 'normal',
        'ticks_per_step': hyperams.ticks_per_step,
        'arena_size': 500,
        'num_pellets': 1000,
        'num_viruses': 25,
        'num_bots': 25,
        'pellet_regen': True,

        "grid_size": hyperams.grid_size,
        "observe_cells": hyperams.observe_cells,
        "observe_others": hyperams.observe_others,
        "observe_viruses": hyperams.observe_viruses,
        "observe_pellets": hyperams.observe_pellets
    })


def make_environment(env_name, hyperams):
    """ makes and configures the specified OpenAI gym environment """

    env_config = dict()

    if env_name == "agario-grid-v0":
        env_config = {
                'num_agents':      hyperams.agents_per_env,
                'difficulty':      hyperams.difficulty,
                'ticks_per_step':  hyperams.ticks_per_step,
                'arena_size':      hyperams.arena_size,
                'num_pellets':     hyperams.num_pellets,
                'num_viruses':     hyperams.num_viruses,
                'num_bots':        hyperams.num_bots,
                'pellet_regen':    hyperams.pellet_regen,
            }

        # observation parameters
        env_config.update({
            "grid_size":       hyperams.grid_size,
            "num_frames":      1,
            "observe_cells":   hyperams.observe_cells,
            "observe_others":  hyperams.observe_others,
            "observe_viruses": hyperams.observe_viruses,
            "observe_pellets": hyperams.observe_pellets
        })

    env = gym.make(env_name, **env_config)
    return env


class HyperParameters:
    def __init__(self):
        self.seed = 0

        # to fill in by sub_classes
        self.env_name = None
        self.EncoderClass = None
        self.action_shape = None

        self.num_envs = None

        # optimizer
        self.learning_rate = None

        self.gamma = None
        self.entropy_weight = None
        self.action_shape = None
        self.batch_size = None

        self.agents_per_env = None
        self.episode_length = None
        self.num_episodes = None

        self.save_frequency = 8

    def override(self, params):
        """Overrides attributes of this object with those of "params".

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
        """Save the hyper-parameters to file in JSON format."""
        with open(file, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @staticmethod
    def restore(file):
        """Restore hyper-parameters from JSON file."""
        with open(file, 'r') as f:
            data = json.load(f)
        hp = HyperParameters()
        hp.__dict__.update(data)
        return hp


class GridEnvHyperparameters(HyperParameters):
    def __init__(self):
        super(GridEnvHyperparameters, self).__init__()

        self.env_name = 'agario-grid-v0'

        self.architecture = 'Basic'
        self.encoder_class = 'CNN'

        self.learning_rate = 0.01
        self.num_episodes = 4096
        self.gamma = 0.95

        self.batch = False
        self.batch_size = 32

        self.num_envs = 4

        self.entropy_weight = 1e-4

        self.action_shape = (8, 1, 1)
        self.episode_length = 11

        self.num_sgd_steps = 64

        # Agario Game parameters
        self.difficulty = "normal"
        self.agents_per_env = 32
        self.ticks_per_step = 4  # set equal to 1 => bug
        self.arena_size = 500
        self.num_pellets = 1000
        self.num_viruses = 25
        self.num_bots = 0
        self.pellet_regen = True

        # observation parameters
        self.num_frames = 1
        self.grid_size = 32
        self.observe_pellets = True
        self.observe_viruses = True
        self.observe_cells   = True
        self.observe_others  = True
