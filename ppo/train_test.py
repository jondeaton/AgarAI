import unittest

import gym, gym_agario
import configuration

import jax
from jax import numpy as np

import ppo.train


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.config = configuration.load_config('test_config')

        # Train Environment.
        self.train_env = gym.make(
            configuration.gym_env_name(self.config.environment),
            **configuration.gym_env_config(self.config.environment))

        # Test Environment
        self.test_env = gym.make(
            configuration.gym_env_name(self.config.test_environment),
            **configuration.gym_env_config(self.config.test_environment))

    def test_training(self):

        ppo.train.train(self.train_env, self.config, test_env=None)




if __name__ == '__main__':
    unittest.main()
