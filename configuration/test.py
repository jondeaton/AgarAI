import unittest

import configuration
from configuration import config_pb2, environment_pb2


class MyTestCase(unittest.TestCase):

    def test_load(self):
        config = configuration.load('test_config')
        self.assertEqual(
            config.environment.agario.difficulty,
            environment_pb2.Agario.Difficulty.EASY)

    def test_make_env_config(self):
        config = configuration.load('test_config')
        env_config = configuration.gym_env_config(config.environment)
        self.assertNotEqual(env_config, {})



if __name__ == '__main__':
    unittest.main()
