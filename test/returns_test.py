"""
File: returns_test
Date: 9/19/19 
Author: Jon Deaton (jonpauldeaton@gmail.com)
"""

import numpy as np
import unittest

from a2c.training import make_returns


class ReturnsTest(unittest.TestCase):
    """ tests the 'make_returns' function """
    def test_len(self):
        for length in (0, 1, 10):
            rewards = np.zeros(length)
            returns = make_returns(rewards, 1)
            self.assertEqual(len(rewards), len(returns))

    def test_zero_rewards(self):
        rewards = np.zeros(10)
        returns = make_returns(rewards, 1)

        self.assertEqual(returns.sum(), 0)

    def test_zero_discount(self):
        rewards = 4 + np.arange(10)
        returns = make_returns(rewards, 0)
        self.assertEqual(len(rewards), len(returns))

        for rew, ret in zip(rewards, returns):
            self.assertEqual(rew, ret)

    def test_returns_discounted(self):
        np.random.seed(10)
        rewards = np.random.randn(30)

        gamma = 0.75
        returns = make_returns(rewards, gamma)
        self.assertEqual(len(rewards), len(returns))

        ret = 0
        for i in reversed(range(len(rewards))):
            ret = rewards[i] + gamma * ret
            self.assertEqual(ret, returns[i])


if __name__ == "__main__":
    unittest.main()