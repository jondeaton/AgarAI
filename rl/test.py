import unittest
from parameterized import parameterized

import rl

import jax
from jax import numpy as np

import numpy as onp


def normal_gae(rewards: onp.ndarray,
               values: onp.ndarray,
               gamma: float, lam: float) -> onp.ndarray:
    delta = rewards + gamma * values[1:] - values[:-1]  # TD residual.
    adv = onp.zeros_like(rewards)

    adv[-1] = delta[-1]
    for t in reversed(range(len(rewards) - 1)):
        adv[t] = delta[t] + (gamma * lam) * adv[t + 1]
    return adv


def make_returns(rewards: onp.ndarray, gamma: float, end_value: float = 0.0) -> onp.ndarray:
    returns = onp.zeros_like(rewards)
    ret = end_value
    for i in reversed(range(len(rewards))):
        ret = rewards[i] + gamma * ret
        returns[i] = ret
    return returns


class TestRL(unittest.TestCase):

    def test_is_same(self):
        T = 100
        for i in range(100):
            key = jax.random.PRNGKey(i)
            r = jax.random.normal(key, shape=(T,))

            _, key = jax.random.split(key)
            v = jax.random.normal(key, shape=(T + 1,))

            _, key = jax.random.split(key)
            gamma = jax.random.uniform(key)

            _, key = jax.random.split(key)
            lam = jax.random.uniform(key)

            onp.testing.assert_allclose(
                rl.gae(r, v, gamma, lam),
                normal_gae(r, v, gamma, lam),
                rtol=1e-4)
            onp.testing.assert_allclose(
                rl.make_returns(r, gamma, end_value=2.0),
                make_returns(r, gamma, end_value=2.0),
                rtol=1e-2)

    @parameterized.expand([
        [1.0], [0.5], [0.0],
    ])
    def test_gae_lam0(self, gamma: float):
        r = np.array([0.0, 3.0, 0.0])
        v = np.array([0.0, 0.0, 0.0, 0.0])

        onp.testing.assert_allclose(
            rl.gae(r, v, gamma, 0.0),
            [0.0, 3.0, 0.0],
        )

    @parameterized.expand([
        [1.0], [0.5], [0.0],
    ])
    def test_gae_values(self, gamma):
        r = np.array([0.0, 3.0, 0.0])
        v = np.array([0.0, 0.0, 0.0, 7.0])

        onp.testing.assert_allclose(
            rl.gae(r, v, gamma=gamma, lam=0.0),
            [0.0, 3.0, 7 * gamma],
        )


if __name__ == '__main__':
    unittest.main()
