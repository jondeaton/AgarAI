
import jax
from jax import numpy as np
from configuration import environment_pb2

from typing import *


def action_size(action_config: environment_pb2.Action) -> int:
    """The total number of actions specified in an action config."""
    num_acts = int(action_config.allow_splitting) + int(action_config.allow_feeding)
    move_size = action_config.num_directions * action_config.num_magnitudes
    return 1 + move_size + action_config.num_directions * num_acts


AgarioAction = Tuple[np.ndarray, int]


def make_action_converter(action_config: environment_pb2.Action):
    """Creates function to turn what comes out of the model into an
    action that can go into the environment.

    Actions have the following form

    index 0: do nothing
    next (num_directions * num_magnitudes) indexes: just move
    next (num_directions * num_actions): act in specified direction
    """
    num_moves = action_config.num_directions * action_config.num_magnitudes

    @jax.jit
    def direction(magnitude: float, theta: float) -> np.ndarray:
        x = np.cos(theta) * magnitude
        y = np.sin(theta) * magnitude
        return np.array([x, y])

    @jax.jit
    def move(index: int) -> AgarioAction:
        theta_index = index % action_config.num_directions
        mag_index = index // action_config.num_directions

        theta = 2 * np.pi * theta_index / action_config.num_directions
        mag = 1 - mag_index / (1 + action_config.num_magnitudes)
        return direction(mag, theta), 0

    @jax.jit
    def act(index: int) -> AgarioAction:
        theta_index = index % action_config.num_directions
        theta = 2 * np.pi * theta_index / action_config.num_directions

        if action_config.allow_splitting and action_config.allow_feeding:
            act = 1 + index / action_config.num_directions
        elif action_config.allow_splitting:
            act = 1
        elif action_config.allow_feeding:
            act = 2
        else:
            act = 0

        return direction(1.0, theta), act

    @jax.jit
    def do_something(index: int) -> AgarioAction:
        return jax.lax.cond(
            index < num_moves,
            lambda i: move(i),
            lambda i: act(i - num_moves),
            operand=index
        )

    @jax.jit
    def action_converter(index: int) -> AgarioAction:
        do_nothing = np.array([0, 0], dtype=np.float32), 0
        return jax.lax.cond(
            index == 0,
            lambda _: do_nothing,
            lambda i: do_something(i - 1),
            operand=index)

    return action_converter
