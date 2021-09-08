"""Configuration utils.

Example usage:

    import configuration

    env_name = configuration.gym_env_name(config.environment)
    env_config = configuration.gym_env_config(config.environment)
    env = gym.make(env_name, **env_config)
"""
import os
from typing import *

from google.protobuf import text_format
from configuration import config_pb2, environment_pb2


Config = config_pb2.Config
Agario = environment_pb2.Agario
Environment = environment_pb2.Environment
Observation = environment_pb2.Observation
Action = environment_pb2.Action
HyperParameters = config_pb2.HyperParameters


def _configs_dir() -> str:
    """Gives directory containing configurations."""
    return os.path.join(os.path.dirname(__file__), "configs")


def load(name: str) -> Config:
    """Gets a configuration by name."""
    config_path = os.path.join(_configs_dir(), f'{name}.textproto')
    with open(config_path, 'r') as f:
        return text_format.MergeLines(f, Config())


_gym_names = {
    Observation.Type.GRID: "agario-grid-v0",
    Observation.Type.RAM: "agario-ram-v0",
    Observation.Type.SCREEN: "agario-screen-v0",
}


def gym_env_name(environment: Environment) -> str:
    return _gym_names[environment.observation.type]


def gym_env_config(environment: Environment) -> Dict[str, Any]:
    """Makes the Gym environment configuration dict from a Config."""
    difficulty = environment_pb2.Agario.Difficulty.DESCRIPTOR.values_by_number[environment.agario.difficulty].name

    env_config = {
            'num_agents':      environment.num_agents,
            'multi_agent':     True, # todo: put this into the proto
            'difficulty':      difficulty.lower(),
            'ticks_per_step':  environment.observation.ticks_per_step,
            'arena_size':      environment.agario.arena_size,
            'num_pellets':     environment.agario.num_pellets,
            'num_viruses':     environment.agario.num_viruses,
            'num_bots':        environment.agario.num_bots,
            'pellet_regen':    environment.agario.pellet_regen,
    }

    if environment.observation.type == Observation.Type.GRID:
        env_config["grid_size"] = environment.observation.grid_size
        env_config["num_frames"] = environment.observation.num_frames
        env_config["observe_cells"] = environment.observation.cells
        env_config["observe_others"] = environment.observation.others
        env_config["observe_pellets"] = environment.observation.pellets
        env_config["observe_viruses"] = environment.observation.viruses

    return env_config
