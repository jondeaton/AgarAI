
import os
from typing import *
from google.protobuf import text_format

from configuration import config_pb2


Config = config_pb2.Config
Agario = config_pb2.Config.Agario
Environment = config_pb2.Config.Environment
Observation = config_pb2.Config.Environment.Observation
Action = config_pb2.Config.Environment.Action
HyperParameters = config_pb2.Config.HyperParameters

def _configs_dir() -> str:
    """Gives directory containing configurations."""
    return os.path.join(os.path.dirname(__file__), "configs")

def load(name: str) -> Config:
    """Gets a configuration by name."""
    config_path = os.path.join(_configs_dir(), f'{name}.textproto')
    with open(config_path, 'r') as f:
        return text_format.Parse(f, Config())

_gym_names = {
        Observation.Type.GRID: "agario-grid-v0",
        Observation.Type.RAM: "agario-ram-v0",
        Observation.Type.SCREEN: "agario-screen-v0",
}
def gym_env_name(observation_type: Observation.Type) -> str:
    return _gym_names[observation_type]


# env_name = configutation.gym_env_name(config.environment.observation.type)
# env_config = configuration.gym_env_config(config)
# env = gym.name(env_name, **env_config)


def gym_env_config(config: Config) -> Dict[str, Any]:

    env_config = {
            'num_agents':      config.environment.num_agents,
            'difficulty':      config.environment.agario.difficulty,
            'ticks_per_step':  config.environment.agario.ticks_per_step,
            'arena_size':      config.environment.agario.arena_size,
            'num_pellets':     config.environment.agario.num_pellets,
            'num_viruses':     config.environment.agario.num_viruses,
            'num_bots':        config.environment.agario.num_bots,
            'pellet_regen':    config.environment.agario.pellet_regen,
    }

    if config.environment.observation.type == Observation.Type.GRID:
        env_config["grid_size"] = config.environment.observation.grid_size
        env_config["num_frames"] = config.environment.observation.num_frames
        env_config["observe_cells"] = config.environment.observation.cells
        env_config["observe_others"] = config.environment.observation.others
        env_config["observe_pellets"] = config.environment.observation.pellets
        env_config["observe_viruses"] = config.environment.observation.viruses

    return env_config
