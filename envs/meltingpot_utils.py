from typing import Tuple

import tree
import dm_env
import dmlab2d
import numpy as np

from gym import spaces
from ray.rllib.agents import trainer
from ml_collections import config_dict
from ray.rllib.env import multi_agent_env
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID



PLAYER_STR_FORMAT = 'player_{index}'


def _timestep_to_observations(timestep: dm_env.TimeStep):
    gym_observations = {}
    for index, observation in enumerate(timestep.observation):
        gym_observations[PLAYER_STR_FORMAT.format(index=index)] = {
            key: value for key, value in observation.items() if 'WORLD' not in key
        }
    return gym_observations


def _timestep_to_global_state(timestep: dm_env.TimeStep):
    gym_global_state = {}
    for index, observation in enumerate(timestep.observation):
        gym_global_state = {
            key: value for key, value in observation.items() if 'WORLD' in key
        }
        break  # global information is the same for all players
    return gym_global_state


def _remove_world_observations_from_space(
    observation: spaces.Dict) -> spaces.Dict:
    return spaces.Dict(
        {key: observation[key] for key in observation if 'WORLD' not in key})


def _get_world_observations_from_space(
    observation: spaces.Dict) -> spaces.Dict:
    return spaces.Dict(
        {key: observation[key] for key in observation if 'WORLD' in key})


def _spec_to_space(spec: tree.Structure[dm_env.specs.Array]) -> spaces.Space:
    """Converts a dm_env nested structure of specs to a Gym Space.

    BoundedArray is converted to Box Gym spaces. DiscreteArray is converted to
    Discrete Gym spaces. Using Tuple and Dict spaces recursively as needed.

    Args:
    spec: The nested structure of specs

    Returns:
    The Gym space corresponding to the given spec.
    """
    if isinstance(spec, dm_env.specs.DiscreteArray):
        return spaces.Discrete(spec.num_values)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        return spaces.Box(spec.minimum, spec.maximum, spec.shape, spec.dtype)
    elif isinstance(spec, dm_env.specs.Array):
        if np.issubdtype(spec.dtype, np.floating):
            return spaces.Box(-np.inf, np.inf, spec.shape, spec.dtype)
        elif np.issubdtype(spec.dtype, np.integer):
            info = np.iinfo(spec.dtype)
            return spaces.Box(info.min, info.max, spec.shape, spec.dtype)
        else:
            raise NotImplementedError(f'Unsupported dtype {spec.dtype}')
    elif isinstance(spec, (list, tuple)):
        return spaces.Tuple([_spec_to_space(s) for s in spec])
    elif isinstance(spec, dict):
        return spaces.Dict({key: _spec_to_space(s) for key, s in spec.items()})
    else:
        raise ValueError('Unexpected spec: {}'.format(spec))
