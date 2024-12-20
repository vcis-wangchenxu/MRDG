import os
import sys
import collections

import cv2
import dm_env
import numpy as np
import tensorflow as tf

from gym import spaces
import envs.meltingpot_utils as utils
from meltingpot.python import substrate, scenario

from functools import partial
from dm_env.specs import Array
from smac.env import MultiAgentEnv
from ml_collections import config_dict


class MeltingPotEnv(MultiAgentEnv):
    def __init__(self, args, key, seed, individual_rewards=False, gray_scale=False) -> None:
        self.args = args
        self.key = key
        self._seed = seed
        self._gray_scale = gray_scale
        self.individual_rewards = individual_rewards  # use individual rewards or global rewards to train the model
        self.create_env()
    
    def create_env(self):
        """Create substrate or evaluation scenario"""
        self._create_env()
    
    def _create_env(self):
        """ Create training environment """

        self.env_type = "train"
        self.env_config = substrate.get_config(self.key)
        self.env_config.unlock()
        self.env_config['env_seed'] = self._seed
        self.env_config.lock()
        
        self.global_observation_names = self.env_config.global_observation_names
        self.individual_observation_names = self.env_config.individual_observation_names
        
        self._env = substrate.build(config_dict.ConfigDict(self.env_config))
        self._num_players = len(self._env.observation_spec())
        self.n_agents = self._num_players
        self._ordered_agent_ids = [
            utils.PLAYER_STR_FORMAT.format(index=index)
            for index in range(self.n_agents)
        ]
        self._agent_ids = set(self._ordered_agent_ids)
        self.episode_limit = self.env_config.lab2d_settings["maxEpisodeLengthFrames"]
        self.observation_spec = self._process_obs_spec(self._env.observation_spec()[0])
    
    def _process_obs_spec(self, spec):
        for k in spec.keys():
            if self._gray_scale and 'RGB' in k : # H, W, C
                spec[k] = Array(shape=(*spec[k].shape[:-1], 1), dtype=spec[k].dtype, name=spec[k].name)
        return spec
    
    def step(self, actions):
        """ Returns reward, terminated, info """
        timestep = self._env.step(actions)
        done = {'__all__': True if timestep.last() else False}
        info = {
            'agents_reward': collections.OrderedDict({
                f'agent_{i}': float(r) for i, r in enumerate(timestep.reward)
            }),
            'individual_rewards': self.individual_rewards
        }

        reward = self.process_rewards(timestep.reward)
        if self.key.startswith('clean_up'):  # clean up uses the same rewards
            info['agents_reward'] = collections.OrderedDict({
                f'agent_{i}': float(reward) for i, _ in enumerate(timestep.reward)
            })
        
        self.observations = utils._timestep_to_observations(timestep)
        self._obs = self._cat_obs(self.observations)
        self.state = self.post_process_state(utils._timestep_to_global_state(timestep))
        return reward, done, info

    def post_process_state(self, state):
        for k, v in state.items():
            if self._gray_scale and 'RGB' in k:
                state[k] = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)[..., None]
        return state

    def process_rewards(self, rewards):
        # per-capita return, the mean indiviau rewards
        if self.env_type == 'train':
            # NOTE: explicitely create two branches for debugging purpose
            reward = sum(rewards) / self.n_agents
        elif self.env_type == 'evaluation':
            reward = sum(rewards) / self.n_agents
        else:
            raise ValueError(f"env_type: {self.env_type} not available")
        return reward

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.observations[utils.PLAYER_STR_FORMAT.format(index=agent_id)]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        obs_spec = self.single_player_observation_space()
        size = self._cal_spec_size(obs_spec)
        return size

    def get_state(self, concated_obs=True):
        return self.state

    def get_state_size(self):
        """ Returns the shape of the state"""
        state_spec = utils._get_world_observations_from_space(
            utils._spec_to_space(self._process_obs_spec(self._env.observation_spec()[0])))
        size = self._cal_spec_size(state_spec)
        return size

    def _cal_spec_size(self, state_spec):
        size = 0
        for k, spec in state_spec.spaces.items():
            shape = spec.shape
            if len(shape) == 0:
                size += 1
            else:
                size += np.prod(shape)
        return size

    def get_avail_actions(self):
        # TODO: check if melting pot supports this function
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        # TODO: check if melting pot supports this function
        """ Returns the available actions for agent_id """
        raise NotImplementedError

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self._env.action_spec()[0].num_values

    def reset(self):
        """ Returns initial observations and states"""
        timestep = self._env.reset()
        self.observations = utils._timestep_to_observations(timestep)
        self._obs = self._cat_obs(self.observations)
        self.state =self.post_process_state(utils._timestep_to_global_state(timestep))

    def _cat_obs(self, obs):
        """Concatenate observations for all agents"""
        res = {}
        for agent_id, obs_dict in obs.items():
            for k, v in obs_dict.items():
                if self._gray_scale and 'RGB' in k:
                    v = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)[..., None]
                if k not in res:
                    res[k] = [v]
                else:
                    res[k].append(v)
        for k, v in res.items():
            res[k] = np.stack(v)
        return res

    def get_dmlab2d_env(self):
        """Returns the underlying DM Lab2D environment."""
        return self._env

    def single_player_observation_space(self) -> spaces.Space:
        """The observation space for a single player in this environment."""
        return utils._remove_world_observations_from_space(
            utils._spec_to_space(self._process_obs_spec(self._env.observation_spec()[0])))

    def single_player_action_space(self):
        """The action space for a single player in this environment."""
        return utils._spec_to_space(self._env.action_spec()[0])

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        # TODO: use observation_spec to create the env_info and the scheme for the replay buffer
        env_info = {
            "state_shape": self.get_state_size(),  # WORLD.RGB size
            "obs_shape": self.get_obs_size(),  # total size excluding WORLD.RGB
            "n_actions": self.get_total_actions(),  # total number of actions
            "n_agents": self.n_agents,  # total number of agents
            "episode_limit": self.episode_limit,  # maximum number of steps per episode
            "observation_spec": self.observation_spec,  # observation spec
            "world_rgb_spec": utils._spec_to_space(self._process_obs_spec(self._env.observation_spec()[0]))['WORLD.RGB'],
            "obs_spec": self.single_player_observation_space(),
            "env_config": self.env_config,
        }
        return env_info


class MeltingPotEvalEnv(MeltingPotEnv):
    def __init__(self, args, key, seed, individual_rewards=False, gray_scale=False) -> None:
        super().__init__(args, key, seed, individual_rewards, gray_scale)
    
    def create_env(self):
        self._create_env()
    
    def _create_env(self):
        """ Create evaluation environment """
        # DO NOT USE GPU, use CPUs
        tf.config.set_visible_devices([], 'GPU')
        
        self.env_type = "evaluation"
        self.env_config = scenario.get_config(self.key)
        self.focal_mask = self.env_config.is_focal
        self.env_config.unlock()
        self.env_config['env_seed'] = self._seed
        self.env_config.lock()
        
        self.global_observation_names = self.env_config.substrate.global_observation_names
        self.individual_observation_names = self.env_config.substrate.individual_observation_names
        
        self._env = scenario.build(config_dict.ConfigDict(self.env_config))
        self._num_players = len(self._env.observation_spec())
        self.n_agents = self._num_players
        assert self.n_agents == sum(self.focal_mask)
        self._ordered_agent_ids = [
            utils.PLAYER_STR_FORMAT.format(index=index)
            for index in range(self.n_agents)
        ]
        self._agent_ids = set(self._ordered_agent_ids)
        self.episode_limit = self.env_config.substrate.lab2d_settings["maxEpisodeLengthFrames"]
        self.observation_spec = self._env.observation_spec()[0]    
