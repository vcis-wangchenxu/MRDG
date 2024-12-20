import os
import sys
import copy
import time
import json
import shutil
import random
import logging
import traceback
import collections

from typing import List

import numpy as np
import torch as th
import torch.nn as nn


logger = logging.getLogger(__name__)


def slice_reward(reward, slice_range, scheme='old'):
    if scheme == 'old':
        return np.round(reward // slice_range, 4)
    elif scheme == 'new':
        if reward >= 0:
            offset = reward // slice_range
            offset += 0 if (reward % slice_range) == 0 else 1
            return np.round(offset * slice_range, 4)
        else:
            return np.round((reward // slice_range) * slice_range, 4)
    else:
        raise ValueError(f'rpm_reward_slice_scheme: {scheme} is not available!')


class RankedPolicyMemory:
    def __init__(self, args) -> None:
        logging.basicConfig(level=args.log_level)
        self.args = args
        self.sample_strategy = self.args.rpm_sample_strategy
        self._memory = dict()
        self.device = th.device(self.args.device)
        self._rpm_slice = args.rpm_reward_slice  # by default 1, for meltingpot 10
        self._save_count_interval = 0
        self._t_memory = dict()
        self._last_t = -10000
    
    def save_rpm(self, t_env):
        _bak = copy.deepcopy(self._memory)
        for k, v in _bak.items():
            _bak[k] = list(v)
        self._t_memory[t_env] = _bak
        with open(os.path.join(self.args.local_results_path, 'rpm_ts.json'), 'w') as f:
            json.dump(self._t_memory, f, indent=4)
        logging.info(f" \n -----> RPM saved (t={t_env}) in {os.path.join(self.args.local_results_path, 'rpm_ts.json')} \n")

    def update(self, t_env: int, checkpoint: dict):
        """Update the RPM memory

        Args:
            checkpoint (dict): the evaluated score and the corresponding model
        """
        reward = slice_reward(checkpoint['reward'],
                              self._rpm_slice,
                              scheme=self.args.rpm_reward_slice_scheme)
        reward = str(reward)
        state_dict = checkpoint['state_dict']

        if self.args.save_to_disk:
            token = t_env
            # save the path to the model
            save_path = os.path.join(self.args.local_results_path, "rpm_models", str(reward), str(t_env))
            os.makedirs(save_path, exist_ok=True)
            th.save(state_dict, "{}/agent.th".format(save_path))
        else:
            # save the model in the RAM, not in GPU
            token = collections.OrderedDict({
                param_name: weights.cpu().numpy() for param_name, weights in state_dict.items()
            })

        self._save(reward, token)
        # self._clean_disk(reward)

    def get_memory(self):
        return self._memory

    def sync_memory(self, memory):
        """Used in parallel setting"""
        self._memory = memory
        return True

    def _save(self, reward, token):
        if reward not in self._memory:
            self._memory[reward] = collections.deque([token], maxlen=100)
        else:
            self._memory[reward].append(token)
    
    def _clean_disk(self, reward):
        # clean the disk
        self._save_count_interval += 1
        if self.args.save_to_disk and self._save_count_interval == 20:
            try:
                st = time.time()
                deleted_files = []
                # get the base path of the reward
                base_path = os.path.join(self.args.local_results_path, "rpm_models", str(reward))
                files = os.listdir(base_path)
                logging.info(f"{self.__class__.__name__}: cleaning: {base_path}, slots in the base_path: {files}"
                             f"checkpoint sets: {set(self._memory[reward])}")
                for f in files:  # f is str type while items in self._memory[reward] are int type
                    # f not in the memory and deque is not full
                    if int(f) not in set(self._memory[reward]) and \
                        len(self._memory[reward]) == self._memory[reward].maxlen:
                        path = os.path.join(base_path, str(f))  # path is a directory
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                            deleted_files.append(path)
                self._save_count_interval = 0
                logging.info(f"{self.__class__.__name__} clean disk, "
                             f"deleted directories: {deleted_files}, time costs: {(time.time()-st):.2f}")
            except Exception as e:
                logging.error(f"Error in {self.__class__.__name__}: {e}")
                traceback.print_exc()
    
    def sample(self, n_agent: int, t: int = -1) -> collections.OrderedDict:
        """Sample the models from the memory
        """
        assert self.memory_key_size > 0, "The memory is empty"

        if t - self._last_t >= 30000:
            self._last_t = t
            self.save_rpm(t)

        if self.args.rpm_sample_strategy == 'uniform':
            keys = list(self._memory.keys())
            if self.args.rpm_top_n:
                _keys = {i: float(v) for i, v in enumerate(keys)}
                _indices = sorted(_keys.items(), key=lambda v:v[1])[-int(n_agent):]
                keys = [keys[i] for i, v in _indices]
            logging.info(f'RPM sampled keys (t={t}): {keys}')
            rewards = np.random.choice(keys, n_agent, replace=True)
            agents_token = [
                (r, np.random.choice(self._memory[r], 1)[0]) for r in rewards
            ]  # it can be token of the saved dict path or the save dict (numpy array) saved in RAM
            return agents_token
        elif self.args.rpm_sample_strategy == 'self_play':
            keys = list(self._memory.keys())
            logging.info(f'RPM sampled keys (t={t}): {keys}')
            reversed_memory = [(r, model) for r, model_lists in self._memory.items() for model in model_lists]
            sorted_indices = sorted(reversed_memory, key=lambda v:v[1])
            if random.random() >= 0.3:  # sample the newest one
                agents_token = []
                for i in range(n_agent):
                    if (i+1) <= len(sorted_indices):
                        agents_token.append(sorted_indices[-(i+1)])
                    else:
                        agents_token.append(sorted_indices[0])
            else:  # sample the old one
                if len(sorted_indices) > 1:
                    sorted_indices = sorted_indices[:-1]
                agents_token = random.choices(sorted_indices, k=n_agent)
            return agents_token
        elif self.args.rpm_sample_strategy == 'ablation_random':
            reversed_memory = [(r, model) for r, model_lists in self._memory.items() for model in model_lists]
            reversed_indices = list(range(len(reversed_memory)))
            agents_token = [
                reversed_memory[i] for i in np.random.choice(reversed_indices, n_agent, replace=True)
            ]
            return agents_token
        else:
            # TODO: create a prioritized sampling method
            raise ValueError(f"rpm_sample_strategy: {self.args.rpm_sample_strategy} not available!")

    def load_rpm_models(self, agent_models: List[nn.Module], t: int=-1) -> List[nn.Module]:
        """Load the agents' model from the disk.
        Should be called by the runner to gather data from world model 
        or the real model for MARL policy training.

        Args:
            agent_models (list): list of the models to load
        """
        agent_tokens = self.sample(len(agent_models), t)
        if self.args.save_to_disk:
            for i, agent_model in enumerate(agent_models):
                score, token = agent_tokens[i][0], agent_tokens[i][1]
                path = os.path.join(self.args.local_results_path, "rpm_models", str(score), str(token))
                agent_model.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
                # model.eval()
        else:
            for i, agent_model in enumerate(agent_models):
                score, state_dict = agent_tokens[i][0], agent_tokens[i][1]
                state_dict = collections.OrderedDict({
                    param_name: th.from_numpy(weights).to(self.device).float() for param_name, weights in state_dict.items()
                })
                agent_model.load_state_dict(state_dict)
                # model.eval()
        return agent_models

    def save_memory(self):
        """Save the memory to disk
        """
        if not self.args.save_to_disk:
            for score, state_dicts in self._memory.items():
                for i, state_dict in enumerate(state_dicts):
                    path = os.path.join(self.args.local_results_path, "rpm_models", str(score), str(i))
                    os.makedirs(path, exist_ok=True)
                    th.save(state_dict, "{}/agent.th".format(path))
        
        logger.info(f"All RPM memory saved to {os.path.join(self.args.local_results_path, 'rpm_models')}")

    @property
    def memory_key_size(self):
        """Return the key size of the memory"""
        return len(self._memory.keys())

    @property
    def memory_full_size(self):
        """Return the size of the memory"""
        return sum([len(self._memory[key]) for key in self._memory.keys()])
