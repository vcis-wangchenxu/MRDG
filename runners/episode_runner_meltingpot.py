import os
import time
import copy
import random
import asyncio
import logging
import traceback
import collections

import ray
import torch as th
import numpy as np

from envs import REGISTRY as env_REGISTRY
from functools import partial

from utils.logging import Logger
from utils.logger_utils import create_logger
from components.episode_buffer import EpisodeBatch
from components.epsilon_schedules import DecayThenFlatSchedule

from ray.util.queue import Queue



class MetricManager(object):
    def __init__(self, args, queue) -> None:
        """
        A MetricManager Ray actor that runs in the background, 
        receives data from the actors and sends data to the runner
        """
        try:
            self.args = args
            self.logger = create_logger(args=args, name=__class__.__name__)
            self.create_data_loggers()
            # actors send metric to the metric manager
            self.actor_to_metric_queue = queue
            self.batch_size = self.args.batch_size_run  # the number of env actors
            
            # NOTE: v1, v2, .. can be list of values
            # {
            #   t_env: {
            #       metric: [v1, v2, ...]
            #   }
            # }
            # v1 can be scalar value or list
            self._train_data = collections.OrderedDict()  # the train metric
            self._test_data = collections.OrderedDict() # the test metric
            # the number of eval actors and train actors
            self._num_train_actors = self.batch_size
            self._num_eval_actors = args.mp_num_eval_actors
            self._total_actors = self._num_train_actors + self._num_eval_actors
            assert self._total_actors > 0 and self._num_train_actors > 0 and self._num_eval_actors
            self._count = 0
            self._recv_t_envs = set()
            self._processed_test_t_envs = set()
            self._processed_train_t_envs = set()
        except Exception as e:
            self.logger.exception(f"Error in initializing {self._class__.__name__}: {e}")
            raise

    def create_data_loggers(self):
        self.train_logger = Logger(self.logger)
        self.train_logger.setup_tb(self.args.local_results_path)
        self.train_logger.setup_json(self.args.local_results_path, name='train_results_info')
        time.sleep(1)
        self.test_logger = Logger(self.logger)
        self.test_logger.setup_tb(self.args.local_results_path)
        self.test_logger.setup_json(self.args.local_results_path, name='test_results_info')
    
    def ready(self):
        try:
            return True
        except Exception as e:
            self.logger.exception(f"Error in initializing {self._class__.__name__}: {e}")
            raise
    
    def run(self):
        """
        Running in the background
        """
        try:
            while True:
                st = time.time()
                # block until get the data
                metric = self.actor_to_metric_queue.get()
                self.logger.debug(f'{self.__class__.__name__} got data: {metric}, time cost: {(time.time()-st):.2f}')
                self._process(metric)
                self._save_data()
        except Exception as e:
            self.logger.exception(f"Error in initializing {self._class__.__name__}: {e}")
            raise

    def _process(self, metric):
        try:
            tag = list(metric.keys())[-1]
            if tag == 'train':
                data = self._train_data
                max_len = self._num_train_actors
                processed_ts = self._processed_train_t_envs
            elif tag == 'test':
                data = self._test_data
                max_len = self._num_eval_actors
                processed_ts = self._processed_test_t_envs
            else:
                raise ValueError(f"tag: {tag} is available!")

            flag = False
            for _t_env in metric[tag].keys():
                # if _t_env in processed_ts:  # skip _t_env if it has been already processed
                #     continue
                
                if _t_env not in data:
                    data[_t_env] = {}
                
                for _metric_name, val in metric[tag][_t_env].items():
                    # NOTE: val can be scalar or list of values (histgram)
                    if _metric_name not in data[_t_env]:
                        data[_t_env][_metric_name] = collections.deque(maxlen=max_len)
                    data[_t_env][_metric_name].append(val)
                    flag = True
                    self._recv_t_envs.add(_t_env)
            if flag:
                self._count += 1
        except Exception as e:
            self.logger.info(f"{self._class__.__name__}, failed to process, continue!")
    
    def _save_data(self):
        def _save(data, logger=None, tag='train_episode_return', num_actors=None, processed_ts=None):
            for _t_env in data.keys():   # t in order
                if _t_env in processed_ts:  # current _t_env processed and check next _t_env
                    continue
                if len(data[_t_env][tag]) >= int(num_actors * 0.8):
                    for k, v in data[_t_env].items():
                        logger.log_stat(k, np.mean(v, axis=0), _t_env)
                    processed_ts.add(_t_env)
                    logger.log_save(_t_env, save_interval=60000)
                    # delete the data of _t_env to save the memory
                    # del data[_t_env]
        
        _save(data=self._train_data, logger=self.train_logger, tag='train_episode_return',
              num_actors=self._num_train_actors, processed_ts=self._processed_train_t_envs)
        _save(data=self._test_data, logger=self.test_logger, tag='test_episode_return',
              num_actors=self._num_eval_actors, processed_ts=self._processed_test_t_envs)

        if len(self._processed_test_t_envs) > 0 and len(self._processed_train_t_envs) > 0 and \
            len(self._processed_test_t_envs) % 1 == 0 and len(self._processed_train_t_envs) % 1 == 0:
            self.logger.info(f"show recent stats, lasted train_t_env: {list(self._processed_train_t_envs)[-1]}, "
                             f"lasted test_t_env: {list(self._processed_test_t_envs)[-1]}")
            self.train_logger.print_recent_stats(key='train_episode_return')
            self.test_logger.print_recent_stats(key='test_episode_return')

    def get(self, t_env, get_train=True):
        """
        Called by the runner
        """
        if get_train:
            data = self._train_data
        elif not get_train:
            data = self._test_data
        else:
            raise ValueError(f"get_train: {get_train} is available!")

        if t_env in self._recv_t_envs and t_env in data:
            _date_send = copy.deepcopy(data[t_env])
            del data[t_env]
            return _date_send, True
        else:
            return None, False


class SingleRunner(object):
    def __init__(self, args, seed, scheme, groups, preprocess, mac, queue, runner_id, 
                 tag='train', runner_to_actor_queue=None, actor_to_runner_queue=None,
                 actor_to_metric_queue=None) -> None:
        """
        queue: shared by all actors, actors send data to the buffer
        runner_to_actor_queue: individual, runner sends data to each actor
        actor_to_runner_queue: individual, each actor sends data to the runner
        actor_to_metric_queue: shared by all actors, actors send data the metric manager
        """
        self.logger = create_logger(args=args, name=__class__.__name__)
        self.logger.setLevel(args.ray_actor_log_level)

        self.batch_size = args.batch_size_run  # the number of env actors
        self._num_eval_actors = args.mp_num_eval_actors  # num evaluation actors
        self._total_runner_to_actor_queue = self.batch_size + self._num_eval_actors

        try:
            self.args = args
            self.scheme = scheme
            self.groups = groups
            self.preprocess = preprocess
            self.args.seed = seed
            self.args.env_args.seed = seed
            # should sync weights from the mac of the main process, it is the private mac
            self.mac = copy.deepcopy(mac)
            if self.args.ray_actor_device == 'cuda':
                self.mac.agent.to(self.args.ray_actor_device)
            self.queue = queue
            self.tag = tag
            if self.args.use_ray:
                 # it means no actor_to_metric_queue is needed
                if args.ray_actor_mode.startswith('async') and runner_id < self._total_runner_to_actor_queue:
                    assert runner_to_actor_queue is not None, f"runner_to_actor_queue is {runner_to_actor_queue}"
                    assert actor_to_metric_queue is not None, f"actor_to_metric_queue is {actor_to_metric_queue}"
            if args.ray_actor_mode.startswith('async') and self.tag.startswith('eval') and self.args.rpm_oracle:
                assert actor_to_runner_queue is not None, f"actor_to_runner_queue is {actor_to_runner_queue}"
            self.runner_to_actor_queue = runner_to_actor_queue  # receive data from the runner
            self.actor_to_runner_queue = actor_to_runner_queue
            self.actor_to_metric_queue = actor_to_metric_queue
            self._jit_test_batch_size = 32
            self.batch_size = 1 if not self.args.run_jit_test else self._jit_test_batch_size # only ONE environment, 32 for jit test
            self._create_env()
            self.t_env = 0  # the global env
            self.runner_id = runner_id
            self._use_rpm = False  # always False in this class
            self._local_logger = None
            self.log_train_stats_t = -1000000
            self.last_log_T = 0
            self.last_test_T = -1000000
        except Exception as e:
            self.logger.exception(f"Error in initializing {self.__class__.__name__}: {e}")
            raise
    
    def _create_env(self):
        self._env = env_REGISTRY[self.args.env](**self.args.env_args.__dict__)
        env_args = copy.deepcopy(self.args.env_args)
        # self.logger.info(f"self.args.env_args: {self.args.env_args}")
        if not self.args.env.startswith('meltingpot'):
            eval_name = self.args.env
            env_args.key = self.args.env_heuristic
        else:
            eval_name = self.args.env + '_eval'
            env_args.key = self.args.mp_eval_env
        self._eval_env = env_REGISTRY[eval_name](**env_args.__dict__)
        self.episode_limit = self._env.episode_limit
        groups = {"agents": self._eval_env.n_agents}  # the number of focal agents
        # actors use cpu to collect data
        self.new_batch = partial(EpisodeBatch,
                                 scheme=self.scheme,
                                 groups=self.groups,
                                 batch_size=self.batch_size,
                                 max_seq_length=min(self.episode_limit, self.args.mp_episode_truncate_len) + 1,
                                 preprocess=self.preprocess,
                                 device="cpu" if self.args.buffer_cpu_only else self.args.device)

        self.new_batch_eval = partial(EpisodeBatch,
                                      scheme=self.scheme,
                                      groups=groups,
                                      batch_size=self.batch_size,
                                      max_seq_length=min(self.episode_limit, self.args.mp_episode_truncate_len) + 1,
                                      preprocess=self.preprocess,
                                      device="cpu" if self.args.buffer_cpu_only else self.args.device)
    
    def ready(self):
        return True
    
    def reset(self, test_mode):
        if test_mode:
            self.batch = self.new_batch_eval()
        else:
            self.batch = self.new_batch()
        if self.args.ray_actor_device == 'cuda':
            self.batch.device = self.args.ray_actor_device
            self.batch.to(self.args.ray_actor_device)
            self.mac.cuda(device=self.args.ray_actor_device)
        self._env.reset()
        self._eval_env.reset()
        self.t = 0

    def _create_new_batch(self, test_mode):
        if test_mode:
            batch = self.new_batch_eval()
        else:
            batch = self.new_batch()
        if self.args.ray_actor_device == 'cuda':
            self.batch.device = self.args.ray_actor_device
            batch.to(self.args.ray_actor_device)
        return batch

    def set_weights(self, state_dict, t_env):
        self.mac.agent.load_state_dict(state_dict)
        self.mac.agent.to(self.args.ray_actor_device)
        self.t_env = t_env
        return 1

    def run_one_episode(self, test_mode=False):
        try:
            return self._run(test_mode, env=self._eval_env if test_mode else self._env)
        except Exception as e:
            self.logger.exception(f"Error!")
            raise
    
    def ready(self):
        return True
    
    def _pre_init(self, test_mode):
        pass

    def _sync_agent_weights(self):
        # only async mode uses runner_to_actor_queue to update sync the weights
        if self.runner_to_actor_queue is not None and self.args.ray_actor_mode.startswith('async'):
            if self.runner_to_actor_queue.qsize() > 0:
                st = time.time()
                data = self.runner_to_actor_queue.get()
                if 't_env' in data:
                    self.t_env = data['t_env']
                if 'state_dict' in data:
                    self.set_weights(state_dict=data['state_dict'], t_env=data['t_env'])
                    self.logger.debug(f"{self.__class__.__name__} sync weights ok, time cost: {(time.time()-st):.2f}")
    
    def _run(self, test_mode, env=None):
        self._pre_init(test_mode)
        self._sync_agent_weights()
        self.reset(test_mode)
        # self.logger.debug(f"agent device: {self.mac.agent.get_device()}")
        self.mac.init_hidden(batch_size=self.batch_size, n_agents=env.n_agents)  # only run with one episode
        data = self._env_loop(test_mode, env)
        # send stats to runner
        self._send_stats_to_runner(data, test_mode)
        return data

    def _env_loop(self, test_mode, env):
        """
        The loop to collect one trajectory
        """
        data = {
            'episode_return': 0,
            'episode_length': 0,
            'agents_episode_return': {},
        }
        terminated = False
        count = 0
        _t = 0
        
        def _send_data(self, test_mode, _t, count, last=False, start_time=0):
            # alow the slicing in evaluation mode
            if (_t > self.args.mp_episode_truncate_len or last) and (
                    self.args.ray_actor_mode.startswith('async') or self.args.evaluate):
                count += 1
                if not test_mode:
                    try:
                        if self.args.ray_actor_device == 'cuda':
                            self.batch.to("cpu")  # make the data to CPU for better serialization
                        if self.queue.qsize() < self.queue.maxsize:
                            self.queue.put_nowait(self.batch)  # send the batches to Buffer runner
                            self.logger.debug(f"{self.__class__.__name__}: {self.runner_id}, {self.tag} (test_mode: {test_mode}) put one sub-episode ({count} in total) "
                                         f"to the queue: global t_env: {self.t_env}, time cost: {(time.time()-start_time):.2f}")
                    except Exception as e:
                        err_msg = f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}"
                        self.logger.debug(f"{self.__class__.__name__}: {self.runner_id}, {self.tag} failed to put one sub-episode "
                                     f"to the queue, qsize: {self.queue.qsize()}. "
                                     f"Global t_env: {self.t_env}, time cost: {(time.time()-start_time):.2f}"
                                     f"err_msg: {err_msg}")
                self.batch = self._create_new_batch(test_mode)
                _t = 0
                start_time = time.time()
            if self.args.mp_truncated_load_model and not test_mode:
                self._sync_agent_weights()
            return _t, count, start_time
        
        _s_st, _ep_st = time.time(), time.time()
        
        episode_rewards = []
        
        while not terminated:
            # self.logger.info(f'test_mode: {test_mode}, step/t_env: {self.t}/{self.t_env}, multi_run_counter: {env._base_env.multi_run_counter}')
            _t, count, _s_st = _send_data(self, test_mode, _t, count, start_time=_s_st)
            pre_transition_data = {}
            if self.args.env.startswith('meltingpot'):
                for k, v in env.get_state().items():
                    if self.args.marl_independent and k == 'WORLD.RGB':
                        continue
                    pre_transition_data[k] = [v] if not self.args.run_jit_test else [v] * self._jit_test_batch_size
                for k, v in env.get_obs().items():
                    if self.args.marl_independent and k == 'WORLD.RGB':
                        continue
                    pre_transition_data[k] = [v] if not self.args.run_jit_test else [v] * self._jit_test_batch_size
            else:
                pre_transition_data["state"] = [env.get_state(concated_obs=self.args.concated_obs)],
                pre_transition_data["avail_actions"] = [env.get_avail_actions()],
                pre_transition_data["obs"] = [env.get_obs()]

            bs = slice(None) if not self.args.run_jit_test else list(range(self._jit_test_batch_size))
            self.batch.update(pre_transition_data, bs=bs, ts=_t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            with th.no_grad():
                actions = self.mac.select_actions(self.batch,
                                                  t_ep=_t,
                                                  bs=bs,
                                                  t_env=self.t_env,
                                                  test_mode=test_mode,
                                                  n_agents=env.n_agents,
                                                  use_rpm=self._use_rpm)
            cpu_actions = actions.cpu().numpy()
            reward, terminated, env_info = env.step(cpu_actions[0])
            episode_rewards.append(np.sum(reward))
            data['episode_return'] += np.sum(reward)  # sum of single scalar or a list of rewards
            for ag_id, v in env_info['agents_reward'].items():  # the reward of each agent
                if ag_id not in data['agents_episode_return']:
                    data['agents_episode_return'][ag_id] = v
                else:
                    data['agents_episode_return'][ag_id] += v

            post_transition_data = {
                "actions": [cpu_actions[0]],
                "terminated": [(terminated['__all__'] if self.args.env.startswith('meltingpot') else terminated,)],
            }
            if not test_mode:  # reward is not needed in test mode, so, do not keep it in test mode
                post_transition_data["reward"] = [
                    (reward,) if not env_info['individual_rewards'] else list(env_info['agents_reward'].values())
                ]
            terminated = terminated['__all__'] if self.args.env.startswith('meltingpot') else terminated
            if self.args.run_jit_test:
                for k, v in post_transition_data.items():
                    post_transition_data[k] = v * self._jit_test_batch_size
            
            bs = slice(None) if not self.args.run_jit_test else list(range(self._jit_test_batch_size))
            self.batch.update(post_transition_data, bs=bs, ts=_t)

            self.t += 1
            _t += 1
        data['episode_length'] = self.t
        last_data = {}
        if self.args.env.startswith('meltingpot'):
            for k, v in env.get_state().items():
                if self.args.marl_independent and k == 'WORLD.RGB':
                    continue
                last_data[k] = [v] if not self.args.run_jit_test else [v] * self._jit_test_batch_size
            for k, v in env.get_obs().items():
                if self.args.marl_independent and k == 'WORLD.RGB':
                    continue
                last_data[k] = [v] if not self.args.run_jit_test else [v] * self._jit_test_batch_size
        else:
            last_data["state"] = [env.get_state(concated_obs=self.args.concated_obs)],
            last_data["avail_actions"] = [env.get_avail_actions()],
            last_data["obs"] = [env.get_obs()]  
        bs = slice(None) if not self.args.run_jit_test else list(range(self._jit_test_batch_size))
        self.batch.update(last_data, bs=bs, ts=_t)
        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch,
                                          t_ep=_t,
                                          bs=bs,
                                          t_env=self.t_env,
                                          test_mode=test_mode,
                                          n_agents=env.n_agents,
                                          use_rpm=self._use_rpm)
        bs = slice(None) if not self.args.run_jit_test else list(range(self._jit_test_batch_size))
        self.batch.update({
            "actions": [actions.cpu().numpy()[0]] 
            if not self.args.run_jit_test else [actions.cpu().numpy()[0]] * self._jit_test_batch_size
        }, bs=bs, ts=_t)
        _t, count, _s_st = _send_data(self, test_mode, _t, count, last=True, start_time=_s_st)
        if self.args.run_rpm:
            data['rpm_sample_prob'] = self.rpm_prob_schedule.eval(self.t_env)
        
        data["one_episode_time_cost_s"] = time.time() - _ep_st
        self.logger.debug(f"{self.__class__.__name__} {self.tag}: {self.runner_id} complete one full episode, "
                    f"global t_env: {self.t_env}, time cost: {(time.time()-_ep_st):.2f}")
    
        if self.args.ray_actor_mode.startswith('parallel') and not test_mode:  # In parallel mode, return the batch data
            data['batch'] = self.batch
        if self.args.ray_actor_mode.startswith('parallel') and test_mode:  # In parallel mode, return the state_dict
            data['state_dict'] = self.mac.agent.state_dict()
        return data

    def run_in_bg(self):
        """
        Run the in background, collect episodes
        """
        # async for run in bg, parallel for vecenv-like runner
        assert self.args.ray_actor_mode.startswith('async')
        
        try:
            n_episodes = 0  # interval counter
            episode = 0  # global counter
            test_mode = False if self.tag.startswith('train') else True  # test mode or train mode
            env = self._eval_env if test_mode else self._env
            prefix = f'train' if self.tag == 'train' else f'test'
            self.logger.info(f"{self.__class__.__name__} {self.tag}: {self.runner_id}, running!")
            while self.t_env <= self.args.t_max:
                if not test_mode:
                    data = self._run(test_mode=test_mode, env=env)
                elif test_mode:
                    self._sync_agent_weights()  # evaluator should sync first
                    if (self.t_env - self.last_test_T) / self.args.test_interval >= 1.0:
                        data = self._run(test_mode=test_mode, env=env)
                        self.last_test_T = self.t_env
                    else:
                        continue  # no sleep
                else:
                    raise ValueError(f'Unknow test_mode value: {test_mode}')
                
                n_episodes += 1
                episode += 1 
                # prepare the metric data and send the to the metric manager
                metric_data = {}
                metric_data[self.t_env] = {}
                metric_data[self.t_env][f'{prefix}_episode_length'] = data['episode_length']
                metric_data[self.t_env][f'{prefix}_episode_return'] = data['episode_return']
                metric_data[self.t_env][f'{prefix}_one_episode_time_cost_s'] = data['one_episode_time_cost_s']
                for k, v in data['agents_episode_return'].items():
                    metric_data[self.t_env][f'{prefix}_agent_episode_return_{k}'] = v
                if self.args.run_rpm and not test_mode:
                    metric_data[self.t_env]['train_rpm_sample_prob'] = data['rpm_sample_prob']
                if hasattr(self.mac.action_selector, "epsilon") and not test_mode:
                    metric_data[self.t_env][f'train_epsilon'] = self.mac.action_selector.epsilon
                try:
                    self.actor_to_metric_queue.put({prefix: metric_data})
                    self.logger.debug(f"{self.__class__.__name__} {self.tag}: {self.runner_id}, "
                                 f"global t_env: {self.t_env}, send data to metric manager")
                except Exception as e:
                    err_msg = f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}"
                    self.logger.debug(f"{self.__class__.__name__} {self.tag}: {self.runner_id}, "
                                 f"global t_env: {self.t_env}, failed to send data to metric manager, "
                                 f"qsize: {self.actor_to_metric_queue.qsize()}, "
                                 f"err_msg: {err_msg}")
                    # self.logger.error(f"Error in {self.__class__.__name__}, {self.tag}: {e}, env: {env}, env_agents: {env.n_agents}")
                n_episodes = 0
            self.logger.info(f"{self.__class__.__name__} {self.tag}: {self.runner_id}, exit, t_env: {self.t_env}")
        except Exception as e:
            self.logger.exception(f"Error in {self.__class__.__name__}: {e}, env: {env}, env_agents: {env.n_agents}")
            raise

    def _send_stats_to_runner(self, data, test_mode):
        # send the stats to the runner to update the RPM memory
        try:
            _st = time.time()
            send_data = {
                't_env': self.t_env,
                'episode_return': data['episode_return'],
                'state_dict': self.mac.agent.state_dict(),
            }
            _can_send = False
            if self.args.rpm_oracle and test_mode:  # oracle and test_mode
                _can_send = True
            elif not self.args.rpm_oracle and not test_mode:  # non-oracle and train_mode
                _can_send = True
            # send data to update the runner's RPM
            if self.actor_to_runner_queue is not None and _can_send:
                self.actor_to_runner_queue.put_nowait(send_data)
                self.logger.info(f"{self.__class__.__name__}, tag: {self.tag}, test_mode: {test_mode}, "
                                 f"runner_id: {self.runner_id}, global t_env: {self.t_env}, "
                                 f"sent t_env: {send_data['t_env']} and episode_return: {send_data['episode_return']}, " 
                                 f"time cost: {(time.time()-_st):.2f}")
        except Exception as e:
            self.logger.debug(f"{self.__class__.__name__} {self.tag}: {self.runner_id} failed to put data to the runner, err msg: {e}. "
                        f"Global t_env: {self.t_env}, time cost: {(time.time()-_st):.2f}")


class SingleRPMRunner(SingleRunner):
    def __init__(self, args, seed, scheme, groups, preprocess, mac, queue, runner_id,
                 tag='train', runner_to_actor_queue=None, actor_to_runner_queue=None,
                 actor_to_metric_queue=None) -> None:        
        self.logger = create_logger(args=args, name=__class__.__name__)
        self.logger.setLevel(args.ray_actor_log_level)
        super().__init__(args, seed, scheme, groups, preprocess, mac, queue,
                         runner_id, tag, runner_to_actor_queue,
                         actor_to_runner_queue, actor_to_metric_queue)
        self.rpm_prob_schedule = DecayThenFlatSchedule(start=args.rpm_sample_prob_start,
                                                       finish=args.rpm_sample_prob_finish,
                                                       time_length=args.rpm_sample_prob_anneal_time,
                                                       decay="linear")
        self.rpm_sample_prob = self.rpm_prob_schedule.eval(0)
        self._use_rpm = False

    def _sync_agent_weights(self):
        if self.args.ray_actor_mode.startswith('async') and \
            self.runner_to_actor_queue.qsize() > 0 and \
                self.runner_to_actor_queue is not None:
            # sync weights from the runner via Queue
            st = time.time()
            data = self.runner_to_actor_queue.get()
            if 't_env' in data:
                self.t_env = data['t_env']
            if 'state_dict' in data:
                state_dict = data['state_dict']
                self.mac.agent.load_state_dict(state_dict)
                self.mac.agent.to(self.args.ray_actor_device)
            if 'rpm_memory' in data:
                rpm_memory = data['rpm_memory']
                self.sync_rpm_memory(rpm_memory)
            self.logger.debug(f"sync weights ok, time cost: {(time.time()-st):.2f}")

    def _pre_init(self, test_mode):
        self._init_rpm(test_mode)

    def sync_rpm_memory(self, rpm_memory):
        """Called by the main runner and will be periodically updated"""
        self.mac.rpm.sync_memory(rpm_memory)
        for agent in self.mac.rpm_agents:  # to device
            if self.args.ray_actor_device == 'cuda':
                agent.to(self.args.ray_actor_device)
        return True

    def _init_rpm(self, test_mode):
        # the policy of agents can be MARL policy or RPM policy
        # get the mode of this episode
        if not test_mode and self.args.run_rpm:  # while training, sampling policies from marl or rpm policies
            self.rpm_sample_prob = self.rpm_prob_schedule.eval(self.t_env)
            if random.random() > self.rpm_sample_prob:
                rpmpolicy_or_marlpolicy = 'marl'
                self._use_rpm = False
            else:
                if self.t_env >= self.args.rpm_start_to_use and self.mac.rpm.memory_key_size >= 1:
                    # (self.mac.rpm.memory_key_size >= self.args.n_agents 
                        # if self.args.env.startswith('meltingpot') else self.mac.rpm.memory_key_size >=1):
                    # NOTE: only when there are n_agents keys can it be sampled
                    rpmpolicy_or_marlpolicy = 'rpm'
                    self._use_rpm = True
                else:
                    rpmpolicy_or_marlpolicy = 'marl'
                    self._use_rpm = False
        else:
            # evaluation mode, use the current policy to evaluate on heuristic_env and the real env
            rpmpolicy_or_marlpolicy = 'marl'
            self._use_rpm = False

        if rpmpolicy_or_marlpolicy == 'rpm':
            # sample policies from the environment and use them to run the episode
            self.mac.sample_rpm(self.t_env)
            self._use_rpm = True


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run  # the number of env actors

        self.args.env_args.args = self.args
        self.args.env_args.seed += 1
        self._env = env_REGISTRY[self.args.env](**self.args.env_args.__dict__)
        self.episode_limit = self._env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.use_ray = self.args.use_ray

        # Log the first run
        self.log_train_stats_t = -1000000
        self._num_eval_actors = args.mp_num_eval_actors  # num evaluation actors
        self._total_runner_to_actor_queue = self.batch_size + self._num_eval_actors
        self._runner_to_actor_queues = [None] * self._total_runner_to_actor_queue

    def setup(self, scheme, groups, preprocess, mac, queue=None):
        self._scheme = scheme
        self._groups = groups
        self._preprocess = preprocess
        self.mac = mac
        self.queue = queue
        self.actors = self._create_env_runners(num_actors=self.batch_size,
                                               use_gpu=self.args.ray_actor_device=="cuda")
        self._post_create_env_runners()

    def _post_create_env_runners(self):
        pass

    def _create_env_runners(self, num_actors=10, use_gpu=False, tag='train'):
        if self.args.run_rpm:
            cls = SingleRPMRunner
        else:
            cls = SingleRunner
        
        if tag == 'train':
            offset = 0
        elif tag == 'eval':
            offset = self.batch_size
        else:
            raise ValueError(f'Unknown tag: {tag}')
        
        st = time.time()
        if self.args.use_ray:
            if use_gpu:
                num_gpus = 0.05
            else:
                num_gpus = 0
            
            actors = [
                ray.remote(cls).options(name=f"SingleRunner_{tag}_{i}",
                                        num_cpus=self.args.ray_actor_num_cpus,
                                        max_concurrency=4,
                                        num_gpus=num_gpus).remote(
                    args=self.args,
                    seed=self.args.env_args.seed+i,
                    scheme=self._scheme,
                    groups=self._groups,
                    preprocess=self._preprocess,
                    mac=self.mac,
                    queue=self.queue,
                    runner_id=offset+i,
                    tag=tag,
                    runner_to_actor_queue=self._runner_to_actor_queues[offset+i],
                    actor_to_runner_queue=self._actors_to_runner_queue[offset+i],
                    actor_to_metric_queue=self._actor_to_metric_queue,
                ) for i in range(num_actors)
            ]
            ready = ray.get([a.ready.remote() for a in actors])
            self.logger.console_logger.info(f'{self.__class__.__name__}, all {num_actors} env {tag} '
                                            f'actors are ready ({ready}), time costs: {(time.time()-st):.2f}')
        else:
            actors = [
                cls(
                    args=self.args,
                    seed=self.args.env_args.seed+i,
                    scheme=self._scheme,
                    groups=self._groups,
                    preprocess=self._preprocess,
                    mac=self.mac,
                    queue=self.queue,
                    runner_id=offset+i,
                    tag=tag,
                    runner_to_actor_queue=self._runner_to_actor_queues[offset+i],
                    actor_to_runner_queue=self._actors_to_runner_queue[offset+i],
                    actor_to_metric_queue=self._actor_to_metric_queue,
                ) for i in range(num_actors)
            ]
            self.logger.console_logger.info(f'{self.__class__.__name__}, all {num_actors} env {tag} '
                                            f'actors are ready (use_ray={self.args.use_ray}), time costs: {(time.time()-st):.2f}')
        return actors

    def get_env_info(self):
        return self._env.get_env_info()

    def save_replay(self):
        self._env.save_replay()

    def _sync_weigths_to_actors(self, actors=None):
        """set the weigths to all actors"""
        st = time.time()
        self.mac.agent.to("cpu") # agents use CPU device
        if self.args.use_ray:
            res = ray.get([a.set_weights.remote(self.mac.agent.state_dict(), self.t_env) for a in actors])
        else:
            res = [a.set_weights(self.mac.agent.state_dict(), self.t_env) for a in actors]
        self.mac.agent.to(self.args.device)
        self.logger.console_logger.info(f'{self.__class__.__name__} sync weights, time costs: {(time.time()-st):.2f}')
        assert sum(res) == len(actors), f"Some actors have not finished the sync, check it res: {res}"
        
        if self.args.run_rpm:
            if self.args.use_ray:
                res = ray.get([actor.sync_rpm_memory.remote(self.mac.rpm.get_memory()) for actor in actors])
            else:
                res = [actor.sync_rpm_memory(self.mac.rpm.get_memory()) for actor in actors]
            assert sum(res) == len(actors), f"Some actors have not finished the sync, check it res: {res}"
            self.logger.console_logger.info(f'{self.__class__.__name__} sync RPM memory ok, time costs: {(time.time()-st):.2f}')

    def close_env(self):
        self._env.close()

    def reset(self):
        # keep it and do not call it
        self._env.reset()
        self.t = 0

    def update(self, test_mode=True):
        # set the weights
        self._sync_weigths_to_actors(actors=self.eval_actors)
    
    def run(self, test_mode=False):
        # set the weights
        if not self.args.run_jit_test and self.args.ray_actor_mode.startswith('async'):
            self._sync_weigths_to_actors()
        
        st = time.time()
        if self.args.use_ray:
            data = ray.get([actor.run_one_episode.remote(test_mode) for actor in self.actors])
        else:
            data = [actor.run_one_episode(test_mode) for actor in self.actors]
        self.logger.console_logger.debug(f'{self.__class__.__name__} got {len(self.actors)} episodes, '
                                        f'mode: {"evaluation" if test_mode else "train"}, '
                                        f'time costs: {(time.time()-st):.2f}')

        episode_lengths = [d['episode_length'] for d in data]
        episode_returns = [d['episode_return'] for d in data]
        # episodes = [d['episodes'] for d in data]

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats["n_episodes"] = len(episode_returns)
        cur_stats["ep_length"] = sum(episode_lengths)
        
        if self.args.run_rpm:
            cur_stats["rpm_sample_prob"] = np.sum([d['rpm_sample_prob'] for d in data])
        
        if not test_mode:
            self.t_env += sum(episode_lengths)

        cur_returns.extend(episode_returns)

        if hasattr(self.mac.action_selector, "epsilon"):
            self.mac.action_selector.epsilon = self.mac.action_selector.schedule.eval(self.t_env)

        if test_mode and (len(episode_returns) == len(self.actors)):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        if self.args.ray_actor_mode.startswith('parallel'):
            res_data = [d['batch'] for d in data]
        else:
            res_data = None
        return res_data

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


class AsyncEpisodeRunner(EpisodeRunner):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        if self.args.ray_actor_mode.startswith('async'):
            self._actor_to_metric_queue = Queue(maxsize=100000)
            if self.args.rpm_oracle:
                self._actors_to_runner_queue = [None] * self.batch_size + \
                    [Queue(maxsize=100) for _ in range(self._num_eval_actors)]
            else:
                self._actors_to_runner_queue = [
                    Queue(maxsize=100) for _ in range(self._num_eval_actors)
                ] + [None] * self.batch_size
            self._runner_to_actor_queues = [
                Queue(maxsize=20) for _ in range(self._total_runner_to_actor_queue)
            ]
        elif self.args.ray_actor_mode.startswith('parallel'):
            self._actor_to_metric_queue = None
            self._actors_to_runner_queue = [None] * (self.batch_size + self._num_eval_actors)
            self._runner_to_actor_queues = [None] * self._total_runner_to_actor_queue
        else:
            raise ValueError(f'Unknown ray_actor_mode: {self.args.ray_actor_mode}')

    def _init_metric_manager(self):
        if not self.args.use_ray:
            return  # use_ray=False for debugging purpose
        
        self._metric_manager = \
            ray.remote(MetricManager).options(name=f"MetricManager",
                                              num_cpus=1,
                                              max_concurrency=2,
                                              num_gpus=0).remote(
            args=self.args,
            queue=self._actor_to_metric_queue
        )
        st = time.time()
        flag = ray.get(self._metric_manager.ready.remote())
        self.logger.console_logger.info(f"{self.__class__.__name__} launched metric manager, ready={flag}, "
                                        f"time cost: {(time.time()-st):.2f}")
        self._metric_manager.run.remote()
    
    def _post_create_env_runners(self):
        self._create_eval_env_runners()
        actors = self.actors + self.eval_actors
        if self.args.ray_actor_mode.startswith('async'):
            self._put_train_actors_in_bg(actors)
            self._init_metric_manager()
        elif self.args.ray_actor_mode.startswith('parallel'):
            self.logger.console_logger.info(f"All actors are not ran in background and no metric manager initialized")

    def _put_train_actors_in_bg(self, actors):
        # make the actors continue runing in the background
        if not self.args.use_ray:
            return  # use_ray=False for debugging purpose
        
        st = time.time()
        self.ws = [actor.run_in_bg.remote() for actor in actors]
        self.logger.console_logger.info(f"{self.__class__.__name__} launched running actors: {self.ws}, "
                                        f"time cost: {(time.time()-st):.2f}")
    
    def _create_eval_env_runners(self):
        self.eval_actors = self._create_env_runners(
            num_actors=self._num_eval_actors,
            use_gpu=self.args.ray_eval_actor_device=='cuda',
            tag='eval'
        )

    def ray_worker_stats(self):
        # TODO: show worker stats
        pass

    def run(self, test_mode=False):
        self.sync_weights(test_mode)
        if test_mode:
            return self._evaluate(test_mode)
        return super().run(test_mode=test_mode)

    def sync_weights(self, test_mode=False):
        if self.args.ray_actor_mode.startswith('parallel'):
            super()._sync_weigths_to_actors(actors=self.eval_actors if test_mode else self.actors)
        elif self.args.ray_actor_mode.startswith('async'):
            self._sync_weigths_to_actors()
        else:
            raise ValueError(f'Unknwon self.args.ray_actor_mode: {self.args.ray_actor_mode}')

    def _sync_weigths_to_actors(self, no_weights=False):
        """sync the weigths to all actors"""
        assert self.args.ray_actor_mode.startswith('async')
        # use queue to send
        data = {
            't_env': self.t_env,
        }

        if not no_weights:
            self.mac.agent.to("cpu")  # agents use CPU device
            data['state_dict'] = self.mac.agent.state_dict()
        
        if self.args.run_rpm:
            data['rpm_memory'] = self.mac.rpm.get_memory()

        st = time.time()
        try:
            for q in self._runner_to_actor_queues:  # broadcast data to all actors
                q.put_nowait(data)
        except Exception as e:
            self.logger.console_logger.info(f"{self.__class__.__name__}, error, failed to send weights and "
                         f"RPM ({self.args.run_rpm}) to actors. Continue!")
        
        self.mac.agent.to(self.args.device)
        self.logger.console_logger.info(f'{self.__class__.__name__} sync agent weights and '
                                        f'RPM ({self.args.run_rpm}) ok, time costs: {(time.time()-st):.2f}')

    def update_rpm(self, test_mode):
        assert self.args.ray_actor_mode.startswith('async')
        self._rpm_temp_data = collections.OrderedDict()  #  key: t_env, value: episode_returns
        if not hasattr(self, "_rpm_processed_t_env"):
            self._rpm_processed_t_env = set()  # save processed t_env
        # receive the data from the remote and update the RPM
        st = time.time()
        _new_keys = []
        _recv_t_envs = set()
        for q in self._actors_to_runner_queue:
            if q is not None and q.size() > 0:
                data = q.get()
                t_env, episode_return, state_dict = data['t_env'], data['episode_return'], data['state_dict']
                if t_env in self._rpm_processed_t_env:
                    # t_env has been process, at this time step, the received
                    # t_env maybe the reset t_env, it not important, so, discard it
                    continue
                _recv_t_envs.add(t_env)
                _new_keys.append({'t_env': t_env, 'episode_return': episode_return})
                if t_env not in self._rpm_temp_data:
                    self._rpm_temp_data[t_env] = [(episode_return, state_dict)]
                else:
                    self._rpm_temp_data[t_env].append((episode_return, state_dict))
        self.logger.console_logger.info(f'{self.__class__.__name__} t_env: {self.t_env}, get episode_return data: {_new_keys} '
                                        f'from all eval actors, time costs: {(time.time()-st):.2f}')
        
        # send only rpm to remote actors
        self._sync_weigths_to_actors(no_weights=True)

        st = time.time()
        updated_key = []
        for t_env in self._rpm_temp_data.keys():
            if len(self._rpm_temp_data[t_env]) >= 1:
                _t_env_state_dicts = self._rpm_temp_data[t_env][0][1] if \
                    self.args.agent.endswith('_ns') else [self._rpm_temp_data[t_env][0][1]]
                
                _mean_episode_return = np.mean([ret for ret, _ in self._rpm_temp_data[t_env]])
                for _state_dict in _t_env_state_dicts:
                    self.mac.rpm.update(
                        t_env=t_env,
                        checkpoint={
                            "reward": _mean_episode_return,
                            "state_dict": _state_dict  # get the first item's state_dict
                        }
                    )
                updated_key.append(t_env)
        # save the processed t_env
        for _t_env in _recv_t_envs:
            self._rpm_processed_t_env.add(_t_env)
        self.logger.console_logger.info(f'{self.__class__.__name__} t_env: {self.t_env}, update RPM memory with keys: {updated_key} '
                                        f'with collected data, time costs: {(time.time()-st):.2f}')
    
    def update(self, test_mode):
        self.sync_weights(test_mode)

    def _evaluate(self, test_mode):
        st = time.time()
        if self.args.use_ray:
            data = ray.get([actor.run_one_episode.remote(test_mode) for actor in self.eval_actors])
        else:
            data = [actor.run_one_episode(test_mode) for actor in self.eval_actors]
        self.logger.console_logger.info(f'{self.__class__.__name__} got {len(self.eval_actors)} episodes, '
                                        f'mode: {"evaluation" if test_mode else "train"}, '
                                        f'time costs: {(time.time()-st):.2f}')

        episode_lengths = [d['episode_length'] for d in data]
        episode_returns = [d['episode_return'] for d in data]

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats["n_episodes"] = len(episode_returns)
        cur_stats["ep_length"] = sum(episode_lengths)
        
        if self.args.run_rpm and not test_mode:
            cur_stats["rpm_sample_prob"] = np.mean([d['rpm_sample_prob'] for d in data])
        
        if self.args.ray_actor_mode.startswith('parallel') and not self.args.evaluate:
            state_dicts = [d['state_dict'] for d in data]
            if not self.args.env.startswith('meltingpot') and np.mean(episode_returns) == 0:
                pass
            else:
                state_dicts = state_dicts if \
                    self.args.agent.endswith('_ns') else [state_dicts[0]]
                _mean_episode_return = np.mean(episode_returns)
                for _state_dict in state_dicts:
                    self.mac.rpm.update(
                        t_env=self.t_env,
                        checkpoint={
                            "reward": _mean_episode_return,
                            "state_dict": _state_dict  # get the first item's state_dict
                    })

        cur_returns.extend(episode_returns)

        if hasattr(self.mac.action_selector, "epsilon"):
            self.mac.action_selector.epsilon = self.mac.action_selector.schedule.eval(self.t_env)

        if test_mode and (len(self.test_returns) == self._num_eval_actors):
            self._log(cur_returns, cur_stats, log_prefix)
            if self.args.evaluate:
                self.logger.log_save(t_env=1000, save_interval=3)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return None
