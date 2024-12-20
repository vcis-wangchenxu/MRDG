import copy
import time
import random
import logging

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np

from components.epsilon_schedules import DecayThenFlatSchedule

logger = logging.getLogger(__name__)


class EpisodeRunnerRPM:
    """For the MARL Runner
    """
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args.__dict__)
        
        assert self.args.env_heuristic[:-7] == self.args.env_args.key[:-2], \
            "heuristic env and real env must be the same"
        
        env_args = copy.deepcopy(self.args.env_args)
        env_args.key = self.args.env_heuristic
        self.heuristic_env = env_REGISTRY[self.args.env](**env_args.__dict__)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        self.rpm_prob_schedule = DecayThenFlatSchedule(start=args.rpm_sample_prob_start,
                                                       finish=args.rpm_sample_prob_finish,
                                                       time_length=args.rpm_sample_prob_anneal_time,
                                                       decay="linear")
        self.rpm_sample_prob = self.rpm_prob_schedule.eval(0)

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.heuristic_env.reset()
        self.t = 0

    def run(self, test_mode=False, run_which_one="base"):
        if not test_mode:
            return self._run(test_mode=test_mode, env=self.env, heuristic_env=False)
        else:
            batch = None
            if run_which_one == 'base':
                # evaluation mode, use the current policy to evaluate on heuristic_env and the real env
                return self._run(test_mode=test_mode, env=self.env, heuristic_env=False)
            elif run_which_one == 'heuristic':
                self._run(test_mode=test_mode, env=self.heuristic_env, heuristic_env=True)  # test env one by one
            else:
                raise ValueError(f"env type ({run_which_one}) not found!")
        return batch  # only returns the batch generated by the real env

    def _run(self, test_mode=False, env=None, heuristic_env=False):
        self.reset()

        # the policy of agents can be MARL policy or RPM policy
        # get the mode of this episode
        if not test_mode:  # while training, sampling policies from marl or rpm policies
            self.rpm_sample_prob = self.rpm_prob_schedule.eval(self.t_env)
            if random.random() > self.rpm_sample_prob:
                rpmpolicy_or_marlpolicy = 'marl'
            else:
                if self.t_env < self.args.rpm_start_to_use or self.mac.rpm.memory_key_size < 1:
                    rpmpolicy_or_marlpolicy = 'marl'
                else:
                    rpmpolicy_or_marlpolicy = 'rpm'
        else:
            # evaluation mode, use the current policy to evaluate on heuristic_env and the real env
            rpmpolicy_or_marlpolicy = 'marl'

        if rpmpolicy_or_marlpolicy == 'rpm':
            # sample policies from the environment and use them to run the episode
            self.mac.sample_rpm()

        if not test_mode:
            assert not heuristic_env, "heuristic env should not be used in training"

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [env.get_state(concated_obs=self.args.concated_obs)],
                "avail_actions": [env.get_avail_actions()],
                "obs": [env.get_obs()]
            }

            avail_actions = pre_transition_data['avail_actions']
            assert np.array(avail_actions).sum() == np.prod(np.array(avail_actions).shape), \
                f"avail_actions: {avail_actions}"

            self.batch.update(pre_transition_data, ts=self.t)

            # NOTE that the actions can be output by the policy to be train in MARL or the RPM MARL policies
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            if rpmpolicy_or_marlpolicy == 'marl':
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            elif rpmpolicy_or_marlpolicy == 'rpm':
                actions = self.mac.rpm_select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            else:
                raise ValueError("rpmpolicy_or_marlpolicy can only be 'marl' or 'rpm'")

            cpu_actions = actions.cpu().numpy()[0]
            reward, terminated, env_info = env.step(cpu_actions)
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            if test_mode and self.args.sleep_render_sec > 0:
                logger.info(f"t: {self.t} actions: {cpu_actions}, reward: {reward}, terminated: {terminated}")
                env.render()
                time.sleep(self.args.sleep_render_sec)

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [env.get_state()],
            "avail_actions": [env.get_avail_actions()],
            "obs": [env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = ("test_heuristic_" if heuristic_env else "test_base_") if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:  # NOTE: do not increase the self.t_env in test mode
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
