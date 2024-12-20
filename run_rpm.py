import os
import copy
import time
import pprint
import datetime
import threading

import ray
import torch as th
import numpy as np

from types import SimpleNamespace as SN
from utils.logging import Logger, save_params

from utils.os_utils import generate_id
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from utils.logger_utils import create_logger
from utils.platform_info import save_hardware_info
from utils.cfg_utils import RecursiveNamespace


def run(_config, _log):

    args = RecursiveNamespace(**_config)
    args.local_results_path = os.path.join(args.local_results_path, generate_id(args.local_results_path))
    os.makedirs(args.local_results_path, exist_ok=True)
    args.device = "cuda" if args.use_cuda else "cpu"

    save_hardware_info(args.local_results_path)
    
    # setup loggers
    _log = create_logger(args=args, name='root')
    logger = Logger(_log)
    # check args sanity
    _config = args_sanity_check(_config, logger)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    save_params(args.local_results_path, _config)
    
    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        logger.setup_tb(args.local_results_path)

    # sacred is on by default
    logger.setup_json(directory_name=args.local_results_path)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):

    ctx = ray.init(ignore_reinit_error=False,
                   include_dashboard=False,
                   log_to_driver=True)

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.n_foods = env_info["n_foods"]
    args.mp_episode_truncate_len = env_info["episode_limit"]
    args.env_info = env_info

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (args.n_agents if args.env_args.individual_rewards else 1,)},
        "reward_type": {"vshape": (1,)},   # 0, 1, 2, ...
        "agent_food_dist": {"vshape": (args.n_foods,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
        args=args
    )

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)  # use macs
    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    if args.use_cuda:
        learner.cuda()
    
    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_train_wm_T = 0
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time in the base env
        episode_batches = runner.run(test_mode=False)
        for episode_batch in episode_batches:
            buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = sample_batch(buffer, args, args.batch_size)
            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            last_test_T = runner.t_env
            _batch = runner.run(test_mode=True)  # evaluate in base env

        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_log_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            logger.print_recent_stats()
            logger.log_save(runner.t_env, save_interval=1000)
            last_log_T = runner.t_env

    logger.log_save(runner.t_env, save_interval=1000)
    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.console_logger.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config


def sample_batch(buffer, args, batch_size):
    episode_sample = buffer.sample(batch_size)

    # Truncate batch to only filled timesteps
    max_ep_t = episode_sample.max_t_filled()
    episode_sample = episode_sample[:, :max_ep_t]

    if episode_sample.device != args.device:
        episode_sample.to(args.device)
    return episode_sample
