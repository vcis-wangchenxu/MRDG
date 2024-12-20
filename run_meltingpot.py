import os
import copy
import time
import pprint
import logging
import datetime
import threading

import ray
import tqdm
import torch as th
import numpy as np

from ray.util.queue import Queue

from types import SimpleNamespace as SN

from utils.logging import Logger, save_params
from utils.timehelper import time_left, time_str
from utils.os_utils import generate_id
from utils.print_time import TimeLeft
from os.path import dirname, abspath

from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer, ReplayBufferwithQueue
from components.transforms import OneHot
from utils.logger_utils import create_logger
from utils.platform_info import save_hardware_info
from utils.cfg_utils import RecursiveNamespace


cfgs = {
    'clean_up': {
        'mp_sync_counts': 20,
        'rpm_reward_slice': 1,
    },
    'stag_hunt_in_the_matrix': {
        'mp_sync_counts': 10,
        'rpm_reward_slice': 1,
    },
    'chicken_in_the_matrix': {
        'mp_sync_counts': 5,
        'rpm_reward_slice': 1,
    },
    'pure_coordination_in_the_matrix': {
        'mp_sync_counts': 20,
        'rpm_reward_slice': 0.01,
    },
    'prisoners_dilemma_in_the_matrix': {
        'mp_sync_counts': 20,
        'rpm_reward_slice': 0.02,
    },
    'rationalizable_coordination_in_the_matrix': {
        'mp_sync_counts': 20,
        'rpm_reward_slice': 0.2,
    },
}


def run(_config, _log=None):

    # update the cfgs
    if _config['mp_cfg_auto']:
        _config['mp_sync_counts'] = cfgs[_config['env_args']['key']]['mp_sync_counts']
        _config['rpm_reward_slice']= cfgs[_config['env_args']['key']]['rpm_reward_slice']

    args = RecursiveNamespace(**_config)
    args.local_results_path = os.path.join(args.local_results_path, generate_id(args.local_results_path))
    os.makedirs(args.local_results_path, exist_ok=True)
    args.device = "cuda" if args.use_cuda else "cpu"

    # save information
    save_hardware_info(args.local_results_path)

    # setup loggers
    _log = create_logger(args=args, name='root')
    logger = Logger(_log)
    # check args sanity
    _config = args_sanity_check(_config, logger)
    logger.console_logger.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    logger.console_logger.info("\n\n" + experiment_params + "\n")

    save_params(args.local_results_path, _config)

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        logger.setup_tb(args.local_results_path)

    # sacred is on by default
    logger.setup_json(directory_name=args.local_results_path)

    # Run and train
    try:
        run_sequential(args=args, logger=logger)
    except:
        logger.console_logger.exception('Error!')
        time.sleep(5)
        raise
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

    if args.use_ray_cluster:
        ctx = ray.init("auto", ignore_reinit_error=True, 
                       include_dashboard=False, log_to_driver=True) 
    else:  # it is ok to launch ray without using ray
        ctx = ray.init(ignore_reinit_error=False,
                       include_dashboard=False,
                       log_to_driver=True)

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args = post_update_args(args, env_info)
    
    timer = TimeLeft(args=args)
        
    scheme, groups, preprocess = create_scheme(args, env_info)
    buffer, queue, buffer_queue, ray_ws = create_buffer(args, scheme, groups, env_info, preprocess, logger, ctx)
    
    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer_scheme(args, buffer), groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, queue=queue)
    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer_scheme(args, buffer), logger, args)
    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            evaluate_sequential(args, runner)
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_train_wm_T = 0
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    wait_for_ready(args, env_info, buffer)
    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    train_update_count = 0
    while runner.t_env <= args.t_max:
        one_round_st = time.time()
        serial_buffer_update(args=args, runner=runner, queue=queue, buffer=buffer, buffer_queue=buffer_queue, logger=logger)
        
        train_update_count, episode = learn(args, env_info, runner, buffer, learner,
                                            buffer_queue, timer, logger, episode, train_update_count)
        
        if args.run_rpm:
            runner.update_rpm(test_mode=False)

        # sync model to remote actors
        if (train_update_count + 1) % args.mp_sync_counts == 0:
            runner.update(test_mode=True)
        
        if args.run_rpm:
            runner.update_rpm(test_mode=False)
        
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

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

        logger.console_logger.info(f"One round time cost: {(time.time()-one_round_st):.2f}") 

    runner.close_env()
    ray_stop(args, ray_ws)
    logger.console_logger.info("Finished Training")


def learn(args, env_info, runner, buffer, learner, buffer_queue, timer, logger, episode=0, train_update_count=0):
    if args.sample_scheme.startswith('normal'):
        if can_sample(args, env_info, buffer, batch_size=args.batch_size):
            episode_sample = sample_batch(buffer, args, args.batch_size, same_device=True, buffer_queue=buffer_queue, logger=logger)
            st = time.time()
            learner.train(episode_sample, runner.t_env)
            logger.console_logger.info(f"Train sample: {(time.time() - st):.2f}")
            timer.print_time_left(logger, runner.t_env)
            train_update_count += 1
    elif args.sample_scheme.startswith('async'):
        if can_sample(args, env_info, buffer, batch_size=args.buffer_size):
            async_sample_learn(args, runner, buffer, learner, buffer_queue, timer, logger)
            train_update_count += 1
    episode = update_info(args, episode, runner)
    return train_update_count, episode


def can_sample_v2(args, env_info, buffer):
    if args.sample_scheme.startswith('normal'):
        return can_sample(args, env_info, buffer, batch_size=args.batch_size)
    elif args.sample_scheme.startswith('async'):
        return can_sample(args, env_info, buffer, batch_size=args.buffer_size)
    else:
        raise ValueError(f"Unknown args.sample_scheme: {args.sample_scheme}")


def wait_for_ready(args, env_info, buffer):
    count = 0
    if not args.use_ray:
        # no ray for collecting resources, no asynchronous running
        return
    while True:
        if can_sample_v2(args, env_info, buffer):
            break
        else:
            time.sleep(60)
            count += 1
        if count > 10:
            break


def async_sample_learn(args, runner, buffer, learner, buffer_queue, timer, logger):
    """
    create many futures and ray.get the first two samples
    """
    learner.sample_counts = args.buffer_size // args.batch_size
    if not hasattr(learner, 'obj_refs') and args.use_ray:
        # get futures
        learner.obj_refs =  [
            buffer.sample_ids.remote(ep_ids=list(
                range(i*args.batch_size,
                      args.buffer_size if i == learner.sample_counts - 1 else (i+1)*args.batch_size)))
            for i in range(learner.sample_counts)
        ]
    
    def _default_v1(post_sample_batch):
        # get sample
        time_costs = []
        time_costs_ray_get = []
        for i in range(learner.sample_counts):
            st = time.time()
            if not args.use_ray:
                episode_sample = buffer.sample_ids(
                    ep_ids=list(range(i*args.batch_size,
                                      args.buffer_size if i == learner.sample_counts - 1 else (i+1)*args.batch_size)
                                )
                )
                episode_sample = post_sample_batch(args, episode_sample, truncate=True, same_device=True)
                time_costs_ray_get.append(time.time() - st)
                logger.console_logger.info(f"Serial got sample {i}, {episode_sample}, time cost: {time_costs_ray_get[-1]:.2f}")
            else:
                episode_sample = ray.get(learner.obj_refs[i])
                time_costs_ray_get.append(time.time() - st)
                logger.console_logger.info(f"Async got sample {i}, {episode_sample}, time cost: {(time.time() - st):.2f}")
                episode_sample = post_sample_batch(args, episode_sample, truncate=True, same_device=True)
                # put new obj_ref to obj_refs, no time cost
                learner.obj_refs[i] = \
                    buffer.sample_ids.remote(
                        ep_ids=list(range(
                                    i*args.batch_size,
                                    args.buffer_size if i == learner.sample_counts - 1 else (i+1)*args.batch_size))
                    )
            # learn sample
            st = time.time()
            # MARL learning or MBMARL learner
            learner.train(episode_sample, runner.t_env)
            # th.cuda.synchronize(device=th.device("cuda"))
            time_costs.append(time.time() - st)
            
            logger.console_logger.info(f"Learn sample {i}, time cost: {(time.time() - st):.2f}")
        timer.print_time_left(logger, runner.t_env)
        runner.logger.log_stat("train_sec_per_update_mean", np.mean(time_costs), runner.t_env)
        runner.logger.log_stat("train_sec_per_ray_get", np.mean(time_costs_ray_get), runner.t_env)
    
    def _default_v2(post_sample_batch):
        # get sample
        time_costs = []
        time_costs_ray_get = []
        for i in range(learner.sample_counts):
            st = time.time()
            episode_sample = ray.get(learner.obj_refs[i])
            time_costs_ray_get.append(time.time() - st)
            logger.console_logger.info(f"Async got sample {i}, {episode_sample}, time cost: {(time.time() - st):.2f}")
            episode_sample = post_sample_batch(args, episode_sample, truncate=True, same_device=True)
            # learn sample
            st = time.time()
            learner.train(episode_sample, runner.t_env)
            time_costs.append(time.time() - st)
            logger.console_logger.info(f"Learn sample {i}, time cost: {(time.time() - st):.2f}")
            # put new obj_ref to obj_refs, no time cost
            learner.obj_refs[i] = \
                buffer.sample_ids.remote(
                    ep_ids=list(range(
                                i*args.batch_size,
                                args.buffer_size if i == learner.sample_counts - 1 else (i+1)*args.batch_size))
                )
        timer.print_time_left(logger, runner.t_env)
        runner.logger.log_stat("train_sec_per_update_mean", np.mean(time_costs), runner.t_env)
        runner.logger.log_stat("train_sec_per_ray_get", np.mean(time_costs_ray_get), runner.t_env)

    def _default_v3(post_sample_batch):
        # get sample
        st = time.time()
        episode_samples = [ray.get(obj_ref) for obj_ref in learner.obj_refs]
        logger.console_logger.info(f"Get {len(episode_samples)} episode_samples, time cost: {(time.time() - st):.2f}")
        time_costs = []
        for i, episode_sample in enumerate(episode_samples):
            episode_sample = post_sample_batch(args, episode_sample, truncate=True, same_device=True)
            # learn sample
            st = time.time()
            learner.train(episode_sample, runner.t_env)
            th.cuda.synchronize()
            time_costs.append(time.time() - st)
            logger.console_logger.info(f"Learn sample {i}, time cost: {(time.time() - st):.2f}")
            # put new obj_ref to obj_refs, no time cost
            learner.obj_refs[i] = \
                buffer.sample_ids.remote(
                    ep_ids=list(range(
                                i*args.batch_size,
                                args.buffer_size if i == learner.sample_counts - 1 else (i+1)*args.batch_size))
                )
        timer.print_time_left(logger, runner.t_env)
        runner.logger.log_stat("train_sec_per_update_mean", np.mean(time_costs), runner.t_env)

    if args.async_sample_scheme == 'v1':
        _default_v1(post_sample_batch)


def ray_stop(args, ray_ws):
    if args.use_ray:
        ray.shutdown()


def update_info(args, episode, runner):
    if args.sample_scheme.startswith('normal'):
        episode += int(args.batch_size * (args.mp_episode_truncate_len / args.episode_limit))
        runner.t_env += args.batch_size * args.mp_episode_truncate_len
    elif args.sample_scheme.startswith('async'):
        runner.t_env += args.buffer_size * args.mp_episode_truncate_len
        episode += (args.buffer_size * args.mp_episode_truncate_len) // args.episode_limit
    else:
        episode += args.batch_size_run
    return episode


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


def post_update_args(args, env_info):
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.env_info = env_info
    args.env_config = env_info["env_config"]
    args.episode_limit = env_info["episode_limit"]
    if args.env.startswith("gymma"):
        args.n_foods = env_info["n_foods"]
    return args


def create_scheme(args, env_info):
    # Default/Base scheme
    scheme = {
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    if args.env_args.individual_rewards:
        scheme["reward"] = {"vshape": (args.n_agents,)}
    else:
        scheme["reward"] = {"vshape": (1,)}
    
    if args.env.startswith("gymma"):
        scheme.update({
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "avail_actions": {
                "vshape": (env_info["n_actions"],),
                "group": "agents",
                "dtype": th.int,
            },
            "agent_food_dist": {"vshape": (args.n_foods,)},
        })
        scheme["reward_type"] = {"vshape": (1,)}   # 0, 1, 2, ...
    elif args.env.startswith("meltingpot"):
        spec = env_info["world_rgb_spec"]
        if args.store_global_states:
            # WORLD.RGB can take very large RAM, do not save it when using independent MARL
            scheme['WORLD.RGB'] = {"vshape": spec.shape if len(spec.shape) != 0 else (1,),
                                    "dtype": getattr(th, spec.dtype.name)}
        
        for name, spec in env_info["obs_spec"].spaces.items():
            dtype = spec.dtype.name
            if spec.dtype.name.endswith('64'):
                dtype = spec.dtype.name[:-2]+'32'
            scheme[name] = {"vshape": spec.shape if len(spec.shape) != 0 else (1,),
                            "group": "agents",
                            "dtype": getattr(th, dtype)}
    else:
        raise ValueError(f"Unknown env: {args.env}")
    
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    return scheme, groups, preprocess
    

def sample_batch(buffer, args, batch_size, truncate=True, same_device=False, buffer_queue=None, logger=None):
    st = time.time()
    if args.use_ray:
        if args.use_buffer_queue:
            episode_sample = buffer_queue.get()
        else:
            episode_sample = ray.get(buffer.sample.remote(batch_size))
    else:
        episode_sample = buffer.sample(batch_size)
    logger.console_logger.info(f"Get episode_sample: {episode_sample}, time cost: {(time.time() - st):.2f}")
    episode_sample = post_sample_batch(args, episode_sample, truncate, same_device)
    return episode_sample


def post_sample_batch(args, episode_sample, truncate=True, same_device=False):
    # Truncate batch to only filled timesteps
    if truncate:
        max_ep_t = episode_sample.max_t_filled()
        episode_sample = episode_sample[:, :max_ep_t]

    if same_device and episode_sample.device != args.device:
        episode_sample.to(args.device)
    return episode_sample


def serial_buffer_update(args, runner, queue, buffer, buffer_queue=None, logger=None):
    """
    NOTE: for debugging purpose!
    Without ray actors, update the buffer by getting the data from the queue
    """
    if not args.use_ray:
        runner.run(test_mode=False)
        st = time.time()
        count = queue.qsize()
        pbar = tqdm.tqdm(count)
        while queue.qsize() > 0:
            buffer.insert_episode_batch(queue.get())
            pbar.update(1)
        logger.console_logger.info(f"Serially got {count} sub-episodes, buffer size: {get_buffer_size(buffer)}, time cost: {(time.time() - st):.2f}")
        if buffer_queue and args.debug:
            data = buffer.sample(get_buffer_size(buffer))
            buffer_queue.put(data)
            st = time.time()
            rev_data = buffer_queue.get()
            logger.console_logger.info(f"Got data: {rev_data}, time cost: {(time.time() - st):.2f}")


def get_buffer_size(buffer):
    try:
        return ray.get(buffer.size.remote())
    except:
        return buffer.size()


def can_sample(args, env_info, buffer, batch_size=64):
    if args.use_ray:
        return ray.get(buffer.can_sample.remote(batch_size))
    else:
        return buffer.can_sample(batch_size)


def save_to_buffer(args, buffer, episodes):
    if args.use_ray:
        for episode in episodes:
            ray.get(buffer.insert_episode_batch.remote(episode))
    else:
        for episode in episodes:
            buffer.insert_episode_batch(episode)


def buffer_scheme(args, buffer):
    if args.use_ray:
        return ray.get(buffer.get_scheme.remote())
    else:
        return buffer.get_scheme()


def create_buffer(args, scheme, groups, env_info, preprocess, logger, ctx):
    if not args.sample_scheme.startswith('normal'):
        args.buffer_size = args.batch_size_run * (env_info["episode_limit"]//args.mp_episode_truncate_len)
        logger.console_logger.info(f"Buffer size: {args.buffer_size}")
    
    queue = Queue(maxsize=args.buffer_size, actor_options={"name": f"QueueActor", "num_cpus": 2, "max_concurrency": 100})
    buffer_queue = None
    ray_ws = None
    if args.use_buffer_queue:
        buffer_queue = Queue(maxsize=10, actor_options={"name": f"BufferQueueActor", "num_cpus": 1, "max_concurrency": 40})
    if args.use_ray:
        st = time.time()
        buffer = ray.remote(ReplayBufferwithQueue).options(name=f"Buffer",
                                                           num_cpus=2,
                                                           max_concurrency=40,
                                                           num_gpus=0,
                                                           scheduling_strategy=
                                                               NodeAffinitySchedulingStrategy(
                                                                   node_id=ctx.address_info["node_id"], soft=False
                                                               )
                                                           ).remote(
            scheme=scheme,
            groups=groups,
            buffer_size=args.buffer_size,
            max_seq_length=min(env_info["episode_limit"], args.mp_episode_truncate_len) + 1,
            preprocess=preprocess,
            device="cpu" if args.buffer_cpu_only else args.device,
            queue=queue,
            buffer_queue=buffer_queue,
            args=args,
        )
        assert ray.get(buffer.ready.remote())
        ray_ws = [buffer.run.remote()]
        logger.console_logger.info(f"The Buffer is ready, time cost: {(time.time()-st):.2f}")
    else:
        buffer = ReplayBuffer(
            scheme=scheme,
            groups=groups,
            buffer_size=args.buffer_size,
            max_seq_length=min(env_info["episode_limit"], args.mp_episode_truncate_len) + 1,
            preprocess=preprocess,
            device="cpu" if args.buffer_cpu_only else args.device,
            args=args,
        )
    return buffer, queue, buffer_queue, ray_ws
