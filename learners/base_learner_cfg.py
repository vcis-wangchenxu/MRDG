import numpy as np
import torch as th


class BaseLearner(object):
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.scheme = scheme
        self.logger = logger

        self.env_info = args.env_info
        if self.args.standardise_state_obs and self.args.env.startswith("gymma"):
            self.observation_space = self.env_info['observation_space']
            self.state_space       = self.env_info['state_space']
            self.obs_max = th.from_numpy(np.array(self.observation_space[0].high)).float().to(self.args.device)
            self.state_max = th.from_numpy(np.array(self.state_space.high)).float().to(self.args.device)

    def _normalise_obs(self, batch):
        if self.args.standardise_state_obs and self.args.env.startswith("gymma"):
            bs, ts, n_agents = batch["state"].shape[0], batch["state"].shape[1], self.args.n_agents
            batch["state"] = batch["state"] / self.state_max[None, None, :].repeat(bs, ts, 1)
            batch["obs"] = batch["obs"] / self.obs_max[None, None, None, :].repeat(bs, ts, n_agents, 1)
        return batch
