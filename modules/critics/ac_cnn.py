import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.agents.rnn_cnn_agent import _get_conv_out_size


class ACCritic(nn.Module):
    def __init__(self, scheme, args):
        super(ACCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.scheme = scheme

        self.output_type = "v"
        self.env_config = args.env_config
        self.obs_specs = self.env_config.timestep_spec.observation
        self.non_rgb_shapes = self._cal_non_rgb_obs_shapes(self.obs_specs)

        self.agent_model_config = self.args.agent_model
        self.env_config = self.args.env_config

        # Set up network layers
        if self.args.use_global_state_in_critic:  # create global state cnn
            self._init_global_cnn()
        self._init_cnn()
        
        # Set up network layers
    def _init_cnn(self):
        if self.args.marl_independent:
            img_shape = self.env_config.timestep_spec.observation['RGB'].shape
        else:
            img_shape = self.env_config.timestep_spec.observation['WORLD.RGB'].shape
        H, W = img_shape[:-1]
        C = 1 if self.args.env_args.gray_scale else 3
        insize = (-1, C, H, W)
        self.conv1 = torch.nn.Conv2d(C,
                                     self.agent_model_config.cnn_layer_1_out_channel, 
                                     self.agent_model_config.cnn_layer_1_kernel,
                                     stride=self.agent_model_config.cnn_layer_1_stride)
        h_out = _get_conv_out_size(in_size=img_shape[0],
                                   kernel_size=self.agent_model_config.cnn_layer_1_kernel, 
                                   stride=self.agent_model_config.cnn_layer_1_stride,
                                   padding=0)
        w_out = _get_conv_out_size(in_size=img_shape[1],
                                   kernel_size=self.agent_model_config.cnn_layer_1_kernel, 
                                   stride=self.agent_model_config.cnn_layer_1_stride,
                                   padding=0)
        assert h_out > 0 and w_out > 0
        out_size = (-1, self.agent_model_config.cnn_layer_1_out_channel, h_out, w_out)
        self.conv2 = torch.nn.Conv2d(self.agent_model_config.cnn_layer_1_out_channel,
                                     self.agent_model_config.cnn_layer_2_out_channel, 
                                     self.agent_model_config.cnn_layer_2_kernel,
                                     stride=self.agent_model_config.cnn_layer_2_stride)
        h_out = _get_conv_out_size(in_size=h_out,
                                   kernel_size=self.agent_model_config.cnn_layer_2_kernel, 
                                   stride=self.agent_model_config.cnn_layer_2_stride,
                                   padding=0)
        w_out = _get_conv_out_size(in_size=w_out,
                                   kernel_size=self.agent_model_config.cnn_layer_2_kernel, 
                                   stride=self.agent_model_config.cnn_layer_2_stride,
                                   padding=0)
        assert h_out > 0 and w_out > 0
        insize = (self.agent_model_config.cnn_layer_2_out_channel, h_out, w_out)
        
        self.state_dim = np.prod(insize)
        
        self.state_dim += self.n_agents  # agent's id
        if self.args.obs_last_action:
            self.state_dim += self.scheme["actions_onehot"]["vshape"][0]
        if self.args.marl_independent:
            self.state_dim += self.non_rgb_shapes
        if self.args.use_global_state_in_critic:
            self.state_dim += self._global_state_cnn_out_dim
        self.fc1 = nn.Linear(self.state_dim, self.agent_model_config.mlp_size)
        self.fc2 = nn.Linear(self.agent_model_config.mlp_size, self.agent_model_config.mlp_size)
        self.fc3 = nn.Linear(self.agent_model_config.mlp_size, 1)  # since there are n agents

    def _init_global_cnn(self):
        img_shape = self.env_config.timestep_spec.observation['WORLD.RGB'].shape
        H, W = img_shape[:-1]
        C = 1 if self.args.env_args.gray_scale else 3
        insize = (-1, C, H, W)
        self._s_conv1 = torch.nn.Conv2d(C,
                                     self.agent_model_config.cnn_layer_1_out_channel, 
                                     self.agent_model_config.cnn_layer_1_kernel,
                                     stride=self.agent_model_config.cnn_layer_1_stride)
        h_out = _get_conv_out_size(in_size=img_shape[0],
                                   kernel_size=self.agent_model_config.cnn_layer_1_kernel, 
                                   stride=self.agent_model_config.cnn_layer_1_stride,
                                   padding=0)
        w_out = _get_conv_out_size(in_size=img_shape[1],
                                   kernel_size=self.agent_model_config.cnn_layer_1_kernel, 
                                   stride=self.agent_model_config.cnn_layer_1_stride,
                                   padding=0)
        assert h_out > 0 and w_out > 0
        out_size = (-1, self.agent_model_config.cnn_layer_1_out_channel, h_out, w_out)
        self._s_conv2 = torch.nn.Conv2d(self.agent_model_config.cnn_layer_1_out_channel,
                                     self.agent_model_config.cnn_layer_2_out_channel, 
                                     self.agent_model_config.cnn_layer_2_kernel,
                                     stride=self.agent_model_config.cnn_layer_2_stride)
        h_out = _get_conv_out_size(in_size=h_out,
                                   kernel_size=self.agent_model_config.cnn_layer_2_kernel, 
                                   stride=self.agent_model_config.cnn_layer_2_stride,
                                   padding=0)
        w_out = _get_conv_out_size(in_size=w_out,
                                   kernel_size=self.agent_model_config.cnn_layer_2_kernel, 
                                   stride=self.agent_model_config.cnn_layer_2_stride,
                                   padding=0)
        assert h_out > 0 and w_out > 0
        insize = (self.agent_model_config.cnn_layer_2_out_channel, h_out, w_out)
        self._global_state_cnn_out_dim = np.prod(insize)

    def _cal_non_rgb_obs_shapes(self, obs_specs):
        non_rgb_shapes = 0
        self.non_rgb_keys = []
        for obs_name, obs_spec in obs_specs.items():
            if "RGB" not in obs_name:
                self.non_rgb_keys.append(obs_name)
                if len(obs_spec.shape) == 0:
                    non_rgb_shapes += 1
                else:
                    non_rgb_shapes += np.prod(obs_spec.shape)
        return non_rgb_shapes

    def forward(self, batch, t=None):
        # independent learner uses GRB (local inputs) as inputs
        rbg = batch['RGB']  # [bs, ts, n_agents, h, w, c]
        # rbg = batch['WORLD.RGB']  # [bs, ts, h, w, c]
        bs = rbg.shape[0]
        ts = rbg.shape[1]
        if self.args.marl_independent:
            n_agent = rbg.shape[2]
            shapes = rbg.shape[3:]
            n_c = shapes[-1]
            img = rbg.view(bs*ts*n_agent, n_c, shapes[0], shapes[1])
        else:
            n_agent = self.n_agents
            shapes = rbg.shape[2:]
            n_c = shapes[-1]
            img = rbg.view(bs*ts*n_agent, n_c, shapes[0], shapes[1])
        x = F.relu(self.conv1(img.float()/255.0))
        x = F.relu(self.conv2(x))
        
        _agent_ids = th.eye(self.n_agents, device=batch.device)[None, None, ...].repeat(bs, ts, 1, 1)
        d = [x.reshape(bs*ts*n_agent, -1), _agent_ids.reshape(bs*ts*n_agent, -1)]
        if self.args.obs_last_action:
            d.append(batch['actions_onehot'].view(bs*ts*n_agent, -1))
        if self.args.marl_independent:
            non_rgb_inputs = [
                batch[key].view(bs*ts*n_agent, -1) \
                    for key in self.non_rgb_keys]
            d.extend(non_rgb_inputs)
        
        if self.args.use_global_state_in_critic:
            _s_rbg = batch['WORLD.RGB']  # [bs, ts, h, w, c]
            _shapes = batch['WORLD.RGB'].shape[2:]
            _s_img = _s_rbg.view(bs*ts, _shapes[-1], _shapes[0], _shapes[1])
            _s_x = F.relu(self._s_conv1(_s_img.float()/255.0))
            _s_x = F.relu(self._s_conv2(_s_x)).view(bs*ts, -1)
            _s_x = _s_x.repeat(n_agent, 1)
            d.append(_s_x)
        
        x = th.cat(d, dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        q = self.fc3(x)
        q = q.view(bs, ts, n_agent, 1)
        return q
