import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter


def _get_conv_out_size(in_size, kernel_size, stride, padding):
    return (in_size - kernel_size + 2 * padding) // stride + 1


class RNNCNNAgent(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.env_config = args.env_config
        self.obs_specs = self.env_config.timestep_spec.observation

        # get non-rgb observations shapes
        self.non_rgb_shapes = self._cal_non_rgb_obs_shapes(self.obs_specs)
        self.agent_model_config = args.agent_model
        self._init_model()

    def get_device(self):
        return self.fc1.weight.device
    
    @property
    def device(self):
        return self.fc1.weight.device

    def _init_model(self):
        img_shape = self.env_config.timestep_spec.observation['RGB'].shape
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
        self.fc1 = nn.Linear(np.prod(insize),
                             self.agent_model_config.mlp_size)
        self.fc2 = nn.Linear(self.agent_model_config.mlp_size, self.agent_model_config.mlp_size)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(self.non_rgb_shapes+self.agent_model_config.mlp_size, self.agent_model_config.rnn_size)
        else:
            self.rnn = nn.Linear(self.non_rgb_shapes+self.agent_model_config.mlp_size, self.agent_model_config.rnn_size)
        self.fc3 = nn.Linear(self.agent_model_config.rnn_size, self.args.n_actions)

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

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(
            (1, self.agent_model_config.rnn_size), device=self.fc1.weight.device)

    def forward(self, inputs, hidden_state, t, ag_index=-1, n_agents=-1):
        # inputs['RGB']: [bs, T+1, n_agents, H, W, C]
        bs = inputs['RGB'][:, t].shape[0]
        
        # if bs > 1: assert bs == self.args.batch_size
    
        # process each agent's data
        def _process(data, t, ag_index):
            if ag_index == -1:
                return data[:, t]
            else:
                return data[:, t, ag_index][:, None, ...]
        
        n_agents = inputs['RGB'][:, t].shape[1] if ag_index == -1 else 1
        shapes = inputs['RGB'][:, t].shape[2:]  # get H, W, C
        img_h_dim, img_c_dim, n_c = shapes[0], shapes[1], shapes[-1]
        img = _process(inputs['RGB'], t, ag_index).contiguous().view(bs*n_agents, n_c, img_h_dim, img_c_dim)
        # prepare the non_rgb data
        non_rgb_inputs = [
            _process(inputs[key], t, ag_index).contiguous().view(bs*n_agents, -1) \
                for key in self.non_rgb_keys]
        
        x = F.relu(self.conv1(img.float()/255.0))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x.view(bs*n_agents, -1)))
        x = F.relu(self.fc2(x))
        
        if ag_index == -1:  # select the corresponding RNN hidden states
            h_in = hidden_state.view(-1, self.agent_model_config.rnn_size)
        else:
            h_in = hidden_state.view(bs, -1, self.agent_model_config.rnn_size)[:, ag_index]
        x = torch.cat([x.view(bs*n_agents, -1)] + non_rgb_inputs, dim=1).float()
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc3(h)
        return q, h
