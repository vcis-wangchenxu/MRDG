from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY

import torch
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from modules.agents.rnn_cnn_agent import _get_conv_out_size


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        # input_shape = self._get_input_shape(scheme)
        self._build_agents()
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        self.avail_actions_set = {}
        self.init_hidden_states = {}

    def _create_available_actions(self, n_agents, size):
        avail_actions = th.ones((size, n_agents, self.args.n_actions), device=self.agent.device)
        return avail_actions

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, n_agents=-1, use_rpm=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = self._create_available_actions(n_agents=n_agents, size=ep_batch.batch_size)
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode, n_agents=n_agents)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions, t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, n_agents=-1):
        agent_inputs = self._build_inputs(ep_batch, t)
        if self.args.use_jit:  # run jit NOTE: discard
            bs = agent_inputs['RGB'][:, t].shape[0]
            n_agent = agent_inputs['RGB'][:, t].shape[1]
            shapes = agent_inputs['RGB'][:, t].shape[2:]
            img_dim = shapes[0]
            n_c = shapes[-1]
            img = agent_inputs['RGB'][:, t].contiguous().view(bs*n_agent, n_c, img_dim, img_dim)
            # prepare the non_rgb data
            non_rgb = []
            for key in self.agent.non_rgb_keys:
                non_rgb.append(agent_inputs[key][:, t].contiguous().view(bs*n_agent, -1))
            non_rgb_inputs = th.cat(non_rgb, dim=-1)
            rnn_size = self.agent.agent_model_config['rnn_size']
            agent_outs, self.hidden_states = self.agent(img, non_rgb_inputs, self.hidden_states, rnn_size, self.args.use_rnn)
        else:
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, t)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            # avail_actions = ep_batch["avail_actions"][:, t]
            avail_actions = th.ones((ep_batch.batch_size, n_agents, self.args.n_actions), device=self.agent.get_device())
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.view(ep_batch.batch_size * n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, n_agents, -1)

    def init_hidden(self, batch_size, n_agents=-1, train=False):
        if n_agents == -1:
            n_agents = self.n_agents
        if self.args.agent.endswith('_ns'):
            # NOTE: NO PARAMETER SHARING
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, -1, -1)  # bav
        else:
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self, device):
        self.agent.cuda(device)

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self):
        self.agent = agent_REGISTRY[self.args.agent](self.args)

    def _build_inputs(self, batch, t):
        return batch

    def _get_input_shape(self, scheme):
        # TODO: will delete this function soon
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape


class BasicMACOPRE(BasicMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.opre_weights_q = OPREWeight(args, is_q=True)  # The $q$ is shared across all agents
        self.opre_weights_p = OPREWeight(args, is_q=False)  # Each has a $p$
    
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, n_agents=-1, use_rpm=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = self._create_available_actions(n_agents=n_agents, size=ep_batch.batch_size)
        agent_outputs, _, _, _ = self.forward(ep_batch, t_ep, test_mode=test_mode, n_agents=n_agents)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions, t_env, test_mode=test_mode)
        return chosen_actions
    
    def forward(self, ep_batch, t, test_mode=False, n_agents=-1):
        agent_inputs = self._build_inputs(ep_batch, t)
        
        z_prob_test = None  # in case the test_mode
        if test_mode:
            z, z_prob = self.opre_weights_p(agent_inputs, self.hidden_states, t)
        else:
            # get both z_prob_test and z_prob for KL loss
            z_test, z_prob_test = self.opre_weights_p(agent_inputs, self.hidden_states.detach(), t)
            z, z_prob = self.opre_weights_q(agent_inputs, self.hidden_states.detach(), t)
        # note that, do not backpropagate the gradients
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, t, z=z.detach(), z_prob=z_prob.detach())

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            # avail_actions = ep_batch["avail_actions"][:, t]
            avail_actions = th.ones((ep_batch.batch_size, n_agents, self.args.n_actions), device=self.agent.get_device())
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.view(ep_batch.batch_size * n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, n_agents, -1), z, z_prob, z_prob_test

    def parameters(self):
        return list(self.agent.parameters()) + \
                    list(self.opre_weights_p.parameters())

    def cuda(self, device):
        self.agent.cuda(device)
        self.opre_weights_q.cuda(device)
        self.opre_weights_p.cuda(device)


class OPREWeight(nn.Module):
    def __init__(self, args, is_q=True):
        super().__init__()
        self.args = args
        self.is_q = is_q
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
        # add n options
        if self.is_q:
            non_rgb_dims = self.non_rgb_shapes * self.args.n_agents
            mlp_dims = self.agent_model_config.mlp_size * self.args.n_agents
            rnn_dims = self.agent_model_config.rnn_size * self.args.n_agents
        else:
            non_rgb_dims = self.non_rgb_shapes
            mlp_dims = self.agent_model_config.mlp_size
            rnn_dims = self.agent_model_config.rnn_size
        self.fc3 = nn.Linear(non_rgb_dims+mlp_dims+rnn_dims, self.args.opre_n_bins)
            
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

    def forward(self, inputs, hidden_state, t, ag_index=-1, n_agents=-1):
        bs = inputs['RGB'][:, t].shape[0]
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
        x = F.relu(self.conv1(img.float()/255.0))
        x = F.relu(self.conv2(x))
        
        # get the feature of CONVS
        if self.is_q:
            new_bs = bs
        else:
            new_bs = bs * n_agents
        x = F.relu(self.fc1(x.view(bs * n_agents, -1)))
        x = F.relu(self.fc2(x))
        
        if ag_index == -1:  # select the corresponding RNN hidden states
            h_in = hidden_state.view(-1, self.agent_model_config.rnn_size)
        else:
            h_in = hidden_state.view(bs, -1, self.agent_model_config.rnn_size)[:, ag_index]
        # prepare the non_rgb data
        non_rgb_inputs = [
            _process(inputs[key], t, ag_index).contiguous().view(new_bs, -1) \
                for key in self.non_rgb_keys]

        x = torch.cat([x.reshape(new_bs, -1)] + non_rgb_inputs + [h_in.reshape(new_bs, -1)], dim=1).float()
        z = self.fc3(x)  # [new_bs, n_bins]
        if self.is_q:  # copy z for n_agents
            z = z[:, None, :].repeat(1, n_agents, 1).view(bs*n_agents, -1)
        z_prob = F.softmax(z, dim=-1)
        z = z.view(bs, n_agents, -1)
        z_prob = z_prob.view(bs, n_agents, -1)
        return z, z_prob
