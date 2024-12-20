import copy

from statistics import mode
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

from training_methods.ranked_policy_memory.ranked_policy_memory import RankedPolicyMemory


# This multi-agent controller shares parameters between agents
class BasicRPMMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

        self.last_rpm_update_T = -self.args.rpm_update_interval

        # the rpm memory and the copied policies for RPM agents
        self.rpm = RankedPolicyMemory(args=self.args)
        self.rpm_agents = [copy.deepcopy(self.agent) for _ in range(self.n_agents)]

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, use_rpm=False, n_agents=-1):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        if use_rpm:
            agent_outputs = self.forward_rpm(ep_batch, t_ep, test_mode=test_mode)
        else:
            agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def rpm_select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        """Select actions for the RPM agents
        """
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward_rpm(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward_rpm(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t).view(ep_batch.batch_size, self.n_agents, -1)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs_list, rnn_hiden_list = [], []
        # self.hidden_states: [bs, n_agents, hidden_size]
        for i, agent in enumerate(self.rpm_agents):  # loop over each agent's policy
            _agent_outs, _hidden_state = agent(agent_inputs[:, i], self.hidden_states[:, i][:, None])
            agent_outs_list.append(_agent_outs)
            rnn_hiden_list.append(_hidden_state)
        
        agent_outs = th.cat(agent_outs_list, dim=0)  # [bsxn_agents, n_actions]
        self.hidden_states = th.stack(rnn_hiden_list, dim=1)   # [bs, n_agents, hidden_size]

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size, n_agents=-1):
        if self.args.agent.endswith('_ns'):
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, -1, -1)  # bav
        else:
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self, device=None):
        if device is not None:
            self.agent.cuda(device)
        else:
            self.agent.cuda()
    
    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def sample_rpm(self, t):
        """Sample RPM policies from the RPM
        """
        if self.rpm.memory_key_size >= self.args.n_agents:
            self.rpm_agents = self.rpm.load_rpm_models(self.rpm_agents, t)

    def update_rpm(self, t_env: int, reward: float):
        """update RPM by the runner

        Args:
            t_env (int): the global time step of the real environment
            reward (float): the evaluated score of the model
        """
        if self.args.rpm_update_interval > 0 and \
            (t_env - self.last_rpm_update_T) / self.args.rpm_update_interval >= 1.0:
            
            self.rpm.update(
                t_env=t_env,
                checkpoint={
                "reward": reward,
                "state_dict": self.agent.state_dict()
            })
            self.last_rpm_update_T = t_env
