import copy
import collections

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
        # input_shape = self._get_input_shape(scheme)
        self._build_agents()
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        self.init_hidden_states = {}

        self.last_rpm_update_T = -self.args.rpm_update_interval

        # the rpm memory and the copied policies for RPM agents
        # use the training performance of agents in the substrates 
        self.rpm = RankedPolicyMemory(args=self.args)
        if self.args.agent.endswith('_ns'):
            self.rpm_agents = [copy.deepcopy(self.agent.agents[i]) for i in range(self.n_agents)]
        else:
            self.rpm_agents = [copy.deepcopy(self.agent) for i in range(self.n_agents)]
        self.avail_actions_set = {}

    def _create_available_actions(self, n_agents, size):
        if size not in self.avail_actions_set:
            self.avail_actions_set[size] = th.ones((size, n_agents, self.args.n_actions), device=self.agent.device)
        avail_actions = self.avail_actions_set[size]
        # TODO create a function to generate avail_actions
        return avail_actions

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, n_agents=-1, use_rpm=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = self._create_available_actions(n_agents=n_agents, size=ep_batch.batch_size)
        if use_rpm:
            agent_outputs = self.forward_rpm(ep_batch, t_ep, test_mode=test_mode, n_agents=n_agents)
        else:
            agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode, n_agents=n_agents)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions, t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, n_agents=-1):
        agent_inputs = ep_batch
        avail_actions = self._create_available_actions(n_agents=n_agents, size=ep_batch.batch_size)
        agent_outs, self.hidden_states = self.agent(inputs=agent_inputs,
                                                    hidden_state=self.hidden_states,
                                                    t=t,
                                                    n_agents=self.n_agents if n_agents == -1 else n_agents)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.view(ep_batch.batch_size * n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, n_agents, -1)

    def forward_rpm(self, ep_batch, t, test_mode=False, n_agents=-1):
        assert not test_mode, "RPM is only called while training"
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = self._create_available_actions(n_agents=n_agents, size=ep_batch.batch_size)
        agent_outs_list, rnn_hiden_list = [], []
        # self.hidden_states: [bs, n_agents, hidden_size]
        for agent_id, agent in enumerate(self.rpm_agents):  # loop over each agent's policy
            _agent_outs, _hidden_state = agent(inputs=agent_inputs[agent_id],
                                               hidden_state=self.hidden_states[:, agent_id][:, None],
                                               t=t,
                                               n_agents=self.n_agents if n_agents == -1 else n_agents)
            agent_outs_list.append(_agent_outs)
            rnn_hiden_list.append(_hidden_state)
        
        agent_outs = th.cat(agent_outs_list, dim=0)  # [bsxn_agents, n_actions]
        self.hidden_states = th.stack(rnn_hiden_list, dim=1)   # [bs, n_agents, hidden_size]

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":  # TODO consider PPO and other methods

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.view(ep_batch.batch_size * n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, n_agents, -1)

    def init_hidden(self, batch_size, n_agents=-1, train=False):
        if n_agents == -1:
            n_agents = self.n_agents
        if train:
            if batch_size not in self.init_hidden_states:
                if self.args.agent.endswith('_ns'):
                    hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, -1, -1)  # bav
                else:
                    hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, n_agents, -1)  # bav
                self.init_hidden_states[batch_size] = hidden_states
            self.hidden_states = self.init_hidden_states[batch_size]
        else:
            if self.args.agent.endswith('_ns'):
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
        inputs = collections.OrderedDict({agent_id: collections.OrderedDict() for agent_id in range(self.n_agents)})
        for agent_id in range(self.n_agents):
            for obs_name, _ in self.agent.obs_specs.items():
                if self.args.marl_independent and obs_name == 'WORLD.RGB':
                    continue
                shapes = batch[obs_name].shape
                inputs[agent_id][obs_name] = \
                    batch[obs_name][:, :, agent_id, ...].view((shapes[0], shapes[1], 1, *shapes[3:]))
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def sample_rpm(self, t: int=-1):
        """Sample RPM policies from the RPM
        """
        if self.rpm.memory_key_size >= self.args.n_agents:
            self.rpm_agents = self.rpm.load_rpm_models(self.rpm_agents, t)
    
    def update_rpm(self, t_env: int, reward: float, update=False):
        """update RPM by the runner

        Args:
            t_env (int): the global time step of the real environment
            reward (float): the evaluated score of the model
        """
        self.rpm.update(
            t_env=t_env,
            checkpoint={
            "reward": reward,
            "state_dict": self.agent.state_dict()
        })
