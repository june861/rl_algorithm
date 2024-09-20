# -*- encoding: utf-8 -*-
'''
@File    :   dqn.py
@Time    :   2024/09/20 16:16:12
@Author  :   junewluo 
'''

import torch
import random
import numpy as np
import torch.nn as nn

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical


class QNet(nn.Module):
    def __init__(self, state_dim, hidden_dims, act_dim) -> None:
        super(QNet,self).__init__()
        # layer structure init. default to use ReLU as activate function
        input_dims = [state_dim] + hidden_dims
        output_dims = hidden_dims + [act_dim]
        self.q_net = []
        for index, (in_d, out_d) in enumerate(zip(input_dims,output_dims)):
            if index != len(input_dims) - 1:
                self.q_net.append(nn.Linear(in_d, out_d))
                self.q_net.append(nn.ReLU())
        self.q_net = nn.Sequential(*self.actor)

    def forward(self, x):
        net_out = self.q_net(x)
        return net_out

class RelayBuffer(object):
    def __init__(self,capacity) -> None:
        self.capacity = capacity
        self.buffer = []
    
    def add(self, obs, action, reward, next_obs):
        """ add frame data to buffer.

        Args:
            obs (_numpy.ndarray_): the observation of env.
            action (_numpy.ndarray_): actions.
            reward (_numpy.ndarray_): reward from env.step().
            next_obs (_numpy.ndarray_): the new observation after taking action in state.

        Returns:
            _int_: return 1 while add success.
        """
        data = (obs, action, reward, next_obs)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
            return 1
        return 0


    def sample(self, mini_batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), mini_batch_size))
        obs, action, reward, next_obs = map(np.array, zip(*batch))
        return obs, action, reward, next_obs


    def clear(self):
        """ clear buffer space. """
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)

class DQN(object):
    def __init__(self, state_dim, hidden_dims, layers, act_dim, args) -> None:
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.layers = layers
        if not isinstance(hidden_dims, list) or (len(hidden_dims) != self.layers -1):
            raise RuntimeError(r'')
        # init two network
        self.q_net = QNet(state_dim = self.state_dim, hidden_dims = hidden_dims, act_dim = self.act_dim)
        self.target_net = QNet(state_dim = self.state_dim, hidden_dims = hidden_dims, act_dim = self.act_dim)
        # make target network parameters equal to q network
        self.target_net.load_state_dict(self.q_net.state_dict())
        # build some parameters for dqn training.
        self.dqn_params = {
            'epsilon': args.epsilon,
            'gamma': args.gamma,
            'mini_batch_size' : args.mini_batch_size,
            'lr': args.learning_rate,
            'device': args.device,
        }


    def select_action(self, obs):
        """ selection action from random or q_net.

        Args:
            obs (_numpy.ndarray_): the current state of env.

        Returns:
            _int_: the action of agent taking.
        """
        if random.random() < self.dqn_params['epsilon']:
            return np.random.uniform(-1, 1, 3) 
        obs = torch.Tensor(obs)
        with torch.no_grad():
            net_out = self.q_net(obs)
            logprob = nn.Softmax(net_out)
            dist = Categorical(logprob)
            action = dist.sample()
        
        return action.numpy().item()

    def learn(self, relay_buffer: RelayBuffer):

        # sample all data from buffer
        obs, actions, rewards, next_obs = relay_buffer.sample(mini_batch_size = self.dqn_params['mini_batch_size'])

        obs = torch.Tensor(obs).to(self.dqn_params['device'])
        actions = torch.Tensor(actions).to(self.dqn_params['device'])
        rewards = torch.Tensor(rewards).to(self.dqn_params['device'])
        next_obs = torch.Tensor(next_obs).to(self.dqn_params['device'])

        