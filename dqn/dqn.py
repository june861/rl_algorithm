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
import torch.optim as optim
import torch.nn.functional as F

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
            self.q_net.append(nn.Linear(in_d, out_d))
            if index != len(input_dims) - 1:
                self.q_net.append(nn.ReLU())
            
        self.q_net = nn.Sequential(*self.q_net)

    def forward(self, x):
        net_out = self.q_net(x)
        return net_out

class RelayBuffer(object):
    def __init__(self,capacity) -> None:
        self.capacity = capacity
        self.buffer = []
    
    def add(self, obs, action, reward, next_obs, done):
        """ add frame data to buffer.

        Args:
            obs (_numpy.ndarray_): the observation of env.
            action (_numpy.ndarray_): actions.
            reward (_numpy.ndarray_): reward from env.step().
            next_obs (_numpy.ndarray_): the new observation after taking action in state.

        Returns:
            _int_: return 1 while add success.
        """
        data = (obs, action, reward, next_obs, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
            return 1
        return 0


    def sample(self, mini_batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), mini_batch_size))
        obs, action, reward, next_obs, done = map(np.array, zip(*batch))
        return obs, action, reward, next_obs, done


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

        # init two network
        self.q_net = QNet(state_dim = self.state_dim, hidden_dims = hidden_dims, act_dim = self.act_dim).to(args.device)
        self.target_net = QNet(state_dim = self.state_dim, hidden_dims = hidden_dims, act_dim = self.act_dim).to(args.device)
        # make target network parameters equal to q network
        self.target_net.load_state_dict(self.q_net.state_dict())
        # build some parameters for dqn training.
        self.dqn_params = {
            'epsilon': args.epsilon,
            'epsilon_min': args.epsilon_min,
            'epsilon_decay': args.epsilon_decay,
            'gamma': args.gamma,
            'mini_batch_size' : args.mini_batch_size,
            'lr': args.lr,
            'device': args.device,
            'update_target' : args.update_target,

        }
        self.optimizer = optim.Adam(params = self.q_net.parameters(), lr = self.dqn_params['lr'])
        # record update number
        self.update_count = 0



    def select_action(self, obs, eval_mode = False):
        """ selection action from random or q_net.

        Args:
            obs (_numpy.ndarray_): the current state of env.

        Returns:
            _int_: the action of agent taking.
        """
        obs = torch.Tensor(obs).to(self.dqn_params['device'])
        if np.random.uniform() <= 1 - self.dqn_params['epsilon'] or eval_mode:
            action_value = self.q_net(obs)
            if len(action_value.shape) == 1:
                action_value = action_value.unsqueeze(0)
            if self.dqn_params['device'].type != 'cpu':
                action_value = action_value.cpu()
            action = torch.max(action_value, dim = 1)[1].data.numpy()
            # print(f"model select is {action}")
        else:
            dim = obs.shape[0] if len(obs.shape) > 1 else 1
            action = np.random.randint(0, self.act_dim, size=(dim,))
            # print(f"random select is {action}")
        return action

    def update_epsilon(self):
        self.dqn_params['epsilon'] = max(self.dqn_params['epsilon_min'], self.dqn_params['epsilon'] - self.dqn_params['epsilon_decay'])

    def set_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def learn(self, relay_buffer: RelayBuffer):
        
       
        # sample all data from buffer
        obs, actions, rewards, next_obs, dones = relay_buffer.sample(mini_batch_size = len(relay_buffer))
        # convert to tensor
        obs = torch.Tensor(obs).to(self.dqn_params['device'])
        actions = torch.LongTensor(actions).to(self.dqn_params['device'])
        rewards = torch.Tensor(rewards).to(self.dqn_params['device']).view(-1,1)
        next_obs = torch.Tensor(next_obs).to(self.dqn_params['device'])
        dones = torch.Tensor(dones).to(self.dqn_params['device']).view(-1,1)
        
        for index in BatchSampler(SubsetRandomSampler(range(len(relay_buffer))), self.dqn_params['mini_batch_size'], False):
            mini_obs, mini_actions, mini_rewards, mini_next_obs, mini_dones = obs[index], actions[index], rewards[index], next_obs[index], dones[index]
            
            # calculate Q-Value
            Q = self.q_net(mini_obs).gather(1, mini_actions.unsqueeze(1))
            Q_ = self.target_net(mini_next_obs).max(1)[0].view(-1, 1)
            Q_ = mini_rewards + self.dqn_params['gamma'] * Q_ * (1 - mini_dones)

            # epsilon decay
            # self.dqn_params['epsilon'] = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # cal loss & parameter step
            self.optimizer.zero_grad()
            q_loss = F.mse_loss(Q, Q_)
            q_loss.mean().backward()
            self.optimizer.step()

        # update target network
        self.update_count += 1
        if self.update_count % self.dqn_params['update_target']:
            self.set_target_network()

        return q_loss.cpu().detach().numpy().item()
