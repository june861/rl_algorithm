# -*- encoding: utf-8 -*-
'''
@File    :   ddpg.py
@Time    :   2024/09/13 14:24:05
@Author  :   junewluo 
@description   :   实现DDPG算法框架
'''

import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import BatchSampler, SubsetRandomSampler


class Actor(nn.Module):
    def __init__(self, obs_dim, hidden_dims, out_dim, use_tanh):
        super(Actor, self).__init__()
        if not isinstance(hidden_dims, list):
            raise TypeError(f'parameter require type of list, but recieve {type(hidden_dims)}')
        
        in_features = [obs_dim] + hidden_dims
        out_features = hidden_dims + [out_dim]

        self.layers = []
        for index, (in_dim, out_dim) in enumerate(zip(in_features, out_features)):
            self.layers.append(nn.Linear(in_dim, out_dim, bias = True))
            if  index < len(in_features) - 1:
                activate_func = nn.Tanh() if use_tanh else nn.ReLU()
                self.layers.append(activate_func)
        
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, obs):
        net_out = self.layers(obs)
        return net_out


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dims, use_tanh):
        super(Critic, self).__init__()
        if not isinstance(hidden_dims, list):
            raise TypeError(f'parameter require type of list, but recieve {type(hidden_dims)}')
        
        in_features = [obs_dim] + hidden_dims
        out_features = hidden_dims + [1]

        self.layers = []
        for index, (in_dim, out_dim) in enumerate(zip(in_features, out_features)):
            self.layers.append(nn.Linear(in_dim, out_dim, bias = True))
            if  index < len(in_features) - 1:
                activate_func = nn.Tanh() if use_tanh else nn.ReLU()
                self.layers.append(activate_func)
        
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim = 1)
        net_out = self.layers(obs)
        return net_out


class ReplayBuffer(object):
    def __init__(self, capacity = 1e5) -> None:
        self.capacity = capacity
        self.buffer = []
    
    def add(self, obs, action, reward, obs_, done):
        data = (obs, action, reward, obs_, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer.pop(0)
            self.buffer.append(data)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        obs, action, reward, obs_, dones = map(np.array, zip(*batch))
        return obs, action, reward, obs_, dones

    def clear(self):
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


# 定义噪声模型
class OUNoise:
    def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state.item()


class DDPG(object):

    def __init__(self, state_dim, act_dim, act_bound, args) -> None:
        # define network's relevant  parameters
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.act_bound = act_bound

        # define training parameter
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.gamma = args.gamma
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.sigma = args.sigma
        self.tau = args.tau
        self.device = args.device
        self.noise_scale = args.noise_scale


        # define online network
        self.actor = Actor(obs_dim = self.state_dim, hidden_dims = args.hidden_dims, out_dim = self.act_dim, use_tanh = args.use_tanh).to(self.device)
        self.critic = Critic(obs_dim = self.state_dim, hidden_dims = args.hidden_dims, use_tanh = args.use_tanh).to(self.device)
        # define target network
        self.target_actor = Actor(obs_dim = self.state_dim, hidden_dims = args.hidden_dims, out_dim = self.act_dim, use_tanh = args.use_tanh).to(self.device)
        self.target_critic = Critic(obs_dim = self.state_dim, hidden_dims = args.   hidden_dims, use_tanh = args.use_tanh).to(self.device)

        # soft update target network
        self.update_target_networks(tau = 1.0)

        # define noise model
        self.noise = OUNoise(action_dim = self.act_dim)
        self.noise.reset()

        # define optimizer
        self.actor_optimizer = optim.Adam(params = self.actor.parameters(), lr = self.lr_a)
        self.critic_optimizer = optim.Adam(params = self.critic.parameters(), lr = self.lr_c)

    def update_target_networks(self, tau = None):
        if tau is None:
            tau = self.tau
        with torch.no_grad():
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
 

    @torch.no_grad()
    def select_action(self, obs, eval_mode = True):
        obs = torch.Tensor(obs).to(self.device)
        net_out = self.actor(obs)
        # use tanh func to shinrk output to (-1,1)
        output = F.tanh(net_out) * self.act_bound
        action = output + self.noise_scale * self.noise.sample()

        return action.cpu().numpy()
    
    def learn(self, replay_buffer):
        """ learn network """

        obs, actions, rewards, obs_, dones = replay_buffer.sample(self.batch_size)

        obs = torch.Tensor(obs).to(self.device)
        actions = torch.Tensor(actions).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        obs_ = torch.Tensor(obs_).to(self.device)
        dones = torch.Tensor(dones).to(self.device)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        steps = 0
        
        for index in BatchSampler(SubsetRandomSampler(range(min(len(replay_buffer), self.batch_size))), self.batch_size, False):
            steps += 1
            batch_obs, batch_reward, batch_next_obs, batch_dones = obs[index], rewards[index], obs_[index], dones[index]

            # target network
            act = F.tanh(self.target_actor(batch_next_obs))
            value = self.critic(batch_obs, act)
            value_ = self.target_critic(batch_next_obs, act)
            y = batch_reward + self.gamma * value_ * (1 - batch_dones)

            # update critic network
            self.critic_optimizer.zero_grad()
            critic_loss = F.mse_loss(y, value, reduction = "mean")
            critic_loss.backward()
            self.critic_optimizer.step()

            # update actor loss
            self.actor_optimizer.zero_grad()
            actor_q_values = F.tanh(self.actor(batch_obs))
            score = self.critic(batch_obs ,actor_q_values)
            actor_loss = - torch.mean(score)
            actor_loss.backward()
            self.actor_optimizer.step()

            total_actor_loss += actor_loss.cpu().detach().numpy().item()
            total_critic_loss += critic_loss.cpu().detach().numpy().item()
        
        return total_actor_loss / steps, total_critic_loss / steps
        
 