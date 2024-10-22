#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ppo_continous.py
@Time    :   2024/10/22 16:27:48
@Author  :   junewluo 
@Email   :   overtheriver861@gmail.com
@description   :   implement ppo continous algorithms
'''

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.distributions import Beta, Normal  
from torch.utils.data import SubsetRandomSampler, BatchSampler

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output



def orthogonal_init(layer, gain = 1.0):
    """ orthogonal initialization

    Args:
        layer (torch.nn.Linear): _description_
        gain (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    nn.init.orthogonal_(layer.weight, gain = gain)
    nn.init.constant_(layer.bias, 0)
    return None


class Replay_Buffer(object):
    def __init__(self, capacity=1e5):
        self.capacity = capacity
        self.buffer = []
        self.index = 0    

    def add(self, obs, action, reward, obs_, action_log_prob, done):
        data = (obs, action, reward, obs_, action_log_prob, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.index] = data
            self.index = self.index % self.capacity
        
    
    def rollout(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        state, action, reward, next_state, a_logprob, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, a_logprob, done


class ActorBeta(nn.Module):
    def __init__(self, obs_dim, hidden_dims, act_dim, use_tanh, use_orthogonal_init):
        super(ActorBeta, self).__init__()
        input_dims = [obs_dim] + hidden_dims
        output_dims = hidden_dims + [act_dim]
        self.layers = []
        self.activate_funcs = [nn.ReLU(), nn.Tanh()]
        for index,(in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            if index < len(output_dims) - 1:
                self.layers.append(orthogonal_init(nn.Linear(in_dim, out_dim)) if use_orthogonal_init else nn.Linear(in_dim, out_dim))
                self.layers.append(self.activate_funcs[use_tanh])
        self.layers = nn.Sequential(*self.layers)

        # add distribution variable
        self.alpha_layer = nn.Linear(hidden_dims[-1], act_dim)
        self.beta_layer = nn.Linear(hidden_dims[-1], act_dim)

        if use_orthogonal_init:
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)
    
    def forward(self, x):
        x = self.layers(x)
        alpha_ = F.softplus(self.alpha_layer(x)) + 1.0
        beta_ = F.softplus(self.beta_layer(x)) + 1.0

        return alpha_, beta_
    
    def get_dist(self, x):
        alpha, beta = self.forward(x)

        return Beta(alpha, beta)

    def mean(self, x):
        alpha, beta = self.forward(x)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean

class ActorGaussian(nn.Module):
    def __init__(self, obs_dim, hidden_dims, act_dim, max_action, use_tanh, use_orthogonal_init):
        super(ActorGaussian, self).__init__()
        self.max_action = max_action
        self.layers = []

        input_dims = [obs_dim] + [hidden_dims]
        output_dims = [hidden_dims] + [act_dim]

        for index, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            if index < len(output_dims) - 1:
                self.layers.append(orthogonal_init(nn.Linear(in_dim, out_dim)) if use_orthogonal_init else nn.Linear(in_dim, out_dim))
                self.layers.append(self.activate_funcs[use_tanh])


        self.mean_layer = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.layers = nn.Sequential(*self.layers)

        if use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.layers(s)
        mean = self.max_action * torch.tanh(self.mean_layer(s))  # [-1,1]->[-max_action,max_action]
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist

class Critic(nn.Module):
    def __init__(self,state_dim:int, hidden_dims:list, use_tanh = False) -> None:
        super(Critic,self).__init__()

        input_dims = [state_dim] + hidden_dims
        output_dims = hidden_dims + [1]
        self.critic = []
        for index,(in_d, out_d) in enumerate(zip(input_dims,output_dims)):
            layer = nn.Linear(in_d, out_d)
            if index != len(input_dims) - 1:
                orthogonal_init(layer = layer)
            else:
                orthogonal_init(layer = layer, gain = 0.1)
            self.critic.append(layer)
            if index != len(input_dims) - 1:
                self.critic.append([nn.ReLU(), nn.Tanh()][use_tanh])
        self.critic = nn.Sequential(*self.critic)
    
    def forward(self,state):
        value = self.critic(state)
        return value


class PPO_continous(object):
    def __init__(self, args):
        # define env info
        self._obs_dim = args.obs_dim
        self._act_dim = args.act_dim
        # define actor-critic structure info
        self._max_action = args.max_action
        self._policy_dist = args.policy_dist
        self._actor_hidden_dims = args.actor_hidden_dims
        self._critic_hidden_dims = args.critic_hidden_dims
        
        # define init & train info
        self._batch_size = args.bacth_size
        self._mini_batch_size = args.mini_batch_size
        self._lr_a = args.lr_a
        self._lr_c = args.lr_c
        self._use_tanh = args.use_tanh
        self._use_orthogonal_init = args.use_orthogonal_init
        self._ppo_epoch = args.ppo_epoch
        self._lambda = args.lambda_
        self._use_gae = args.use_gae



        # define AC framework
        self.critic = Critic(state_dim = self._obs_dim, hidden_dims = self._critic_hidden_dims, use_tanh = self._use_tanh)
        if self._policy_dist == 'Beta':
            self.actor = ActorBeta(obs_dim = self._obs_dim, 
                                   hidden_dims = self._actor_hidden_dims, 
                                   use_orthogonal_init = self._use_orthogonal_init, 
                                   use_tanh = self._use_tanh)
        elif self._policy_dist == "Gaussion":
            self.actor = ActorGaussian(obs_dim = self._obs_dim,
                                       hidden_dims = self._actor_hidden_dims,
                                       act_dim = self._act_dim,
                                       max_action = self._max_action,
                                       use_tanh = self._use_tanh,
                                       use_orthogonal_init = self._use_orthogonal_init
                                    )
        
        # define optimizer 
        self.actor_optimizer = optim.Adam(params = self.actor.parameters(), lr = self._lr_a)
        self.critic_optimizer = optim.Adam(params = self.critic.parameters(), lr = self._lr_c)
    

    @torch.no_grad()
    def eval_stage(self, obs):
        obs = check(obs)
        if self._policy_dist == "Beta":
            a = self.actor.mean(obs).detach().numpy().flatten()
        else:
            a = self.actor(obs).detach().numpy().flatten()
        return a

    @torch.no_grad()
    def selection_action(self, obs):
        obs = check(obs)
        if self.policy_dist == "Beta":
            with torch.no_grad():
                dist = self.actor.get_dist(obs)
                a = dist.sample()  # Sample the action according to the probability distribution
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        else:
            with torch.no_grad():
                dist = self.actor.get_dist(obs)
                a = dist.sample()  # Sample the action according to the probability distribution
                a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        return a.numpy().flatten(), a_logprob.numpy().flatten()

    @torch.no_grad()
    def cal_adv(self, obs, rewards, obs_, dones):

        value = self.critic(obs)
        value_ = self.critic(obs_)
        deltas = rewards + self._lambda * value_ - value
        # use gae function
        if self._use_gae:
            pass


    def data_generator(self, states, actions, rewards, next_states, a_logprobs, dones):
        for indice in BatchSampler(SubsetRandomSampler(range(states.shape[0])), self.ppo_params['mini_batch_size'], False):
            obs = states[indice]
            action = actions[indice]
            reward = rewards[indice]
            obs_ = next_states[indice]
            a_log_prob = a_logprobs[indice]
            done = dones[indice]

            yield obs, action, reward, obs_, a_log_prob, done

    def update(self, sample,):
        pass
    

    def learn(self, replay_buffer):

        train_info =  {
            'ratio': 0.0,
            'policy_loss': 0.0,
            'value_loss' : 0.0,
            'dist_entropy': 0.0,
        }

        for _ in range(self._ppo_epoch):
            states, actions, rewards, next_states, a_logprobs, dones = replay_buffer.rollout(min(self._batch_size, len(replay_buffer)))
            generators = self.data_generator()

            for sample in generators:
                pass

        

