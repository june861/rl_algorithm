#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ppo.py
@Time    :   2024/09/10 09:49:47
@Author  :   junewluo 
@Email   :   overtheriver861@gmail.com
@description   :   实现PG算法框架
'''

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    def __init__(self,input_dim,out_dim,hidden_dims) -> None:
        super(PolicyNet, self).__init__()
        
        self.layers = []
        layer_input_dims = [input_dim] + hidden_dims
        layer_output_dims = hidden_dims + [out_dim]
        
        for i_dim, o_dim in zip(layer_input_dims,layer_output_dims):
            layer = nn.Linear(i_dim,o_dim)
            self.layers.append(layer)
        # 列表转成Seq对象
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        prob = F.softmax(x)

        return prob


class PG(object):
    def __init__(self,input_dim,out_dim,hidden_dims,layer_num,device,gama=0.95,lr=1e-5) -> None:
        # 记录维度数据
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.layer_num = layer_num
        
        # 训练参数
        self.lr = lr
        self.gama = gama
        self.device =device

        # 训练过程中产生的数据保存
        self.ep_obs, self.ep_act, self.ep_reward = [], [], []
        
        if len(self.hidden_dims) != self.layer_num -1:
            raise RuntimeError(f'hidden_dims size must be {self.layer_num-1}, but now recieve {len(self.hidden_dims)}')

        self.policy_net = PolicyNet(input_dim = input_dim, out_dim = out_dim, hidden_dims = hidden_dims).to(device)
        self.optimizer = optim.Adam(params = self.policy_net.parameters(), lr = self.lr)

        self.loss = []
        
    
    def set_optimizer(self,optim):
        """ 修改优化器 """
        self.optimizer = optim
    
    def set_learning_rate(self,lr):
        """ 修改学习率 """
        self.lr = lr
    
    def select_action(self, state):
        """ 选择动作 """

        #  选择动作，这个动作不是根据Q值来选择，而是使用softmax生成的概率来选
        state = torch.from_numpy(state).to(self.device).float().unsqueeze(0)
        with torch.no_grad():
            prob = self.policy_net(state)
        m = Categorical(prob.cpu())
        action = m.sample()
        return action.item()

    def store_transition(self,obs,action,reward):
        self.ep_obs.append(obs)
        self.ep_act.append(action)
        self.ep_reward.append(reward)

    def cal_reward(self):
        """ 计算reward """
        pass

    def learn(self):
        """ 训练过程 """
        self.policy_net = self.policy_net.to(self.device)
        discounted_reward = np.zeros_like(self.ep_reward)
        adding_rewards = 0

        for i in range(len(self.ep_reward)-1,-1,-1):
            adding_rewards = self.gama * adding_rewards + self.ep_reward[i]
            discounted_reward[i] = adding_rewards
        
        # 对discounted_reward标准化
        mean_reward = np.mean(discounted_reward)
        st_reward = np.std(discounted_reward)
        discounted_reward = (discounted_reward - mean_reward) / st_reward
        discounted_reward = torch.tensor(discounted_reward).to(self.device)

        # 前向传播
        tensor_obs = torch.tensor(self.ep_obs).to(self.device)
        target = torch.tensor(self.ep_act).to(self.device)
        probs = self.policy_net(tensor_obs)
        neg_log_prob = F.cross_entropy(input = probs, target = target, reduction = 'none')

        loss = torch.mean(neg_log_prob * discounted_reward)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss.append(loss.cpu().detach().numpy().item())
        self.ep_obs, self.ep_act, self.ep_reward = [], [], []