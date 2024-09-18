# -*- encoding: utf-8 -*-
'''
@File    :   ppo.py
@Time    :   2024/09/12 09:11:33
@Author  :   junewluo 
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from copy import deepcopy
from ppo.trick import orthogonal_initialization
from torch.distributions import Categorical

""" PPO Algorithm 
PPO算法是一种基于Actor-Critic的架构， 它的代码组成和A2C架构很类似。

1、 传统的A2C结构如下：
                                    ⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅ backward ⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅  
                                    ⬇                                                    ⬆
    State & (R, new_State) ----> Critic ----> value -- 与真正的价值reality_value做差 --> td_e = reality_v - v
                                                                                                    ⬇
    State ----> Actor ---Policy--> P(s,a) ➡➡➡➡➡➡➡➡➡➡➡ 两者相加 ⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅
                                    ⬇                             ⬇
                                    ⬇                             ⬇
                                    ⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅ actor_loss = Log[P(s,a)] + td_e

    ** Actor Section **：由Critic网络得到的输出td_e，和Actor网络本身输出的P(s,a)做Log后相加得到了Actor部分的损失，使用该损失进行反向传播
    ** Critic Section **: Critic部分，网络接收当前状态State，以及env.step(action)返回的奖励(Reward，简写为R)和新的状态new_State
                                    
2、 实际上PPO算法的Actor-Critic框架实现：
                                    ⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅ backward ⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅  
                                    ⬇                                                    ⬆
    State & (r, new_State) ➡➡➡ Critic ----> value -- 与真正的价值reality_value做差 --> td_e = reality_v - v 
                                                                                                           ⬇
    State ➡➡➡ Actor[old_policy] ➡➡➡ P_old(s,a) ➡➡➡➡➡➡➡➡➡➡ ratio = P_new(s,a) / P_old(s,a) ➡➡ 依据式子(1)计算loss ➡➡➡➡ loss
      ⬇                                                              ⬆                                                                   ⬇ 
      ⬇                                                              ⬆[两者做商,得到重要性权值]                                            ⬇             
      ⬇                                                              ⬆                                                                   ⬇ 
      ⬇ ➡➡➡➡ Actor[new_policy] ➡➡➡ P_new(s,a) ➡➡➡➡➡➡➡➡➡ ⬆                                                                   ⬇ 
                        ⬆                                                                                                                ⬇
                        ⬆                                                                                                                ⬇
                        ⬆⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅ backbard ⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬅⬇
同时将会实现PPO的一些技巧:

"""




class Actor(nn.Module):
    def __init__(self,state_dim:int, act_dim:int, hidden_dims:list, use_tanh = False) -> None:
        super(Actor,self).__init__()
        # 网络结构初始化
        input_dims = [state_dim] + hidden_dims
        output_dims = hidden_dims + [act_dim]
        self.actor = []
        for index, (in_d, out_d) in enumerate(zip(input_dims,output_dims)):
            layer = nn.Linear(in_d, out_d)
            if index != len(input_dims) - 1:
                orthogonal_initialization(layer = layer)
            else:
                orthogonal_initialization(layer = layer, gain = 0.1)
            self.actor.append(layer)
            # self.actor.append(nn.Linear(in_d, out_d))
            if index != len(input_dims) - 1:
                if use_tanh:
                    self.actor.append(nn.Tanh())
                else:
                    self.actor.append(nn.ReLU())
        self.actor = nn.Sequential(*self.actor)
    
    def forward(self,state):
        """ actor网络输出动作的概率 """
        out = self.actor(state)
        a_prob = torch.softmax(out, dim = 1) 
        return a_prob


class Critic(nn.Module):
    def __init__(self,state_dim:int, act_dim:int, hidden_dims:list, use_tanh = False) -> None:
        super(Critic,self).__init__()
        act_dim = 1
        input_dims = [state_dim] + hidden_dims
        output_dims = hidden_dims + [act_dim]
        self.critic = []
        for index,(in_d, out_d) in enumerate(zip(input_dims,output_dims)):
            layer = nn.Linear(in_d, out_d)
            if index != len(input_dims) - 1:
                orthogonal_initialization(layer = layer)
            else:
                orthogonal_initialization(layer = layer, gain = 0.1)
            self.critic.append(layer)
            # self.critic.append(nn.Linear(in_d, out_d))
            if index != len(input_dims) - 1:
                if use_tanh:
                    self.critic.append(nn.Tanh())
                else:
                    self.critic.append(nn.ReLU())
        self.critic = nn.Sequential(*self.critic)
    
    def forward(self,state):
        value = self.critic(state)
        return value


class PPO(object):
    def __init__(self,state_dim, act_dim, hidden_dims,layer_nums,train_params) -> None:
        """ PPO Init """
        self.device = train_params['device']
        self.ppo_params = {
            # 网络层参数
            'state_dim' : state_dim,
            'act_dim' : act_dim,
            'layer_nums' : layer_nums,

            # ppo algorithm params
            'clip_param' : train_params['clip_param'],

            # 训练参数
            'lr_a': train_params['lr_a'],
            'lr_c': train_params['lr_c'],
            'gamma': train_params['gamma'],
            'lamda': train_params['lamda'],
            'entropy_coef': train_params['entropy_coef'],
            'batch_size' : train_params['batch_size'],
            'mini_batch_size' : train_params['mini_batch_size'],

            # trick params
            'off_policy' : False, # use off-policy or on-policy
            'use_buffer' : False, # use buffer to store or not
            'use_tanh' : train_params['use_tanh'], # use tanh activate func or ReLU func
            'use_adv_norm' : train_params['use_adv_norm'], # use advantage normalization
            'use_grad_clip' : train_params['use_grad_clip'], # use grad clip in model params.
            'grad_clip_params': train_params['grad_clip_params']
            
        }

        if not isinstance(hidden_dims,list) :
            raise RuntimeError(f"hidden_dims type must be list, now receive {type(hidden_dims)}. ")
        if len(hidden_dims) != layer_nums - 1:
            raise RuntimeError(f"hidden_dims'len expect {layer_nums-1}, but now receive {len(hidden_dims)}. ")
        
        self.actor = Actor(state_dim = state_dim, act_dim = act_dim, hidden_dims = hidden_dims, use_tanh = self.ppo_params['use_tanh']).to(self.device)
        self.critic = Critic(state_dim = state_dim, act_dim = act_dim, hidden_dims = hidden_dims, use_tanh = self.ppo_params['use_tanh']).to(self.device)
        # use off-policy
        if self.ppo_params['off_policy']:
            self.actor_old = deepcopy(self.actor).to(self.device)
        # define optimizer
        self.actor_optim = optim.AdamW(params = self.actor.parameters(), lr = self.ppo_params['lr_a'])
        self.critic_optim = optim.AdamW(params  = self.critic.parameters(), lr = self.ppo_params['lr_c'])

    def select_action(self, state, eval_mode = False):
        """ 选择动作 """
        state = torch.Tensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            if self.ppo_params['off_policy'] and eval_mode == False:
                a_probs = self.actor_old(state).cpu()
            else:
                a_probs = self.actor(state).cpu()
            dist = Categorical(probs = a_probs)
            a = dist.sample()
            a_logprobs = dist.log_prob(a)
        return a.numpy().item(), a_logprobs.numpy().item()

    
    def update_old_net(self):
        """ 更新actor_old参数 """
        if self.ppo_params['off_policy']:
            self.actor_old.load_state_dict(self.actor.state_dict())
    
    def cal_adv(self,states, next_states, rewards, dw, dones):
        # calculate adavantage functions

        adv = []
        gae = 0
        with torch.no_grad():
            value = self.critic(states)
            value_ = self.critic(next_states)
            deltas = rewards + self.ppo_params['gamma'] * (1.0 - dones) * value_ - value
            v_target = rewards + self.ppo_params['gamma'] * (1.0 - dones) * value_
            for delta, done in zip(reversed(deltas.cpu().flatten().numpy()), reversed(dones.cpu().flatten().numpy())):
                # gae = delta + self.ppo_params['gamma'] * self.ppo_params['lamda'] * gae * (1.0 - done)
                gae = delta + self.ppo_params['gamma'] * self.ppo_params['lamda'] * gae
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).to(self.device).view(-1, 1)
            # v_target = rewards + self.ppo_params['gamma'] * (1.0 - dones) * value_
            if self.ppo_params['use_adv_norm']:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-8))

        return adv, v_target

    def learn(self, replay_buffer, batch_size):
        """ 训练参数 """

        if len(replay_buffer) < batch_size:
            batch_size = len(replay_buffer)

        # 从经验缓冲区中取出数据
        states, actions, rewards, next_states, a_logprob, dw, dones = replay_buffer.sample()
        states = torch.Tensor(states).to(self.device)
        actions = torch.Tensor(actions).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)
        a_logprobs = torch.Tensor(a_logprob).to(self.device)
        batch_dw = torch.Tensor(dw).to(self.device)
        dones = torch.Tensor(dones).to(self.device)

        actor_total_loss, critic_total_loss = 0.0, 0.0
        step = 0

        for index in BatchSampler(SubsetRandomSampler(range(self.ppo_params['batch_size'])), self.ppo_params['mini_batch_size'], True):
            state, next_state, reward, dw, done = states[index], next_states[index], rewards[index], batch_dw[index], dones[index]
            # state = (state - state.mean()) / (state.std() + 1e-8)
            action, a_logprob = actions[index], a_logprobs[index]                                                                                                    
            # calculate adv
            adv, v_target = self.cal_adv(states = state, next_states = next_state, rewards = reward, dw = dw, dones = done)
            dist_now = Categorical(probs=self.actor(state))
            # shape is (batch_size, 1)
            dist_entropy = dist_now.entropy().view(-1, 1)
            # shape is (batch_size, 1)
            a_logprob_now = dist_now.log_prob(action.squeeze()).view(-1, 1)  
            # a/b=exp(log(a)-log(b)), shape is (batch_size, 1)
            ratios = torch.exp(a_logprob_now - a_logprob)
            surr1 = ratios * adv  
            surr2 = torch.clamp(ratios, 1 - self.ppo_params['clip_param'], 1 + self.ppo_params['clip_param']) * adv
            actor_loss = - torch.min(surr1, surr2) - self.ppo_params['entropy_coef'] * dist_entropy  # shape(mini_batch_size X 1)
            # actor_loss = torch.max(-surr1, -surr2).mean()
            # actor_loss = - torch.min(surr1, surr2)
            # Update actor
            self.actor_optim.zero_grad()
            actor_loss.mean().backward()
            if self.ppo_params['use_grad_clip']:  # Trick 7: Gradient clip
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.ppo_params['grad_clip_params'])
            self.actor_optim.step()

            v_s = self.critic(state)
            # critic_loss = 0.5 * ((v_s - v_target) **2).mean()
            critic_loss = F.mse_loss(v_s, v_target).mean()
            #  critic_loss = torch.clamp(critic_loss, min = - 10.0, max = 10.0)
            # Update critic
            self.critic_optim.zero_grad()
            critic_loss.backward()
            if self.ppo_params['use_grad_clip']:  # Trick 7: Gradient clip
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.ppo_params['grad_clip_params'])
            self.critic_optim.step()

            actor_total_loss += actor_loss.mean().detach().cpu().numpy().item()
            critic_total_loss += critic_loss.detach().cpu().numpy().item()

            step += 1
        
        return actor_total_loss / step, critic_total_loss / step
        
    


        