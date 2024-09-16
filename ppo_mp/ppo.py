# -*- encoding: utf-8 -*-
'''
@File    :   ppo.py
@Time    :   2024/09/12 09:11:33
@Author  :   junewluo 
'''
import os
import numpy as np
import torch
import datetime
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from time import time

from copy import deepcopy
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
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
            self.actor.append(nn.Linear(in_d, out_d))
            if index != len(input_dims) - 1:
                if use_tanh:
                    self.actor.append(nn.Tanh())
                else:
                    self.actor.append(nn.ReLU())
        self.actor = nn.Sequential(*self.actor)
    
    def forward(self,state):
        """ actor网络输出动作的概率 """
        out = self.actor(state)
        a_prob = torch.softmax(out, dim = 0) 
        return a_prob


class Critic(nn.Module):
    def __init__(self,state_dim:int, act_dim:int, hidden_dims:list, use_tanh = False) -> None:
        super(Critic,self).__init__()
        act_dim = 1
        input_dims = [state_dim] + hidden_dims
        output_dims = hidden_dims + [act_dim]
        self.critic = []
        for index,(in_d, out_d) in enumerate(zip(input_dims,output_dims)):
            self.critic.append(nn.Linear(in_d, out_d))
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
            'hidden_dims': hidden_dims,

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
            'off_policy' : train_params['off_policy'], # use off-policy or on-policy
            'use_buffer' : train_params['use_buffer'], # use buffer to store or not
            "use_adv_norm" : train_params['use_adv_norm'], # use advantage normalization
            "use_state_norm" : train_params['use_state_norm'], # use state normalization
            "use_reward_norm" : train_params['use_reward_norm'], # use reward normalization
            'use_tanh' : train_params['use_tanh'], # use tanh activate func or ReLU func
            'use_adv_norm' : train_params['use_adv_norm'], # use advantage normalization
            'use_grad_clip' : train_params['use_grad_clip'], # use grad clip in model params.
            'grad_clip_params': train_params['grad_clip_params'],
            'use_ppo_clip': train_params['use_ppo_clip'], # use ppo clip param method.
            'use_lr_decay': train_params['use_lr_decay'],
            "use_gae": train_params['use_gae'],
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
        self.actor_optim = optim.Adam(params = self.actor.parameters(), lr = self.ppo_params['lr_a'])
        self.critic_optim = optim.Adam(params  = self.critic.parameters(), lr = self.ppo_params['lr_c'])

    def select_action(self, state, eval_mode = False, renturn_entropy = False):
        """ select action

        Args:
            state (_numpy.ndarray_): the env state.
            eval_mode (bool, optional): whether is eval stage. Defaults to False.

        Returns:
            _numpy.ndarray_ : return action.
            _numpy.ndarray_ : return action prob.
        """
        state = torch.Tensor(state).to(self.device)
        with torch.no_grad():
            if self.ppo_params['off_policy'] and eval_mode == False:
                a_probs = self.actor_old(state).cpu()
            else:
                a_probs = self.actor(state).cpu()
            dist = Categorical(probs = a_probs)
            a = dist.sample()
            a_logprobs = dist.log_prob(a)
        if renturn_entropy == False:
            return a.numpy(), a_logprobs.numpy()
        else:
            return a, a_logprobs, dist.entropy()

    def get_value(self, state):
        with torch.no_grad():
            value = self.critic(state.to(self.device))
        return value.cpu().numpy()
    
    def update_old_net(self):
        """ 更新actor_old参数 """
        if self.ppo_params['off_policy']:
            self.actor_old.load_state_dict(self.actor.state_dict())
    
    def cal_adv(self,obs, next_obs, rewards, dones, values, next_done):
        # bootstrap value if not done
        batch_per_step = obs.shape[0]
        with torch.no_grad():
            next_value = self.get_value(next_obs).reshape(1, -1)
            if self.ppo_params['use_gae']:
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                # obs.shape[0] = args.batch_per_steps
                for t in reversed(range(batch_per_step)):
                    if t == batch_per_step - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = torch.tensor(next_value).to(self.device)
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = torch.tensor(values[t + 1]).to(self.device)
                    delta = rewards[t] + self.ppo_params['gamma'] * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + self.ppo_params['gamma'] * self.ppo_params['lamda'] * nextnonterminal * lastgaelam
                returns = advantages + values
                return advantages, returns
            else:
                returns = torch.zeros_like(rewards).to(self.device)
                for t in reversed(range(batch_per_step)):
                    if t == batch_per_step - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + self.ppo_params['gamma'] * nextnonterminal * next_return
                advantages = returns - values
                return advantages, returns


    def learn(self, batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_dones, batch_values, obs, dones, update_epoch):
        
        # calculate advantage function.
        adv, returns = self.cal_adv(obs = batch_obs, 
                                    next_obs = obs, 
                                    rewards = batch_rewards, 
                                    dones = batch_dones,
                                    values = batch_values,
                                    next_done = dones
                                )

        # flatten the batch
        b_obs = batch_obs.reshape((-1,) + (self.ppo_params['state_dim'],))
        b_logprobs = batch_log_probs.reshape(-1)
        # b_actions = batch_actions.reshape((-1,) + self.ppo_params['act_dim'])
        b_advantages = adv.reshape(-1)
        b_returns = returns.reshape(-1)
        # b_values = batch_values.reshape(-1)

        # Optimizing the policy and value network
        # b_inds = np.arange(self.ppo_params['batch_size'])
        
        loss_record = []
        for epoch in range(update_epoch):
            # np.random.shuffle(b_inds)
            actor_total_loss, critic_total_loss = 0.0, 0.0
            total_steps = 0
            for index in BatchSampler(SubsetRandomSampler(range(b_obs.shape[0])), self.ppo_params['mini_batch_size'], False):
            # for start in range(0, self.ppo_params['batch_size'], self.ppo_params['mini_batch_size']):
                total_steps += 1
                # end = start + self.ppo_params['mini_batch_size']
                
                a_probs = self.actor(b_obs[index])
                dist = Categorical(probs = a_probs)
                a_tmp = dist.sample()
                a_logprobs = dist.log_prob(a_tmp)
                newlogprob = a_logprobs
                entropy = dist.entropy()
                # _, newlogprob, entropy = self.select_action(b_obs[index], renturn_entropy = True)
                newvalue = self.critic(b_obs[index])
                logratio = newlogprob - b_logprobs[index]
                ratio = torch.exp(logratio)

                mb_advantages = b_advantages[index]
                if self.ppo_params['use_adv_norm']:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                surr1 = -mb_advantages * ratio
                surr2 = -mb_advantages * torch.clamp(ratio, 1 - self.ppo_params['use_ppo_clip'], 1 + self.ppo_params['use_ppo_clip'])
                entropy_loss = entropy.mean()
                self.actor_optim.zero_grad()
                actor_loss = torch.max(surr1, surr2).mean() - self.ppo_params['entropy_coef'] * entropy_loss
                actor_loss.backward()
                self.actor_optim.step()

                # Value loss
                newvalue = newvalue.view(-1)
                self.critic_optim.zero_grad()
                critic_loss = 0.5 * ((newvalue - b_returns[index]) ** 2).mean()
                critic_loss.backward()
                self.critic_optim.step()

                actor_total_loss += actor_loss.detach().cpu().numpy().item()
                critic_total_loss += critic_loss.detach().cpu().numpy().item()
            actor_total_loss /= total_steps
            critic_total_loss /= total_steps
            loss_record.append((actor_total_loss, critic_total_loss))
        
        return loss_record
    
    def checkpoint_attributes(self, only_net):
        if only_net:
            attr = {
                'state_dim': self.ppo_params['state_dim'],
                'act_dim': self.ppo_params['act_dim'],
                'layer_nums' : self.ppo_params['layer_nums'],
                'hidden_dims': self.ppo_params['hidden_dims'],
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
            }
        else:
            attr = {
                'ppo_params': self.ppo_params,
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'actor_optim' : self.actor_optim.state_dict(),
                'critic_optim' : self.critic_optim.state_dict(),
            }

    @classmethod
    def from_checkoint(cls, checkpoint):
        agent_instance = cls(
            checkpoint['ppo_params']
        )

        agent_instance.actor.load_state_dict(checkpoint['actor'])
        agent_instance.critic.load_state_dict(checkpoint['critic'])
        agent_instance.actor_optimizer.load_state_dict(checkpoint["actor_optim"])
        agent_instance.critic_optimizer.load_state_dict(checkpoint["critic_optim"])

    def save_checkpoint(self,only_net = False):
        model_save_dir = './models/ppo_mp'
        if not os.path.exists('./models/'):
            os.mkdir('./models/')
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        now_time = datetime.datetime.now().strftime("%Y-%m-%d")
        model_save_path = os.path.join(model_save_dir, f'{now_time}_{os.getpid()}_{str(only_net)}')
        torch.save(self.checkpoint_attributes(only_net = only_net), model_save_path)

    


        