#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   replay_buffer.py
@Time    :   2024/10/23 11:40:14
@Author  :   junewluo 
@Email   :   overtheriver861@gmail.com
@description   :   xxxxxxxxx
'''

import torch
import numpy as np
import random
from torch.utils.data import RandomSampler, BatchSampler

def check(input):
    output = torch.Tensor(input) if type(input) == np.ndarray else input
    return output

class ReplayBuffer(object):
    def __init__(self, args):
        self._capacity = args.capacity
        self._obs_dim = args.obs_dim
        self._action_dim = args.act_dim
        self._steps_per_bacth = args.max_step_per_batch
        self._num_env = args.n_rollout_threads

        self._obs = np.zeros(shape=(self._steps_per_bacth, self._num_env, self._obs_dim))
        self._reward = np.zeros(shape=(self._steps_per_bacth, self._num_env, 1))
        self._action = np.zeros(shape=(self._steps_per_bacth, self._num_env, self._action_dim))
        self._a_logprobs = np.zeros(shape=(self._steps_per_bacth, self._num_env, 1))
        self._obs_ = np.zeros(shape=(self._steps_per_bacth, self._num_env, self._obs_dim))
        self._dones = np.zeros(shape=(self._steps_per_bacth, self._num_env, 1))

        self.index = -1
    

    def add(self, obs, reward, action, a_logprobs, obs_, dones):
        self.index += 1
        self._obs[self.index % self._steps_per_bacth] = obs
        self._reward[self.index % self._steps_per_bacth] = reward.reshape((reward.shape[0],1))
        self._action[self.index % self._steps_per_bacth] = action
        self._a_logprobs[self.index % self._steps_per_bacth] = a_logprobs
        self._obs_[self.index % self._steps_per_bacth] = obs_
        self._dones[self.index % self._steps_per_bacth] = dones.reshape((dones.shape[0],1))

        


    def sample(self, sample_n):
        indice = random.sample(range(sample_n), min(self.__len__(), sample_n))

        sample_obs = self._obs[indice]
        sample_r = self._reward[indice]
        sample_act = self._action[indice]
        sample_a_logprobs = self._a_logprobs[indice]
        sample_obs_ = self._obs_[indice]
        sample_done = self._dones[indice]

        return sample_obs, sample_r, sample_act, sample_a_logprobs, sample_obs_, sample_done
    

    def rollout(self, mini_batch_size, device):
        if self.index < 0:
            raise RuntimeError(f'buffer empty')
        obs = self._obs.reshape((self._obs.shape[0] * self._obs.shape[1],-1))
        actions = self._action.reshape((self._action.shape[0] * self._action.shape[1], -1))
        rewards = self._reward.reshape((self._reward.shape[0] * self._reward.shape[1], -1))
        obs_ = self._obs_.reshape((self._obs_.shape[0] * self._obs_.shape[1], -1))
        a_logprobs = self._a_logprobs.reshape((self._a_logprobs.shape[0] * self._a_logprobs.shape[1], -1))
        dones = self._dones.reshape((self._dones.shape[0] * self._dones.shape[1], -1))

        for indice in BatchSampler(RandomSampler(range(obs.shape[0])), mini_batch_size, False):
            _obs, action, reward, _obs_, a_logprob, done \
                = obs[indice], actions[indice], rewards[indice], obs_[indice], a_logprobs[indice], dones[indice]
            _obs, action, reward, _obs_, a_logprob, done = check(_obs).to(device), check(action).to(device), \
                        check(reward).to(device), check(_obs_).to(device), check(a_logprob).to(device), \
                        check(done).to(device)
            yield _obs, action, reward, _obs_, a_logprob, done
    

    def __len__(self):
        if self.index < self._capacity:
            return self.index + 1
        return self._capacity