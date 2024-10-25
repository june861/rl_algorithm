#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   replay_buffer.py
@Time    :   2024/10/23 11:40:14
@Author  :   junewluo 
@Email   :   overtheriver861@gmail.com
@description   :   xxxxxxxxx
'''

import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, args):
        self._capacity = args.capacity
        self._obs_dim = args.obs_dim
        self._action_dim = args.act_dim
        self._steps_per_bacth = args.steps_per_bacth
        self._num_env = args.env_num

        self._obs = np.zeros(shape=(self._steps_per_bacth, self._num_env, self._obs_dim))
        self._reward = np.zeros(shape=(self._steps_per_bacth, self._num_env, 1))
        self._action = np.zeros(shape=(self._steps_per_bacth, self._num_env, self._action_dim))
        self._a_logprobs = np.zeros(shape=(self._steps_per_bacth, self._num_env, 1))
        self._obs_ = np.zeros(shape=(self._steps_per_bacth, self._num_env, self._obs_dim))
        self._dones = np.zeros(shape=(self._steps_per_bacth, self._num_env, 1))

        self.index = -1
    

    def add(self, obs, reward, action, a_logprobs, obs_, dones):
        self.index += 1
        self._obs[self.index % self._capacity] = obs
        self._reward[self.index % self._capacity] = reward
        self._action[self.index % self._capacity] = action
        self._a_logprobs[self.index % self._capacity] = a_logprobs
        self._obs_[self.index % self._capacity] = obs_
        self._dones[self.index % self._capacity] = dones

        


    def sample(self, sample_n):
        indice = random.sample(range(sample_n), min(self.__len__(), sample_n))

        sample_obs = self._obs[indice]
        sample_r = self._reward[indice]
        sample_act = self._action[indice]
        sample_a_logprobs = self._a_logprobs[indice]
        sample_obs_ = self._obs_[indice]
        sample_done = self._dones[indice]

        return sample_obs, sample_r, sample_act, sample_a_logprobs, sample_obs_, sample_done
    

    def rollout(self):
        if self.index < 0:
            raise RuntimeError(f'buffer empty')
        return self._obs.reshape()

    

    def __len__(self):
        if self.index < self._capacity:
            return self.index + 1
        return self._capacity