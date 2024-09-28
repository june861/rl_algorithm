# -*- encoding: utf-8 -*-
'''
@File    :   relaybuffer.py
@Time    :   2024/09/12 15:40:40
@Author  :   junewluo
@description : 缓冲区存储之前采样的数据
'''

import random
import torch
import numpy as np
from copy import deepcopy

class RelayBuffer(object):
    """ 经验放回缓冲区 """
    def __init__(self,buffer_capacity = 10000) -> None:
        self.buffer_capacity = buffer_capacity
        self.buffer = []

    def add(self, state, action, reward, next_state, a_logprob, done, must_add = True):
        # action , a_logprob = np.array([action]), np.array([a_logprob])
        # reward, dw, done = np.array([reward]), np.array([dw]), np.array([done])
        data = (state, action, reward, next_state, a_logprob, done)
        if len(self.buffer) < self.buffer_capacity:
            self.buffer.append(data)
        # 如果缓冲区溢出
        else:
            self.buffer.pop(0)
            self.buffer.append(data)


    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        state, action, reward, next_state, a_logprob, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, a_logprob, done

    def clear(self):
        """ 清除缓冲区 """
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)
        