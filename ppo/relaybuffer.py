# -*- encoding: utf-8 -*-
'''
@File    :   relaybuffer.py
@Time    :   2024/09/12 15:40:40
@Author  :   junewluo
@description : 缓冲区存储之前采样的数据
'''

import random
import numpy as np

class RelayBuffer(object):
    """ 经验放回缓冲区 """
    def __init__(self,buffer_capacity = 10e6) -> None:
        self.buffer_capacity = buffer_capacity
        self.buffer = []

    def add(self, state, action, reward, next_state, a_logprob, dw, done,must_add = True):
        data = (state, action, reward, next_state, a_logprob, dw, done)
        if len(self.buffer) < self.buffer_capacity:
            self.buffer.append(data)
            return 1
        # 如果缓冲区溢出
        else:
            if must_add == True:
                self.buffer.pop(0)
                self.buffer.append(data)
                return 1
        # 数据写入缓冲区失败
        return 0
        
    def sample(self, mini_batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), mini_batch_size))
        state, action, reward, next_state, a_logprob, dw, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, a_logprob, dw, done
        # data_lengths = len(self.buffer)
        # # 获取每个batch中数据在self.buffer中的下标
        # batch_start = np.arange(0, data_lengths, mini_batch_size)
        # # 生成indices = []
        # indices = np.arange(data_lengths, dtype=np.int64)
        # # 打乱下标
        # np.random.shuffle(indices)
        # mini_batches = [indices[i:i + mini_batch_size] for i in batch_start]

        # return mini_batches 

    def clear(self):
        """ 清除缓冲区 """
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)
        