#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   trick.py
@Time    :   2024/09/13 11:27:40
@Author  :   junewluo 
@Email   :   overtheriver861@gmail.com
@description   :   some trick for ppo algorithm
'''

# In this .py file, I will implenment some useful trick for ppo.
# 1. Advantage Normalization
# 2. State Normalization
# 3. Reward Normalization & Reward Scaling
# 4. Policy Entropy
# 5. Learning Rate Decay
# 6. Model Gradient clip
# 7. Orthogonal Initialization
# 8. Adam Optimizer Epsilon Parameter
# 9. Tanh Activation Function
# But not all tricks are implemented in this file. such as 9 will be ensemble in model(Actor, Critic).

def adv_norm():
    pass


def state_norm():
    pass

def reward_norm():
    pass

def policy_entropy():
    pass

def lr_decay(optimizer, cur_setp, max_step):
    """use linear decay to update lr

    Args:
        optimizer (torch.optim.Adam): 
        cur_setp (): _description_
        max_step (_type_): _description_
    """
