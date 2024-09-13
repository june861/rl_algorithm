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
# 10. PPO clip parameter annealing
# 11. DropOut
# But not all tricks are implemented in this file. such as 9 will be ensemble in model(Actor, Critic).

import torch.nn as nn
from torch.nn.utils import clip_grad_norm_


def adv_norm():
    pass


def state_norm():
    pass

def reward_norm():
    pass

def policy_entropy():
    pass

def reward_normalization():
    pass

def reward_scaling():
    pass

def lr_decay(optimizer, cur_lr, cur_step, max_step):
    """ use linear decay to update lr

    Args:
        optimizer (torch.optim.Adam): the optimizer of model
        cur_lr (int) : the current learning rate
        cur_step (int): the current step
        max_step (int): the maximun step
    """
    lr_now = cur_lr * (1 - cur_step / max_step)
    for p in optimizer.param_groups:
        p['lr'] = lr_now
    return lr_now

def grad_clip(model,grad_clip_param):
    """     model gradient clip. The principle of gradient 
        clipping is to limit the parameters of the model 
        within a reasonable range. 
            There is no code written in this function, and the 
        technique of gradient clipping will be integrated into 
        the PPO class. Code is avaliable:

        # if self.ppo_params['use_grad_clip']:
        #     torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        # if self.ppo_params['use_grad_clip']:  # Trick 7: Gradient clip
        #     torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

    Args:
        model (torch.nn.Module): model parameter
        grad_clip_param (float): the bound/upper param to clip gradient.
    """
    # clip_grad_norm_(model.parameters(), max_norm = grad_clip_param)
    return None


def orthogonal_initialization(layer, gain = 1.0):
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

def use_tanh_activate(x):
    """ use tanh function as layer activate function. But this function will not be written any code.
        this trick will be ensembled into Class.Actor & Class.Critic in ppo.py Code is avaliable:
        # if use_tanh:
        #     self.actor.append(nn.Tanh())
        # else:
        #     self.actor.append(nn.ReLU())  

    Args:
        x (torch.tensor): input
    """
    return None

def ppo_clip_param_annealing():
    pass

def drop_out():
    """ dropout trick

    Returns:
        _None_: None
    """
    return None