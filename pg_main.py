#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   pg_main.py
@Time    :   2024/09/13 09:21:04
@Author  :   junewluo 
@Email   :   overtheriver861@gmail.com
@description   :   test PG algorithm 
'''

import torch
import os
import argparse
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from pg.pg import PG



## TEST CODE ##
def main(args,pid,seed):
    # env_name = 'CartPole-v1'
    # ENV_NAME = 'Pendulum-v1'
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(args.env_name,  render_mode = args.render_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    max_iter = args.max_iter
    max_step = args.max_step
    writer = SummaryWriter(log_dir='runs/PG_{}_number_seed_{}'.format(args.env_name,pid,seed))

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    layer_num = 3
    hidden_dims = [128,128]

    agent = PG(input_dim = state_dim, out_dim = action_dim, hidden_dims = hidden_dims, layer_num = layer_num, device = device)
    train_loss = []
    train_reward = []

    for i in range(args.max_iter):
        state, _ = env.reset()
        total_reward = 0
        for step in range(args.max_step):
            action = agent.select_action(state = state) # softmax概率选择action
            next_state, reward, done, truncation, _ = env.step(action)
            total_reward += reward
            agent.store_transition(state, action, reward)   # 新函数 存取这个transition
            state = next_state
            if done:
                # print("stick for ",step, " steps")
                agent.learn()   # 更新策略网络
                break
        if (i+1) % 500 == 0:
            train_loss.append(sum(agent.loss) / len(agent.loss))
            agent.loss = []
        train_reward.append(total_reward)
        # Test every 100 episodes
        if (i+1) % 5 == 0:
            times = 10
            total_reward = 0
            for k in range(times):
                state, _ = env.reset()
                for j in range(max_step):
                    # env.render()
                    action = agent.select_action(state)  # direct action for test
                    next_state, reward, done, truncation, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / 10
            print('episode: ',i, 'Evaluation Average Reward:', ave_reward)
            writer.add_scalar('eval_rewards_{}'.format(args.env_name), ave_reward, global_step = i)
    # 最终测试
    state, _ = env.reset()
    if args.render_mode == "human":
        env.render()
    done = False
    total_reward += 0
    while not done:
        action = agent.select_action(state = state)
        next_state, reward, done, truncation, _ = env.step(action)
        total_reward += reward
        if done:
            break
    print(f'test stage: reward is {total_reward}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO")
    parser.add_argument('--max_iter', type = int, default = 1000, help = "The maximum number of rounds")
    parser.add_argument('--max_step', type = int, default = 1000, help = "The maximum number of moves per round")
    parser.add_argument('--env_name', type = str, default = 'CartPole-v1', help = "which env used")
    parser.add_argument('--render_mode', type = str, default = 'human', help = "render mode choosen")

    args = parser.parse_args() 
    pid = 10
    seed = 42
    main(args = args, pid = pid, seed = seed)