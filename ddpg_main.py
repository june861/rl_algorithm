# -*- encoding: utf-8 -*-
'''
@File    :   ddpg_main.py
@Time    :   2024/10/06 22:55:19
@Author  :   junewluo 
'''

import argparse
import torch
import os
import time
import wandb
import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter
from ddpg.ddpg import DDPG, ReplayBuffer
from share_func import build_env, clear_folder
from dqn.trick import lr_decay
from share_func import run2gif, write_metric

parser = argparse.ArgumentParser("DDPG Parameter Setting")
 
# env setting
parser.add_argument("--env_name",type=str,default="MountainCarContinuous-v0",help="The Env Name")
parser.add_argument("--env_num",type=int,default=20,help="The number of envs that are activated")
parser.add_argument("--max_eposide_step", type=int, default=500, help="the max step in one eposide game")
parser.add_argument("--seed", type=int, default=1, help="random seed")
# training setting
parser.add_argument("--max_train_steps", type=int, default=200, help="the max train steps")
parser.add_argument("--evaluate_freq", type=int, default=200, help="evaluate frequency")
parser.add_argument("--evaluate_times", type=int, default=1, help="evaluate times in one evaluation eposide")
parser.add_argument("--lr_a", type=float, default=5e-3, help="learning rate of Actor")
parser.add_argument("--lr_c", type=float, default=5e-4, help="learning rate of Critic")
parser.add_argument("--gamma", type=float, default=0.9, help="discounted element")
parser.add_argument("--sigma", type=float, default=0.5, help="")
parser.add_argument("--tau", type=float, default=0.4, help="")
parser.add_argument("--noise_scale", type=float, default=0.1)
parser.add_argument("--use_tanh", type=int, default=1)
parser.add_argument("--batch_size",type=int,default=2048 ,help="mini batch size to sample from buffer")
parser.add_argument("--mini_batch_size",type=int,default=256,help="mini batch size to sample from buffer")
parser.add_argument("--capacity",type=int,default=int(1e5),help="the capacity of buffer to store data")
parser.add_argument("--use_lr_decay", type=int, default=1, help="use learning rate decay")
parser.add_argument("--update_target", type=int, default=100, help="update target network")
# network setting
parser.add_argument("--layers",type=int,default=3,help="the number of layer in q_net")
parser.add_argument("--hidden_dims", type=int, nargs='+', default=[128, 128], help='Sizes of the hidden layers (e.g., --hidden_sizes 50 30)')
# monitor setting
parser.add_argument("--wandb", type=int, default=0, help="use wandb to monitor train process")
parser.add_argument("--tensorboard", type=int, default=1, help="use tensorboard to monitor training process")

def main(args):
    if args.env_num <= 0:
        args.env_num = 1

    eval_env, env = build_env(env_name = args.env_name, env_num = args.env_num, seed = args.seed)

    # set some useful parameter
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    act_bound = eval_env.action_space.high.item()
    layers = args.layers
    hidden_dims = args.hidden_dims if args.hidden_dims else []
    if args.use_tanh == 1:
        args.use_tanh = True
    else:
        args.use_tanh = False
    
    # raise exception
    if not isinstance(layers, int) or layers <= 0:
        raise ValueError(f'--layers expected recieve "int" type and larger than zero! but now recieve {(type(layers), layers)}')
    if not isinstance(hidden_dims, list) or (len(hidden_dims) != layers -1):
        raise ValueError(f"can't not convert {len(hidden_dims)} dims to correspond {layers} network")

    # define dqn agent
    ddpg_agent = DDPG(state_dim = state_dim, 
                    act_dim = act_dim, 
                    act_bound = act_bound,
                    args = args,
                )
    # define relay_buffer
    replay_buffer = ReplayBuffer(capacity = args.capacity)
    replay_buffer.clear()

    # init monitor tools
    if args.wandb == 1:
        print("use wandb : ",args.wandb)
        now_time = datetime.datetime.now().strftime("%Y-%m-%d")
        name = f'{args.env_name}_{now_time}_{os.getpid()}'
        wandb.init(project = 'dqn_train', name = name)
    
    if args.tensorboard == 1:
        print("use tensorboard : ", args.tensorboard)
        log_dir = f'./runs/DQN_{args.env_name}_{os.getpid()}_{int(time.time())}'
        clear_folder(log_dir)
        writer = SummaryWriter(log_dir = log_dir)

    train_total_steps = 0
    eval_total_freq = 0
    for step in range(args.max_train_steps):
        obs, _ = env.reset()
        done = np.zeros(args.env_num)

        # # 重新采样数据
        # if len(relay_buffer) == relay_buffer.capacity:
        #     relay_buffer.clear()
        train_rewards = 0.0
        for k in range(args.max_eposide_step):
            action = ddpg_agent.select_action(obs = obs)
            obs_, reward, done, truncation, _ = env.step(action)
            train_rewards += np.mean(reward)
            for i in range(args.env_num):
                single_obs = obs[i]
                single_obs_ = obs_[i]
                single_reward = reward[i]
                single_done = done[i]
                single_action = action[i]
                replay_buffer.add(single_obs, single_action, single_reward, single_obs_, single_done)
        # print(f'start train, step is {step}')
        loss = ddpg_agent.learn(replay_buffer = replay_buffer)
        write_metric(env_name = args.env_name, 
                    use_wandb = args.wandb, 
                    use_tensorboard = args.tensorboard, 
                    writer = writer,
                    global_step = train_total_steps,
                    actor_train_loss = loss[0],
                    critic_train_loss = loss[1],
                    train_rewards = train_rewards,
                    )
        train_total_steps += 1
        obs = obs_
        # print(f'end train, step is {step}')
        if args.use_lr_decay :
            cur_lr_a = ddpg_agent.lr_a
            cur_lr_c = ddpg_agent.lr_c
            new_lr_a = lr_decay(ddpg_agent.actor_optimizer, cur_step = step, max_step = args.max_train_steps, cur_lr = cur_lr_a)
            new_lr_c = lr_decay(ddpg_agent.critic_optimizer, cur_step = step, max_step = args.max_train_steps, cur_lr = cur_lr_c)
            ddpg_agent.lr_a = new_lr_a
            ddpg_agent.lr_c = new_lr_c

        # evaluate process
        if step % args.evaluate_freq == 0 and step > 0:
            eval_times = args.evaluate_times
            total_rewards = 0.0
            eval_total_steps = 0
            for _ in range(eval_times):
                obs, _ = eval_env.reset()
                done = False
                while not done:
                    action = ddpg_agent.select_action(obs = obs)
                    obs_, reward, done, truncation, _ = eval_env.step(action)
                    replay_buffer.add(obs_, action, reward, obs_, done)
                    obs = obs_
                    total_rewards += reward
                    eval_total_steps += 1
            print(f'env:{args.env_name}, eval num is {eval_total_freq}, eposide rewards is {round(total_rewards / eval_times, 2)}, eposide step is {round(eval_total_steps / eval_times, 2)}')
            write_metric(
                        env_name = args.env_name, 
                        use_wandb = args.wandb, 
                        use_tensorboard = args.tensorboard, 
                        writer = writer,
                        global_step = eval_total_freq,
                        eval_reward = total_rewards / eval_times,
                        eposide_steps = eval_total_steps / eval_times,
                    )
            eval_total_freq += 1
    # 测试保存为gif动态图
    gif_name = f'{args.env_name}_dqn_{int(time.time())}_{os.getpid()}.gif'
    run2gif(env = eval_env, agent = ddpg_agent, gif_name = gif_name)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args = args)
