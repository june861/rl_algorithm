# -*- encoding: utf-8 -*-
'''
@File    :   dqn_main.py
@Time    :   2024/09/20 17:07:49
@Author  :   junewluo 
'''
import argparse
import torch
import gym
import os
import time
import wandb
import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter
from dqn.dqn import DQN, RelayBuffer
from share_func import make_env, clear_folder
from dqn.trick import lr_decay
from env.flappy_bird import FlappyBirdWrapper
from env.catcher import CatcherWrapper

parser = argparse.ArgumentParser("DQN Parameter Setting")

# env setting
parser.add_argument("--env_name",type=str,default="CartPole-v1",help="The Env Name of Gym")
parser.add_argument("--env_num",type=int,default=50,help="The number of envs that are activated")
parser.add_argument("--max_eposide_step", type=int, default=500, help="the max step in one eposide game")
parser.add_argument("--seed", type=int, default=1, help="random seed")
# training setting
parser.add_argument("--max_train_steps", type=int, default=500, help="the max train steps")
parser.add_argument("--learn_freq", type=int, default=5, help="the q net learning frequency")
parser.add_argument("--evaluate_freq", type=int, default=10, help="evaluate frequency")
parser.add_argument("--evaluate_times", type=int, default=3, help="evaluate times in one evaluation eposide")
parser.add_argument("--lr", type=float, default=2e-3, help="learning rate of Deep Q-network")
parser.add_argument("--gamma", type=float, default=0.9, help="discounted element")
parser.add_argument("--epsilon", type=float, default=0.4,help="The probability of randomly generated actions")
parser.add_argument("--epsilon_min", type=float, default=1e-3,help="The minimum probability of randomly generated actions")
parser.add_argument("--epsilon_decay", type=float, default=1e-4)
parser.add_argument("--batch_size",type=int,default=2048,help="mini batch size to sample from buffer")
parser.add_argument("--mini_batch_size",type=int,default=128,help="mini batch size to sample from buffer")
parser.add_argument("--capacity",type=int,default=int(5000 ),help="the capacity of buffer to store data")
parser.add_argument("--use_lr_decay", type=bool, default=True, help="use learning rate decay")
parser.add_argument("--update_target", type=int, default=200, help="update target network")
# network setting
parser.add_argument("--layers",type=int,default=3,help="the number of layer in q_net")
parser.add_argument("--hidden_dims", type=int, nargs='+', default=[128, 128], help='Sizes of the hidden layers (e.g., --hidden_sizes 50 30)')
# monitor setting
parser.add_argument("--wandb", type=bool, default=False, help="use wandb to monitor train process")
parser.add_argument("--tensorboard", type=bool, default=True, help="use tensorboard to monitor training process")

def write_metric(env_name, use_wandb, use_tensorboard, writer, global_step, **kwargs):
    if use_wandb:
        wandb.log(kwargs)
    if use_tensorboard :
        for key,val in kwargs.items():
            writer.add_scalar(tag = f'{env_name}_{key}', scalar_value = val, global_step = global_step)

def build_env(env_name, env_num, seed):
    # build envs
    if env_name == "FlappyBird":
        eval_env = FlappyBirdWrapper()
    elif env_name == "Catcher":
        eval_env = CatcherWrapper()
    else:
        eval_env = gym.make(args.env_name, render_mode = 'rgb_array')
    
    train_envs = [ make_env(env_name = args.env_name, seed = args.seed, idx = i, run_name = f'{env_name}_video{i}') for i in range(env_num) ]
    envs = gym.vector.SyncVectorEnv(train_envs)
    return eval_env, envs

def main(args):
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.env_num <= 0:
        args.env_num = 1

    eval_env, env = build_env(env_name = args.env_name, env_num = args.env_num, seed = args.seed)

    # set some useful parameter
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        state_dim = eval_env.observation_space.shape[0]
    except:
        state_dim = eval_env.observation_space.n
    act_dim = eval_env.action_space.n
    layers = args.layers
    hidden_dims = args.hidden_dims if args.hidden_dims else []

    # raise exception
    if not isinstance(layers, int) or layers <= 0:
        raise ValueError(f'--layers expected recieve "int" type and larger than zero! but now recieve {(type(layers), layers)}')
    if not isinstance(hidden_dims, list) or (len(hidden_dims) != layers -1):
        raise ValueError(f"can't not convert {len(hidden_dims)} dims to correspond {layers} network")

    # define dqn agent
    dqn_agent = DQN(state_dim = state_dim, 
                    hidden_dims = hidden_dims, 
                    layers = layers, 
                    act_dim = act_dim, 
                    args = args
                )
    # define relay_buffer
    relay_buffer = RelayBuffer(capacity = args.capacity)
    relay_buffer.clear()

    # init monitor tools
    if args.wandb:
        now_time = datetime.datetime.now().strftime("%Y-%m-%d")
        name = f'dqn_train_{now_time}_{os.getpid()}_{int(time.time())}'
        wandb.init(projetc = 'dqn_train', name = name)
    
    if args.tensorboard:
        log_dir = f'./runs/DQN_{args.env_name}_{os.getpid()}_{int(time.time())}'
        clear_folder(log_dir)
        writer = SummaryWriter(log_dir = log_dir)
    else:
        writer = None

    train_total_steps = 0
    eval_total_freq = 0
    for step in range(args.max_train_steps):
        obs, _ = env.reset()
        done = np.zeros(args.env_num)

        # # 重新采样数据
        # if len(relay_buffer) == relay_buffer.capacity:
        #     relay_buffer.clear()

        for k in range(args.max_eposide_step):
            action = dqn_agent.select_action(obs = obs)
            obs_, reward, done, truncation, _ = env.step(action)
            for i in range(args.env_num):
                single_obs = obs[i]
                single_obs_ = obs_[i]
                single_reward = reward[i]
                single_done = done[i]
                single_action = action[i]
                relay_buffer.add(single_obs, single_action, single_reward, single_obs_, single_done)
            if (k+1) % args.learn_freq == 0:
                loss = dqn_agent.learn(relay_buffer = relay_buffer)
                write_metric(env_name = args.env_name, 
                            use_wandb = args.wandb, 
                            use_tensorboard = args.tensorboard, 
                            writer = writer,
                            global_step = train_total_steps,
                            loss = loss
                            )
                train_total_steps += 1
                dqn_agent.update_epsilon()
            obs = obs_

        if args.use_lr_decay :
            cur_lr = dqn_agent.dqn_params['lr']
            new_lr = lr_decay(dqn_agent.optimizer, cur_step = step, max_step = args.max_train_steps, cur_lr = cur_lr)
            dqn_agent.dqn_params['lr'] =new_lr

        # evaluate process
        if step % args.evaluate_freq == 0:
            print(f'q_net has been trained {train_total_steps} times')
            eval_times = args.evaluate_times
            total_rewards = 0.0
            eval_total_steps = 0
            for _ in range(eval_times):
                obs, _ = eval_env.reset()
                done = False
                while not done:
                    action = dqn_agent.select_action(obs = obs).item()
                    obs_, reward, done, truncation, _ = eval_env.step(action)
                    relay_buffer.add(obs_, action, reward, obs_, done)
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
if __name__ == '__main__':
    args = parser.parse_args()
    main(args = args)


