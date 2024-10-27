# -*- encoding: utf-8 -*-
'''
@File    :   ppo_continous_main.py
@Time    :   2024/10/22 21:29:40
@Author  :   junewluo 
'''

import os
import wandb
import sys
import torch
import datetime
import argparse
import numpy as np
from logger.logger import Logger
from tensorboardX import SummaryWriter
from ppo_continous.ppo_continous import PPO_continous
from ppo_continous.replay_buffer import ReplayBuffer
from share_func import build_env, run2gif

def script_conf():
    parser = argparse.ArgumentParser("PPO Continous Argumentor")
    parser.add_argument("--env_name", type=str, default=None, help="ENV Name")
    parser.add_argument("--n_rollout_threads", type=int, default=8, help="how many env will be used")
    # network parameter
    parser.add_argument("--policy_dist", type=str, default="Gaussion", help="which distribution of policy net to be used", choices = ["Gaussion", "Beta"])
    parser.add_argument("--policy_layers", type=int, default=3, help="the number of layers for policy net")
    parser.add_argument("--policy_hidden_dims", nargs="+", default=[128,128], help="the input dim of hidden layers")
    parser.add_argument("--lr_a", type=float, default=1e-3, help="the learning rate of actor network")
    
    parser.add_argument("--critic_layers", type=int, default=3, help="the number of layers for critic net")
    parser.add_argument("--critic_hidden_dims", nargs="+", default=[128,128], help="the input dim of hidden layers")
    parser.add_argument("--lr_c", type=float, default=2e-4, help="the learning rate of critic network")
    
    # common parameter
    parser.add_argument("--batch_size", type=int, default=2056, help="Batch Size")
    parser.add_argument("--mini_batch_size", type=int, default=256, help="Mini Batch Size")
    parser.add_argument("--capacity", type=int, default=1e4, help="capacity of replay buffer")
    parser.add_argument("--eval_freq", type=int, default=20, help="the evaluation stage in training stage")
    parser.add_argument("--eval_times", type=int, default=5, help="the eval times")
    parser.add_argument("--max_iter_steps", type=int, default=1000, help="max iteration of training steps")
    parser.add_argument("--ppo_epoch", type=int, default=10, help="the number of iteration for ppo update")
    parser.add_argument("--max_step_per_batch", type=int, default=500, help="")
    parser.add_argument("--use_tanh", action='store_true', default=False)
    parser.add_argument("--use_orthogonal_init", action="store_true", default=False)
    parser.add_argument("--lambda_", type=float, default=0.99, help="discounter factor")
    parser.add_argument("--gamma", type=float, default=0.98, help="")
    parser.add_argument("--epsilon", type=float,default=0.2, help="the weight for dist entropy")
    parser.add_argument("--entropy_coef", type=float, default=0.02, help="the discout factor of entropy coef")
    parser.add_argument("--use_gae", action="store_true", default=False, help="use gae func to cal ppo adv")
    parser.add_argument("--use_policy_grad_norm", action = "store_false", default=True, help="use value normalization on policy net")
    parser.add_argument("--use_value_grad_norm", action="store_false", default= True, help="use grad normalization on value net")
    parser.add_argument("--use_adv_norm", action="store_true", default=True, help="use normalization for advantage")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="the maxinum of gradien")

    # device setting
    parser.add_argument("--use_cuda", action="store_true", default=False, help="use gpu to help accelerating training")
    parser.add_argument("--cuda_rank", type=int, default=0, help="which gpu device used to train")
    parser.add_argument("--use_wandb", action="store_true", default=False)

    args = parser.parse_args()
    check_args(args)

    return args

def interaction(envs, actions, reset_tag = False):
    if reset_tag:
        obs, _ = envs.reset()
        return obs
    else:
        obs_, reward, done, trun, _ = envs.step(actions)
        return obs_, reward, done, trun


def check_args(args):
    if args.n_rollout_threads <= 0:
        raise ValueError(f"[--n_rollout_thread] expected a integer larger than 0, but now recieve {args.n_rollout_thread}")
    if args.policy_layers - 1 != len(args.policy_hidden_dims):
        raise ValueError(f'[--args.policy_hidden_dims] call a __len__ must be {args.policy_layers - 1}, but now recieve is {args.policy_hidden_dims}')
    if args.critic_layers - 1 != len(args.critic_hidden_dims):
        raise ValueError(f'[--args.critic_hidden_dim] call a __len__ must be {args.critic_layers - 1}, but now recieve is {args.critic_hidden_dims}')
    if args.mini_batch_size > args.batch_size:
        raise ValueError(f'[--mini_batch_size] must smaller than [--batch_size]')

def main(args):
    eval_env, envs = build_env(env_name = args.env_name, env_num = args.n_rollout_threads, seed = 1)
    logger = Logger(log_file = None, std_out_console = True)
    if args.use_cuda and torch.cuda.is_available():
        if torch.cuda.device_count() <= args.cuda_rank:
            logger.warning(f'cuda device only have {torch.cuda.device_count()}, but recieve cuda_rank is {args.cuda_rank}. It will set to 0 as default!')
            args.cuda_rank = 0 
        args.device = torch.device(f"cuda:{args.cuda_rank}")
    else:
        args.device = torch.device(f"cpu")
    
    args.obs_dim = eval_env.observation_space.shape[0] 
    args.act_dim = eval_env.action_space.shape[0]
    args.max_action = eval_env.action_space.high.item()

    if args.use_wandb:
        curr_time = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        wandb.init(project = f'ppo-continous', name = f"{args.env_name}_{curr_time}")
    else:
        curr_time = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        log_dir = f'./runs/{curr_time}_{args.env_name}'
        writer = SummaryWriter(log_dir = log_dir)
    
    ppo_agent = PPO_continous(args = args)
    replay_buffer = ReplayBuffer(args = args)
    eval_steps = 0
    # start to train
    for j in range(args.max_iter_steps):
        obs = interaction(envs = envs, actions = None, reset_tag = True)
        for i in range(args.max_step_per_batch):
            action, action_log_probs = ppo_agent.selection_action(obs)
            obs_, reward, done, trun = interaction(envs, action)
            replay_buffer.add(obs, reward, action, action_log_probs, obs_, done)
            obs = obs_

        # ppo update
        train_info = ppo_agent.learn(replay_buffer)
        # record metric
        if args.use_wandb:
            wandb.log(train_info)
        else:
            for k,v in train_info.items():
                writer.add_scalar(k,v)


        if (j+1) % args.eval_freq == 0:
            total_eval_reward = 0.0
            for k in range(args.eval_times):
                obs = interaction(envs, None, True)
                for p in range(args.max_step_per_batch):
                    action, _ = ppo_agent.selection_action(obs)
                    obs_, reward, done, trun = interaction(envs, action)
                    total_eval_reward += reward.mean()
                    obs = obs_
            total_eval_reward /= args.eval_times
            if args.use_wandb:
                wandb.log({'eval_reward': total_eval_reward})
            else:
                writer.add_scalar('eval_reward', total_eval_reward, eval_steps)
            eval_steps += 1

    if not args.use_wandb:
        writer.close()

    # test
    gif_name = f'{args.env_name}_ppo_continous_{os.getpid()}.gif'
    run2gif(env = eval_env, agent = ppo_agent, gif_name = gif_name, generate_times = 1)


if __name__ == '__main__':
    args = script_conf()
    main(args = args)