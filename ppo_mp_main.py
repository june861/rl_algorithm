import os
import gym.vector
import gym.vector
import torch
import numpy as np
import gym
# import gymnasium as gym
import argparse
import wandb
import warnings
warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter
from ppo.trick import (
    state_norm, reward_norm, adv_norm, 
    lr_decay, orthogonal_initialization,
    ppo_clip_param_annealing, KeepMeanStd
)
from ppo.relaybuffer import RelayBuffer
from ppo_mp.ppo import PPO
from share_func import clear_folder, make_env, generate_frame_data
from multiprocessing import Process, Manager, Lock 


def build_ppo_param(args, device):
    ppo_params = {
            # ppo algorithm params
            'clip_param' : args.epsilon,

            # 训练参数
            'lr_a': args.lr_a,
            'lr_c': args.lr_c,
            'gamma': args.gamma,
            'lamda': args.lamda,
            'batch_size' : args.batch_size,
            'mini_batch_size': args.mini_batch_size,

            # trick params

            'off_policy' : args.use_off_policy, # use off-policy or on-policy
            'use_buffer' : args.use_buffer, # use buffer to store or not
            "use_ppo_clip":args.use_ppo_clip , # use ppo clip param annealing
            "use_adv_norm" : args.use_adv_norm, # use advantage normalization
            "use_state_norm" : args.use_state_norm, # use state normalization
            "use_reward_norm" : args.use_reward_norm, # use reward normalization
            'use_tanh' : args.use_tanh, # use tanh activate func or ReLU func
            'use_adv_norm' : args.use_adv_norm, # use advantage normalization
            'use_grad_clip' : args.use_grad_clip, # use grad clip in model params.
            'use_gae': args.use_gae,
            'grad_clip_params': args.grad_clip_param,
            'use_lr_decay': args.use_lr_decay,
            'entropy_coef': args.entropy_coef,
            'device': device,        
    }

    return ppo_params


def main(args, number, seed):
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # build envs
    train_envs = [ make_env(env_name = args.env_name, seed = seed,idx = i, run_name = f'{args.env_name}_video{i}') for i in range(args.env_num) ]
    if args.use_multiprocess:
        envs = gym.vector.AsyncVectorEnv(train_envs)
    else:
        envs = gym.vector.SyncVectorEnv(train_envs)
    val_env = gym.make(args.env_name)
    eval_env = gym.make(args.env_name, render_mode = "human")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n
    layer_nums = 3
    hidden_dims = [64,64]
    # args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print(f'======== run ppo algorithm =========')
    print("env = {}".format(args.env_name))
    print("device = {}".format(device))
    print("state_dim = {}".format(state_dim))
    print("action_dim = {}".format(action_dim))
    print('max_train_steps = {}'.format(args.max_train_steps))
    print('eval_freq = {}'.format(args.evaluate_freq))
    print(f'=====================================')

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    # init the agent
    ppo_params = build_ppo_param(args = args, device = device)
    agent = PPO(state_dim = state_dim, 
                act_dim = action_dim, 
                hidden_dims = hidden_dims, 
                layer_nums = layer_nums, 
                train_params = ppo_params
            )
    # monitor tools init
    if args.monitor == "wandb":
        wandb.init(project = "ppo_mp")
    else:
        # clear dir or make dir
        tensorboard_logdir = 'runs/PPO_mp_{}_seed_{}'.format(args.env_name, number, seed)
        clear_folder(folder_path = tensorboard_logdir)
        # Build a tensorboard
        writer = SummaryWriter(log_dir=tensorboard_logdir)
    # pre-define, batch_obs shape is [args.per_batch_steps, args.env_num, envs.single_observation_space.shape]
    batch_obs = torch.zeros((args.per_batch_steps, args.env_num) + envs.single_observation_space.shape)
    batch_actions = torch.zeros((args.per_batch_steps, args.env_num) + envs.single_action_space.shape)
    batch_log_probs = torch.zeros((args.per_batch_steps, args.env_num))
    batch_rewards = torch.zeros((args.per_batch_steps, args.env_num))
    batch_dones = torch.zeros((args.per_batch_steps, args.env_num))
    batch_values = torch.zeros((args.per_batch_steps, args.env_num))

    # generated data
    obs, _  = envs.reset()
    obs = torch.tensor(obs)
    done = torch.zeros(args.env_num)

    # training process
    for train_step in range(args.max_train_steps):
        ## Trick5 : Learning Rate Decay ##

        for step in range(args.per_batch_steps):
            batch_obs[step] = obs
            batch_dones[step] = done

            # notice: action, a_logprob and value is a numpy.ndarray
            action, a_logprob = agent.select_action(obs)
            value = agent.get_value(obs)
            batch_values[step] = torch.tensor(value).flatten()
            batch_actions[step] = torch.tensor(action)
            batch_log_probs[step] = torch.tensor(a_logprob)

            next_obs, reward, done, truncation, _ = envs.step(action)
            batch_rewards[step] = torch.tensor(reward).view(-1)
            obs, done = torch.Tensor(next_obs), torch.Tensor(done)
        
        # start optimizer process while collect enough datas.
        loss =  agent.learn(batch_obs = batch_obs.to(device),
                            batch_actions = batch_actions.to(device),
                            batch_log_probs=  batch_log_probs.to(device),
                            batch_rewards = batch_rewards.to(device),
                            batch_dones = batch_dones.to(device),
                            batch_values = batch_values.to(device),
                            obs = obs.to(device),
                            dones = done.to(device),
                            update_epoch = 10
                        )
        for index, (a_loss, c_loss) in enumerate(loss):
            total_steps += 1
            if args.monitor == "wandb":
                wandb.log({'actor_loss': a_loss, 'critic_loss': c_loss})
            elif args.monitor == "tensorboard":
                writer.add_scalar(tag = f'train_actor_loss_{args.env_name}', scalar_value = a_loss, global_step = total_steps)
                writer.add_scalar(tag = f'train_critic_loss_{args.env_name}', scalar_value = c_loss, global_step = total_steps)
        if train_step % args.evaluate_freq == 0:
            eval_times = 5
            round_count = 0
            val_reward = 0
            for k in range(eval_times):
                done = False
                step = 0
                episode_reward = 0.0
                state, _ = val_env.reset()
                # val_env.render()
                while not done:
                    step += 1
                    done = False
                    action, _  = agent.select_action(state, eval_mode = True)  # We use the deterministic policy during the evaluating
                    state_, reward, done,trun, _ = val_env.step(action)
                    episode_reward += reward
                    state = state_
                val_reward += episode_reward
                round_count += step
            print(f'step is {train_step}, validation reward is {val_reward / eval_times}, every round count is {round_count / eval_times}')
            if args.monitor == "wandb":
                wandb.log({'eval_reward':val_reward / eval_times, "eval_steps": (round_count / eval_times)})
            else:
                writer.add_scalar(tag = f'validation_reward_{args.env_name}', scalar_value = val_reward / eval_times, global_step = evaluate_num)
                writer.add_scalar(tag = f'validation_rounds_{args.env_name}', scalar_value = round_count / eval_times, global_step = evaluate_num)
            evaluate_num += 1

    # save model, can choice
    agent.save_checkpoint(only_net = False)

    # # 测试模型
    # round_count = 0
    # while True:
    #     round_count += 1
    #     state, _ = eval_env.reset()
    #     eval_env.render()
    #     done = False
    #     episode_reward = 0.0
    #     step = 0
    #     while not done:
    #         step += 1
    #         action, _  = agent.select_action(state, eval_mode = True)  # We use the deterministic policy during the evaluating
    #         state, reward, done,trun, _ = eval_env.step(action)
    #         if done: 
    #             break
    #         episode_reward += reward
    #     print(f'round{round_count}: total step is {step}, total reward is {episode_reward}, every step reward is {episode_reward / step}')            


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO")
    # env variable setting
    parser.add_argument("--env_name",type=str,default="CartPole-v1",help="The Env Name of Gym")
    parser.add_argument("--env_num",type=int,default=100,help="The number of envs that are activated")
    parser.add_argument("--use_multiprocess",type=bool,default=False,help="use multi-process to generated frame data.")
    # training variable setting
    parser.add_argument("--max_train_steps", type=int, default=50000, help=" Maximum number of training steps")
    parser.add_argument("--per_batch_steps", type=int, default=500, help="max step in a round.")
    parser.add_argument("--evaluate_freq", type=int, default=20, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=5e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    # some setting
    parser.add_argument("--use_off_policy",type=bool,default=False,help="PPO use off-policy or on-policy")
    parser.add_argument("--use_buffer",type=bool,default=True,help="use buffer to store frame datas")
    parser.add_argument("--use_gae",type=bool,default=True,help="whether use gae function to cal adv")
    parser.add_argument("--grad_clip_param",type=float,default=0.5,help="parameters for model grad clip")
    # training trcik setting
    
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--use_ppo_clip",type=bool,default=True,help="Trick 11: use ppo param to clip")
    # monitor tools setting
    parser.add_argument("--monitor",type=str,default="tensorboard",choices=["tensorboard","wandb"],help="use Dynamic tools to monitor train process")

    args = parser.parse_args()
    main(args, number=1, seed=0)
