import datetime
import time
import os
import wandb
import torch
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from ppo.relaybuffer import RelayBuffer
from ppo.trick import lr_decay, state_norm, reward_norm, KeepMeanStd
from ppo.ppo import PPO
from share_func import clear_folder, run2gif, build_env

parser = argparse.ArgumentParser("Hyperparameter Setting for PPO")
# env variable setting
parser.add_argument("--env_name",type=str,default="CartPole-v1",help="The Env Name of Gym")
parser.add_argument("--env_num",type=int,default=50,help="The number of envs that are activated")
parser.add_argument("--seed", type=int, default=41, help="make random seed statistic")
parser.add_argument("--capacity",type=int,default=int(1e5),help="the capacity of buffer to store data")
# network setting
parser.add_argument("--layers", type=int, default=3, help="the number of hidden layers")
parser.add_argument("--hidden_dims", type=int, nargs='+', default=[128,128], help="The number of neurons in hidden layers of the neural network")
# training variable setting
parser.add_argument("--max_train_steps", type=int, default=2000, help=" Maximum number of training steps")
parser.add_argument("--per_batch_steps", type=int, default=500, help="max step in a round.")
parser.add_argument("--evaluate_freq", type=int, default=20, help="Evaluate the policy every 'evaluate_freq' steps")
parser.add_argument("--evaluate_times", type=int, default=3, help="Evaluate the policy every 'evaluate_freq' steps")
parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
parser.add_argument("--mini_batch_size", type=int, default=512, help="Minibatch size")
parser.add_argument("--lr_a", type=float, default=1e-3, help="Learning rate of actor")
parser.add_argument("--lr_c", type=float, default=1e-4, help="Learning rate of critic")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
# some setting
parser.add_argument("--use_buffer",type=bool,default=True,help="use buffer to store frame datas")
parser.add_argument("--use_gae",type=bool,default=True,help="whether use gae function to cal adv")
parser.add_argument("--grad_clip_param",type=float,default=0.5,help="parameters for model grad clip")
# training trcik setting
parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
parser.add_argument("--set_adam_eps", type=float, default=1e-5, help="Trick 9: set Adam epsilon=1e-5")
parser.add_argument("--use_tanh", type=bool, default=False, help="Trick 10: tanh activation function")
parser.add_argument("--use_ppo_clip",type=bool,default=True,help="Trick 11: use ppo param to clip")
# monitor setting
parser.add_argument("--wandb", type=int, default=0, help="use wandb to monitor train process")
parser.add_argument("--tensorboard", type=int, default=1, help="use tensorboard to monitor training process")


def main(args):
    eval_env, env = build_env(env_name = args.env_name, env_num = args.env_num, seed = args.seed)
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state_dim = eval_env.observation_space.shape[0] 
    action_dim = eval_env.action_space.n
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

    replay_buffer = RelayBuffer(buffer_capacity = args.capacity)
    # build relative parameters
    ppo_params = {
            # ppo algorithm params
            'clip_param' : args.epsilon,

            # 训练参数
            'lr_a': args.lr_a,
            'lr_c': args.lr_c,
            'gamma': args.gamma,
            'lamda': args.lamda,
            'batch_size' : args.batch_size,
            'mini_batch_size' : args.mini_batch_size,

            'use_tanh' : args.use_tanh, # use tanh activate func or ReLU func
            'use_adv_norm' : args.use_adv_norm, # use advantage normalization
            'use_grad_clip' : args.use_grad_clip, # use grad clip in model params.
            'grad_clip_params': 0.5,
            'entropy_coef': args.entropy_coef,
            'device': device,        
    }
    agent = PPO(state_dim = state_dim, 
                act_dim = action_dim, 
                hidden_dims = args.hidden_dims, 
                layer_nums = args.layers, 
                train_params = ppo_params
            )

    if args.wandb == 1:
        now_time = datetime.datetime.now().strftime("%Y-%m-%d")
        name = f'{args.env_name}_{now_time}_{os.getpid()}'
        wandb.init(project = f"ppo-{os.getpid()}-{int(time.time())}", name = name)
    if args.tensorboard == 1:
        # Build a tensorboard 
        log_dir = 'runs/PPO_{}_number_seed_{}'.format(args.env_name, args.seed)
        clear_folder(folder_path = log_dir)
        writer = SummaryWriter(log_dir=log_dir)

    # state_norm_obj = KeepMeanStd(shape = state_dim)
    # action_norm_obj = KeepMeanStd(shape = action_dim)
    # reward_norm_obj = KeepMeanStd(shape = 1)
    optimizer_steps = 0
    while total_steps < args.max_train_steps:
        total_steps += 1
        obs, _ = env.reset()
        done = np.zeros(args.env_num)
        for _ in range(args.per_batch_steps):
            action, a_logprob = agent.select_action(obs)  # Action and the corresponding log probability
            obs_, reward, done, trun, _ = env.step(action)
            for i in range(args.env_num):
                single_obs = obs[i]
                single_obs_ = obs_[i]
                single_reward = reward[i]
                single_done = done[i]
                single_action = action[i]
                single_alogprob = a_logprob[i]
                replay_buffer.add(single_obs, single_action, single_reward, single_obs_, single_alogprob, single_done)
            obs = obs_


        # print(f'step is {total_steps}, get in train, now buffer len is {len(replay_buffer)}')
        learn_steps = 1 + int(len(replay_buffer) / ppo_params['mini_batch_size'])
        # learn_steps = 10
        a_total_loss, c_total_loss = 0.0, 0.0
        for _ in range(learn_steps):
            a_loss, c_loss = agent.learn(replay_buffer = replay_buffer, batch_size = args.batch_size)
            a_total_loss += a_loss
            c_total_loss += c_loss
        a_loss = a_total_loss / learn_steps
        c_loss = c_total_loss / learn_steps
        if args.wandb == 1:
            wandb.log({'actor_loss': a_loss, 'critic_loss': c_loss})
        if args.tensorboard == 1:
            writer.add_scalar(tag = f'actor_loss_{args.env_name}', scalar_value = a_loss, global_step = optimizer_steps)
            writer.add_scalar(tag = f'critic_loss_{args.env_name}', scalar_value = c_loss, global_step = optimizer_steps)
        optimizer_steps += 1
        # agent.update_old_net()
        replay_buffer.clear()

        if args.use_lr_decay :
            cur_lr_a = agent.ppo_params['lr_a']
            cur_lr_c = agent.ppo_params['lr_c']
            new_lr_a = lr_decay(agent.actor_optim, cur_step = total_steps, max_step = args.max_train_steps, cur_lr = cur_lr_a)
            new_lr_c = lr_decay(agent.critic_optim, cur_step = total_steps, max_step = args.max_train_steps, cur_lr = cur_lr_c)
            agent.ppo_params['lr_a'] = new_lr_a
            agent.ppo_params['lr_c'] = new_lr_c

        # Evaluate the policy every 'evaluate_freq' steps
        if total_steps % args.evaluate_freq == 0:
            evaluate_reward = 0
            total_frame = 0
            for _ in range(args.evaluate_times):
                state, _ = eval_env.reset()
                done = False
                episode_reward = 0.0
                episode_frame = 0
                while not done:
                    episode_frame += 1
                    action, _  = agent.select_action(state, eval_mode = True)  # We use the deterministic policy during the evaluating
                    state, reward, done,trun, _ = eval_env.step(action.item())
                    episode_reward += reward
                    if done or episode_frame >= 10000: 
                        break
                total_frame += episode_frame
                evaluate_reward += episode_reward
            evaluate_rewards.append(evaluate_reward / args.evaluate_times)
            evaluate_num += 1
            print(f"total_step is {total_steps}\t evaluate_num:{evaluate_num} \t evaluate_reward:{evaluate_reward / args.evaluate_times} \t")
            if args.wandb == 1 :
                wandb.log({"eval_rewards": (evaluate_reward / args.evaluate_times), "eval_steps" : (total_frame / args.evaluate_times)})
            if args.tensorboard == 1:
                writer.add_scalar('step_rewards_{}'.format(args.env_name), evaluate_reward / args.evaluate_times, global_step = evaluate_num)
                writer.add_scalar('frame_length_{}'.format(args.env_name), total_frame / args.evaluate_times, global_step = evaluate_num)
            
    if args.tensorboard:
        writer.close()

    # test
    gif_name = f'{args.env_name}_ppo_{int(time.time())}_{os.getpid()}.gif'
    run2gif(env = eval_env, agent = agent, gif_name = gif_name)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
