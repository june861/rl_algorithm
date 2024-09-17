import torch
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from ppo.relaybuffer import RelayBuffer
from ppo.trick import lr_decay, state_norm, reward_norm, KeepMeanStd
from ppo.ppo import PPO
from share_func import clear_folder
def main(args, env_name, number, seed):
    env = gym.make(env_name)
    eval_env = gym.make(env_name, render_mode = "human")
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state_dim = env.observation_space.shape[0] 
    action_dim = env.action_space.n
    layer_nums = 3
    hidden_dims = [64,64]
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print(f'======== run ppo algorithm =========')
    print("env = {}".format(env_name))
    print("state_dim = {}".format(state_dim))
    print("action_dim = {}".format(action_dim))
    print("max_episode_steps = {}".format(args.max_episode_steps))
    print('max_train_steps = {}'.format(args.max_train_steps))
    print('eval_freq = {}'.format(args.evaluate_freq))
    print(f'=====================================')

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    replay_buffer = RelayBuffer()
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

            # trick params
            'off_policy' : False, # use off-policy or on-policy
            'use_buffer' : False, # use buffer to store or not  
            'use_tanh' : False, # use tanh activate func or ReLU func
            'use_adv_norm' : True, # use advantage normalization
            'use_grad_clip' : False, # use grad clip in model params.
            'grad_clip_params': 0.5,
            'entropy_coef': args.entropy_coef,
            'device': device,        
    }
    agent = PPO(state_dim = state_dim, 
                act_dim = action_dim, 
                hidden_dims = hidden_dims, 
                layer_nums = layer_nums, 
                train_params = ppo_params
            )

    # Build a tensorboard 
    log_dir = 'runs/PPO_{}_number_seed_{}'.format(env_name, number, seed)
    clear_folder(folder_path = log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # state_norm_obj = KeepMeanStd(shape = state_dim)
    # action_norm_obj = KeepMeanStd(shape = action_dim)
    # reward_norm_obj = KeepMeanStd(shape = 1)
    optimizer_steps = 0
    while total_steps < args.max_train_steps:
        total_steps += 1
        while len(replay_buffer) < args.batch_size:
            state, _  = env.reset()
            episode_steps = 0
            done = False
            while not done:
                episode_steps += 1
                action, a_logprob = agent.select_action(state)  # Action and the corresponding log probability
                state_, reward, done, trun, _ = env.step(action)
                if done:
                    break
                # if args.use_state_norm:
                #     state_norm_ = state_norm(state_, state_norm_obj)
                # if args.use_reward_norm:
                #     reward_norm_ = reward_norm(reward, norm_obj = reward_norm_obj)
                # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                # dw means dead or win,there is no next state s';
                # but when reaching the max_episode_steps,there is a next state s' actually.
                if done and episode_steps != args.max_episode_steps:
                    dw = True
                else:
                    dw = False

                replay_buffer.add(state, action, reward, state_, a_logprob, dw, done)
                state = state_

        # When the number of transitions in buffer reaches batch_size,then update
        if len(replay_buffer) >= args.batch_size:
            # print(f'step is {total_steps}, get in train, now buffer len is {len(replay_buffer)}')
            learn_steps = int(ppo_params['batch_size'] / ppo_params['mini_batch_size'])
            # learn_steps = 1
            a_total_loss, c_total_loss = 0.0, 0.0
            for _ in range(learn_steps):
                a_loss, c_loss = agent.learn(replay_buffer = replay_buffer, batch_size = args.batch_size)
                a_total_loss += a_loss
                c_total_loss += c_loss
            a_loss = a_total_loss / learn_steps
            c_loss = c_total_loss / learn_steps
            writer.add_scalar(tag = f'actor_loss_{env_name}', scalar_value = a_loss, global_step = optimizer_steps)
            writer.add_scalar(tag = f'critic_loss_{env_name}', scalar_value = c_loss, global_step = optimizer_steps)
            optimizer_steps += 1
            # agent.update_old_net()
            replay_buffer.clear()

        if args.use_lr_decay  :
            cur_lr_a = agent.ppo_params['lr_a']
            cur_lr_c = agent.ppo_params['lr_c']
            new_lr_a = lr_decay(agent.actor_optim, cur_step = total_steps, max_step = args.max_train_steps, cur_lr = cur_lr_a)
            new_lr_c = lr_decay(agent.critic_optim, cur_step = total_steps, max_step = args.max_train_steps, cur_lr = cur_lr_c)
            agent.ppo_params['lr_a'] = new_lr_a
            agent.ppo_params['lr_c'] = new_lr_c

        # Evaluate the policy every 'evaluate_freq' steps
        if total_steps % args.evaluate_freq == 0:
            times = 1
            evaluate_reward = 0
            total_frame = 0
            for j in range(times):
                state, _ = env.reset()
                done = False
                episode_reward = 0.0
                episode_frame = 0
                while not done:
                    episode_frame += 1
                    action, _  = agent.select_action(state, eval_mode = True)  # We use the deterministic policy during the evaluating
                    state, reward, done,trun, _ = env.step(action)
                    if done: 
                        break
                    episode_reward += reward
                total_frame += episode_frame
                evaluate_reward += episode_reward
            evaluate_rewards.append(evaluate_reward / times)
            evaluate_num += 1
            print(f"total_step is {total_steps}\t evaluate_num:{evaluate_num} \t evaluate_reward:{evaluate_reward / times} \t")
            writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_reward / times, global_step = evaluate_num)
            writer.add_scalar('frame_length_{}'.format(env_name), total_frame / times, global_step = evaluate_num)
            
            # Save the rewards
            # if evaluate_num % args.save_freq == 0:
            #     np.save('./data_train/PPO_discrete_env_{}_number_{}_seed_{}.npy'.format(env_name, number, seed), np.array(evaluate_rewards))
    writer.close()

    # 测试模型
    round_count = 0
    while True:
        round_count += 1
        state, _ = eval_env.reset()
        eval_env.render()
        done = False
        episode_reward = 0.0
        step = 0
        while not done:
            step += 1
            action, _  = agent.select_action(state, eval_mode = True)  # We use the deterministic policy during the evaluating
            state, reward, done,trun, _ = eval_env.step(action)
            if done: 
                break
            episode_reward += reward
        print(f'round{round_count}: total step is {step}, total reward is {episode_reward}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO")
    parser.add_argument("--max_train_steps", type=int, default=int(200), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=int, default=20, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20,  help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=512, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=1e-3, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=1.0, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization") 
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    env_name = ['CartPole-v1', 'LunarLander-v2']
    env_index = 0
    main(args, env_name=env_name[env_index], number=1, seed=0)
