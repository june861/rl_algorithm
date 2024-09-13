import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import argparse
from ppo.relaybuffer import RelayBuffer
from ppo.ppo import PPO

def main(args, env_name, number, seed):
    env = gym.make(env_name)
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    layer_nums = 3
    hidden_dims = [128,128]
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print(f'======== run ppo algorithm =========')
    print("env = {}".format(env_name))
    print("state_dim = {}".format(state_dim))
    print("action_dim = {}".format(action_dim))
    print("max_episode_steps = {}".format(args.max_episode_steps))
    print('max_train_steps = {}'.format(args.max_train_steps))
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

            # trick params
            'off_policy' : True, # use off-policy or on-policy
            'use_buffer' : False, # use buffer to store or not
            'use_tanh' : True, # use tanh activate func or ReLU func
            'use_adv_norm' : False, # use advantage normalization
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
    writer = SummaryWriter(log_dir='runs/PPO_{}_number_seed_{}'.format(env_name, number, seed))

    while total_steps < args.max_train_steps:
        total_steps += 1
        state, _  = env.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            action, a_logprob = agent.select_action(state)  # Action and the corresponding log probability
            state_, reward, done, trun, _ = env.step(action)
            if done:
                break
            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            replay_buffer.add(state, action, reward, state_, a_logprob, dw, done)
            state = state_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if len(replay_buffer) > args.batch_size:
                learn_steps = 1 + len(replay_buffer) / replay_buffer.buffer_capacity
                for _ in range(learn_steps):
                    agent.learn(replay_buffer = replay_buffer, batch_size = args.batch_size)
                    # 修改agent
                agent.update_old_net()

        # Evaluate the policy every 'evaluate_freq' steps
        if total_steps % args.evaluate_freq == 0:
            times = 2
            evaluate_reward = 0
            for _ in range(times):
                state, _ = env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action, _  = agent.select_action(state, eval_mode = True)  # We use the deterministic policy during the evaluating
                    state, reward, done,trun, _ = env.step(action)
                    if done: 
                        break
                    episode_reward += reward
                evaluate_reward += episode_reward
            evaluate_num += 1
            evaluate_rewards.append(evaluate_reward / times)
            print(f"total_step is {total_steps}\t evaluate_num:{evaluate_num} \t evaluate_reward:{evaluate_reward} \t")
            writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step = total_steps)
            # Save the rewards
            # if evaluate_num % args.save_freq == 0:
            #     np.save('./data_train/PPO_discrete_env_{}_number_{}_seed_{}.npy'.format(env_name, number, seed), np.array(evaluate_rewards))
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO")
    parser.add_argument("--max_train_steps", type=int, default=int(200), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=int, default=10, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
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

    args = parser.parse_args()

    env_name = ['CartPole-v1', 'LunarLander-v2']
    env_index = 0
    main(args, env_name=env_name[env_index], number=1, seed=0)

