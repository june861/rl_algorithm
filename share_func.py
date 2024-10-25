# -*- encoding: utf-8 -*-
'''
@File    :   share_func.py
@Time    :   2024/09/14 09:42:23
@Author  :   junewluo 
'''
import os
import wandb
import gym
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from env.flappy_bird import FlappyBirdWrapper
from env.catcher import CatcherWrapper
from env.pixelcopter import PixelcopterWrapper
from env.pong import PongWrapper
from env.puckworld import PuckWorldWrapper
from env.raycastmaze import RaycastMazeWrapper
from env.snake import SnakeWrapper
from env.waterworld import WaterWorldWrapper


ple_games = [
    "FlappyBird", "Catcher", "Pixelcopter", "Pong", "PuckWorld", "RaycastMaze", "Snake", "WaterWorld"
]
ple_games_func = [
    FlappyBirdWrapper, CatcherWrapper, PixelcopterWrapper, PongWrapper, PuckWorldWrapper, RaycastMazeWrapper, SnakeWrapper, WaterWorldWrapper
]

def _t2n(input):
    pass


def clear_folder(folder_path, rm_file = True, rm_dir = True):
    """ remove dirs and files from the folder_path.

    Args:
        folder_path (_str_): 
        rm_file (bool, optional): _description_. Defaults to True.
        rm_dir (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if not os.path.exists(folder_path):
        print(f'folder path {folder_path} not exist! Create it.')
        os.mkdir(folder_path)
        return 0
    
    list_ = os.listdir(folder_path)
    # clear all files in the folder_path
    for f_d in list_:
        f_d_path = os.path.join(folder_path, f_d)
        is_file = os.path.isfile(f_d_path)
        is_dir = os.path.isdir(f_d_path)
        if is_file and rm_file:
            print(f'remove file: {f_d_path}!')
            os.remove(f_d_path)
        if is_dir and rm_dir:
            print(f'remove dir: {f_d_path}')
            os.rmdir(f_d_path)
    return 1


def make_env(env_name, seed, idx, run_name, capture_video = False):
    """ build env.

    Args:
        env_name (_str_): which env to use.
        seed (_int_): set random seed.
        idx (_int_): index of the env list.
        run_name (_str_): the path to save env videos.
        capture_video (bool, optional): if need to capture videos. Defaults to False.

    Returns:
        _type_: return env .
    """
    if env_name in ple_games: 
        def chunk():
            env = ple_games_func[ple_games.index(env_name)]()
            return env
    # else env_name in ['CartPole-v1', 'LunarLander-v2','BipedalWalker-v3']:
    else:
        def chunk():
            env = gym.make(env_name)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            # env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
    return chunk

def build_env(env_name, env_num, seed):
    # build envs
    if env_name in ple_games:
        func = ple_games_func[ple_games.index(env_name)]
        eval_env = func()
    else:
        eval_env = gym.make(env_name, render_mode = 'rgb_array')
    
    train_envs = [ make_env(env_name = env_name, seed = seed, idx = i, run_name = f'{env_name}_video{i}') for i in range(env_num) ]
    envs = gym.vector.SyncVectorEnv(train_envs)
    return eval_env, envs



def generate_frame_data(env, env_index, agent, lock, relay_buffer, capacity):
    """ use env to generated frame data.

    Args:
        env (_type_): base env. create from gym.make(env_name)
        env_index (_int_): which env.
        agent (_class_): agent define, must include method:select_action() and actor network.
        lock (_lock_): use lock to sync different frame data when use multiprocess.
        relay_buffer (_list_): a list but create from multiprocess.Manager().list()
        capacity (_int_): the capacity of relay buffer to store frame data.
    """
    
    # reset env
    state, _ = env.reset()
    frame_data = []
    while True:
        action, a_logprob = agent.select_action(state)
        state, reward, done, truncation, _ = env.step(action)
        frame_data.append((state, action, reward, state, a_logprob, done))
        if done:
            break
    with lock:
        for d_index, (state, action, reward, state, a_logprob, done) in enumerate(frame_data):
            if len(relay_buffer) > capacity:
                return
            relay_buffer.append((state, action, reward, state, a_logprob, done))
        # print(f'env{env_index} write frame_data to buffer!')


def draw_metric():
    pass



def display_frames_as_gif(frames, gif_name):
    """ save frame data to gif

    Args:
        frames (_list_): frame data.
    """
    if not os.path.exists("./gifs/"):
        os.mkdir("./gifs/")

    patch = plt.imshow(frames[0])
    plt.axis("off")
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval = 5)
    anim.save(os.path.join("./gifs/",gif_name), writer="pillow", fps = 120)

def run2gif(env, agent, gif_name):
    
    # 测试模型
    round_count = 0
    last_frames = []
    last_step = 0
    max_steps = 50000
    
    while round_count <= 5:
        frames = []
        round_count += 1
        state, _ = env.reset()
        done = False
        episode_reward = 0.0
        step = 0
        while not done:
            frames.append(env.render())
            step += 1
            action = agent.select_action(state, eval_mode = True)  # We use the deterministic policy during the evaluating
            if isinstance(action, tuple):
                action = action[0]
            try:
                state, reward, done,trun, _ = env.step(action.item())
            except:
                state, reward, done,trun, _ = env.step(action)
            episode_reward += reward
            if done or step > max_steps: 
                break
        if step > last_step:
            last_frames = frames
            display_frames_as_gif(last_frames, gif_name)
            last_step = step
        # 释放内存
        del frames
        print(f'round{round_count}: total step is {step}, total reward is {episode_reward}')
    

def write_metric(env_name, use_wandb, use_tensorboard, writer, global_step, **kwargs):
    if use_wandb == 1:
        wandb.log(kwargs)
    if use_tensorboard == 1:
        for key,val in kwargs.items():
            writer.add_scalar(tag = f'{env_name}_{key}', scalar_value = val, global_step = global_step)