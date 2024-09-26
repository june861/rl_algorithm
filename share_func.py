# -*- encoding: utf-8 -*-
'''
@File    :   share_func.py
@Time    :   2024/09/14 09:42:23
@Author  :   junewluo 
'''
import os
import gym
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from env.flappy_bird import FlappyBirdWrapper
from env.catcher import CatcherWrapper

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
    if env_name in ['CartPole-v1', 'LunarLander-v2','BipedalWalker-v3']:
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
    elif env_name in ['FlappyBird','Catcher']:
        def chunk():
            if env_name == 'FlappyBird':
                env = FlappyBirdWrapper()
            elif env_name == "Catcher":
                env = CatcherWrapper()
            return env
    return chunk


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
    # 6s内播放完gif
    fps = len(frames) // 6
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval = 5)
    anim.save(os.path.join("./gifs/",gif_name), writer="pillow", fps = 120)

def run2gif(env, agent, gif_name):
    
    # 测试模型
    round_count = 0
    last_frames = []
    last_step = 0
    while round_count <= 20:
        frames = []
        round_count += 1
        state, _ = env.reset()
        done = False
        episode_reward = 0.0
        step = 0
        while not done:
            frames.append(env.render())
            step += 1
            action, _  = agent.select_action(state, eval_mode = True)  # We use the deterministic policy during the evaluating
            state, reward, done,trun, _ = env.step(action.item())
            if done: 
                break
            episode_reward += reward
        if step > last_step:
            last_frames = frames
            display_frames_as_gif(last_frames, gif_name)
            last_step = step
        print(f'round{round_count}: total step is {step}, total reward is {episode_reward}')
    

    