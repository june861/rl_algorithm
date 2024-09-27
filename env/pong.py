# -*- encoding: utf-8 -*-
'''
@File    :   pong.py
@Time    :   2024/09/27 16:01:07
@Author  :   junewluo 
'''

import numpy as np
from gym import Env, spaces
from ple import PLE
from ple.games import Pong

class PongWrapper(Env):
    metadata = {
        'render.mode':['human','rgb_array'],
    }

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.game = Pong()
        self.p = PLE(self.game, **kwargs)
        self.action_set = self.p.getActionSet()
        
        # 3个输入状态：见函数self._get_obs
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(7,), dtype=np.float32)
        # 两个输出状态：跳或者不跳
        self.action_space = spaces.Discrete(len(self.action_set))

    def _get_obs(self):
        states = self.p.getGameState()
        # {   
        #     'player_y': 24.0,
        #     'player_velocity': 0,
        #     'cpu_y': 24.0,
        #     'ball_x': 32.0,
        #     'ball_y': 24.0,
        #     'ball_velocity_x': 36.0,
        #     'ball_velocity_y': 7.182433798184739
        # }

        # agent location
        player_y = states['player_y']
        # agent velocity
        player_vel = states['player_velocity']
        # cpu-player location
        cpu_y = states['cpu_y']
        # ball location-x
        ball_X = states['ball_x']
        # ball location-y
        ball_y = states['ball_y']
        # ball velocity-x
        ball_velocity_x = states['ball_velocity_x']
        # ball velocity-y
        ball_velocity_y = states['ball_velocity_y']

        obs = np.array([
            player_y, player_vel, cpu_y,
            ball_X, ball_y, ball_velocity_x, ball_velocity_y
        ])

        return obs

    def reset(self):
        self.p.reset_game()
        return self._get_obs(), dict()
    
    def step(self, action):
        reward = self.p.act(self.action_set[action])
        obs = self._get_obs()
        done = self.p.game_over()
        truncation = False
        return obs, reward, done, truncation, dict()

    def seed(self, *args, **kwargs):
        pass

    def render(self, *args, **kwargs):
        """ default return rgb array
        """
        rgb_array = self.p.getScreenRGB()
        rgb_array = np.rot90(rgb_array, k = -1)
        return rgb_array