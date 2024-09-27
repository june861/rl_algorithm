# -*- encoding: utf-8 -*-
'''
@File    :   pong.py
@Time    :   2024/09/27 16:01:07
@Author  :   junewluo 
'''

import numpy as np
from gym import Env, spaces
from ple import PLE
from ple.games import PuckWorld

class PuckWorldWrapper(Env):
    metadata = {
        'render.mode':['human','rgb_array'],
    }

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.game = PuckWorld()
        self.p = PLE(self.game, **kwargs)
        self.action_set = self.p.getActionSet()

        # 3个输入状态：见函数self._get_obs
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(8,), dtype=np.float32)
        # 两个输出状态：跳或者不跳
        self.action_space = spaces.Discrete(len(self.action_set))

    def _get_obs(self):
        states = self.p.getGameState()
        # {
        #     'player_x': 4.5,
        #     'player_y': 4.5,
        #     'player_velocity_x': 0,
        #     'player_velocity_y': 0,
        #     'good_creep_x': 56.49369639963426,
        #     'good_creep_y': 19.45319673968546,
        #     'bad_creep_x': 64,
        #     'bad_creep_y': 64
        # }

        # agent location
        player_x = states['player_x']
        player_y = states['player_y']
        # agent velocity
        player_vel_x = states['player_velocity_x']
        player_vel_y = states['player_velocity_y']
        # cpu-player location
        good_creep_x = states['good_creep_x']
        good_creep_y = states['good_creep_y']

        bad_creep_x = states['bad_creep_x']
        bad_creep_y = states['bad_creep_y']

        obs = np.array([
            player_x, player_y, player_vel_x, player_vel_y,
            good_creep_x, good_creep_y, bad_creep_x, bad_creep_y

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