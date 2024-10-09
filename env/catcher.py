# -*- encoding: utf-8 -*-
'''
@File    :   catcher.py
@Time    :   2024/09/22 21:44:18
@Author  :   junewluo 
'''

import os
import numpy as np
from gym import Env, spaces
from ple import PLE
from ple.games import Catcher

class CatcherWrapper(Env):
    metadata = {
        'render.mode':['human','rgb_array'],
    }
    # 如果想把画面渲染出来，就传参display_screen=True
    def __init__(self, **kwargs):
        self.game = Catcher()
        self.p = PLE(self.game, **kwargs)
        self.action_set = self.p.getActionSet()

        # 3个输入状态：见函数self._get_obs
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(4,), dtype=np.float32)
        # 两个输出状态：跳或者不跳
        self.action_space = spaces.Discrete(len(self.action_set))

    def _get_obs(self):
        # 获取游戏的状态
        state = self.game.getGameState()
        player_x = state['player_x']
        player_vel = state['player_vel']
        fruit_x = state['fruit_x']
        fruit_y = state['fruit_y']
        # 将这些信息封装成一个数据返回
        return np.array([player_x, player_vel, fruit_x, fruit_y])

    def reset(self):
        self.p.reset_game()
        return self._get_obs(), dict()

    def step(self, action):
        reward = self.p.act(self.action_set[action])
        obs = self._get_obs()
        done = self.p.game_over()
        return obs, reward, done, False, dict()

    def seed(self, *args, **kwargs):
        pass

    def render(self, *args, **kwargs):
        """ default return rgb array
        """
        rgb_array = self.p.getScreenRGB()
        rgb_array = np.rot90(rgb_array, k=-1)
        return rgb_array