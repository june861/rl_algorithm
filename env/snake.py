# -*- encoding: utf-8 -*-
'''
@File    :   pong.py
@Time    :   2024/09/27 16:01:07
@Author  :   junewluo 
'''

import numpy as np
from gym import Env, spaces
from ple import PLE
from ple.games import Snake

class SnakeWrapper(Env):
    metadata = {
        'render.mode':['human','rgb_array'],
    }

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.game = Snake()
        self.p = PLE(self.game, **kwargs)
        self.action_set = self.p.getActionSet()

        # 3个输入状态：见函数self._get_obs
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(13,), dtype=np.float32)
        # 两个输出状态：跳或者不跳
        self.action_space = spaces.Discrete(len(self.action_set))

    def _get_obs(self):
        states = self.p.getGameState()
        # {
        #     'snake_head_x': 32.0,
        #     'snake_head_y': 32.0,
        #     'food_x': 12,
        #     'food_y': 18,
        #     'snake_body': [0.0, 3.0, 6.0],
        #     'snake_body_pos': [[32.0, 32.0], [29.0, 32.0], [26.0, 32.0]]
        # }

        snake_head_x = states['snake_head_x']
        snake_head_y = states['snake_head_y']
        food_x = states['food_x']
        food_y = states['food_y']
        obs = [snake_head_x, snake_head_y, food_x, food_y] + states['snake_body']
        snake_body_pos = states['snake_body_pos']
        for pos in snake_body_pos:
            obs.extend(pos)
        obs = np.array(obs)

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