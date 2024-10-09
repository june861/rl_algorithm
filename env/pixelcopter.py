# -*- encoding: utf-8 -*-
'''
@File    :   pixelcopter.py
@Time    :   2024/09/27 15:03:22
@Author  :   junewluo 
'''

import os
import numpy as np
from gym import Env, spaces
from ple import PLE
from ple.games import Pixelcopter

class PixelcopterWrapper(Env):
    metadata = {
        'render.mode':['human','rgb_array'],
    }

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.game = Pixelcopter()
        self.p = PLE(self.game, **kwargs)
        self.action_set = self.p.getActionSet()

        # 3个输入状态：见函数self._get_obs
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(7,), dtype=np.float32)
        # 两个输出状态：跳或者不跳
        self.action_space = spaces.Discrete(len(self.action_set))

    def _get_obs(self):
        states = self.p.getGameState()

        # player location
        player_y = states['player_y']
        # player velocity
        player_vel = states['player_vel']
        # dist to ceil or floor
        player_dist_to_ceil = states['player_dist_to_ceil']
        player_dist_to_floor = states['player_dist_to_floor']
        # the distance between player and next gate.
        next_gate_dist_to_player = states['next_gate_dist_to_player']
        next_gate_block_top = states['next_gate_block_top']
        next_gate_block_bottom = states['next_gate_block_bottom']

        obs = np.array([
            player_y, player_vel, player_dist_to_ceil, player_dist_to_floor,
            next_gate_dist_to_player, next_gate_block_top, next_gate_block_bottom
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