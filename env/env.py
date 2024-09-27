# -*- encoding: utf-8 -*-
'''
@File    :   pong.py
@Time    :   2024/09/27 16:01:07
@Author  :   junewluo 
'''

import numpy as np
from collections.abc import Iterable
from gym import Env, spaces
from ple import PLE

class PLEEnv2GymEnvWrapper(Env):
    metadata = {
        'render.mode':['human','rgb_array'],
    }

    def __init__(self, func, **kwargs) -> None:
        super().__init__()
        self.game = func()
        self.p = PLE(self.game, **kwargs)
        self.action_set = self.p.getActionSet()
        self.state_dict = self.p.getGameState()

        # 观测空间
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(len(self.state_dict),), dtype=np.float32)
        # 动作空间
        self.action_space = spaces.Discrete(len(self.action_set))

    def _get_obs(self):
        states = self.p.getGameState()
        obs = []
        for k, v in states.items():
            m = np.array([v]).reshape(1,-1)
            while len(m.shape) > 1:
                m = m.squeeze(0)
            for m_ in m:
                obs.append(m_.item())
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