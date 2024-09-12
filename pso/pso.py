# -*- encoding: utf-8 -*-
'''
@File    :   pso.py
@Time    :   2024/09/08 10:16:42
@Author  :   junewluo 
'''

import numpy as np

class Particle(object):
    def __init__(self,dim,pos_min,pos_max,v_min,v_max) -> None:
        """初始化函数

        参数:
            dim : 待解决问题的维度设计，粒子速度和位置维度均为dim
            
        """
        self.__pos = np.random.uniform(low = pos_min, high = pos_max, size = (1,dim))
        self.__velocity = np.random.uniform(low = v_min, high = v_max, size = (1,dim))
        self.__dim = dim

        # 粒子的最好的位置记录
        self.__bestPos = self.__pos
        self.__fitness = None

    def get_pos(self):
        return self.__pos

    def get_velocity(self):
        return self.__velocity

    def get_best_pos(self):
        return self.__bestPos
    
    def get_fitness(self):
        return self.__fitness
    
    def set_velocity(self, velocity):
        self.__velocity = velocity

    def set_pos(self,pos):
        self.__pos = pos

    def set_best_pos(self,pos):
        self.__bestPos = pos
    

class PSO(object):
    def __init__(self,C1,C2,Omega,dim,p_num,x_min,x_max,v_min,v_max,limit_x=True,limit_vel=True):

        # 速度更新公式参数
        self.C1 = C1
        self.C2 = C2
        self.Omega = Omega
        self.dim = dim
        self.p_num = p_num

        # 参数限制
        self.x_min = x_min
        self.x_max = x_max
        self.v_min = v_min
        self.v_max = v_max

        # 参数限制标志位
        self.limit_x_tag = limit_x
        self.limit_vel_tag = limit_vel

        # 初始化PSO粒子数
        self.particles = [
            Particle(dim = self.dim, pos_max = x_max, pos_min = x_min, v_min = v_min, v_max = v_max)
            for i in range(p_num)
        ]

        # 记录全局最佳位置
        self.g_best = np.random.uniform(low = x_min, high = x_max, size = (1,dim))

    def cal_fitness(self,func,x):
        return func(x)

    def update(self,func):
        """ 更新每个粒子的位置和速度
        """
        for particle in self.particles:
            r1 = np.random.rand()
            r2 = np.random.rand()
            # 获得粒子当前的速度和位置
            cur_vel = particle.get_velocity()
            cur_pos = particle.get_pos()
            cur_pos_best = particle.get_best_pos()
            # 更新速度
            next_vel = self.Omega * cur_vel + self.C1 * r1 * (cur_pos_best - cur_pos) \
                       + self.C2 * r2 * (self.g_best - cur_pos)
            if self.limit_vel_tag:
                next_vel = np.clip(next_vel, self.v_min, self.v_max)
            if self.limit_x_tag:
                next_pos = np.clip(cur_pos + next_vel, self.x_min, self.x_max)
            next_pos = cur_pos + next_vel
            particle.set_velocity(next_vel)
            particle.set_pos(next_pos)
            # 计算适应度
            fitness = self.cal_fitness(func, particle.get_pos())
            cur_best_fitness = self.cal_fitness(func, cur_pos_best)
            g_best_fitness = self.cal_fitness(func, self.g_best)
            # 更新粒子的局部最优解
            if fitness < cur_best_fitness:
                particle.set_best_pos(particle.get_pos())
            # 更新全局最优解
            if fitness < g_best_fitness:
                self.g_best = particle.get_pos()

## Test Code