#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   pso.py
@Time    :   2024/09/10 09:20:48
@Author  :   junewluo 
@Email   :   overtheriver861@gmail.com
@description   :   实现PSO算法框架
'''

import numpy as np

class Particle(object):
    def __init__(self,dim,x_min,x_max,v_min,v_max,func) -> None:
        self.__dim = dim
        self.__pos = np.random.randint(x_min,x_max,size=(1,self.__dim))
        self.__vel = np.random.randint(v_min,v_max,size=(1,self.__dim))

        self.__p_best = np.random.randint(x_min,x_max,size=(1,self.__dim))
        self.__pbest_fitness = func(self.__p_best)
    
    def get_pos(self):
        return self.__pos

    def get_vel(self):
        return self.__vel
    
    def get_pbest(self):
        return self.__p_best
    
    def get_pbest_fit(self):
        return self.__pbest_fitness
    
    def set_pos(self,pos):
        self.__pos = pos
    
    def set_vel(self,vel):
        self.__vel = vel
    
    def set_pbest(self,p_best):
        self.__p_best = p_best
    
    def set_pbest_fit(self,pbest_fitness):
        self.__pbest_fitness = pbest_fitness


class PSO(object):
    def __init__(self,C1,C2,W,dim,p_num,func,x_min=-30,x_max=30,v_min=-10,v_max=10,x_limit_flag=True,v_limit_flag=True,epslion=1e-5) -> None:
        # 维度
        self.dim = dim
        self.p_num = p_num

        # 速度更新参数
        self.C1 = C1
        self.C2 = C2
        self.W = W

        # 位置和速度的限制区间
        self.x_min = x_min
        self.x_max = x_max
        self.v_min = v_min
        self.v_max = v_max
        self.epslion = epslion

        # pso优化过程中是否限制位置和速度
        self.x_limit_flag = x_limit_flag
        self.v_limit_flag = v_limit_flag

        # 适应度函数
        self.func = func
        # 初始化粒子信息
        self.particles = [ 
            Particle(dim = self.dim, x_min = self.x_min, x_max = self.x_max, v_min = self.v_min, v_max = self.v_max, func = self.func)
            for i in range(self.p_num)
        ]


        # 全局最佳位置
        self.g_best = np.random.randint(self.x_min,self.x_max,size=(1,self.dim))
        self.gbest_fitness = self.func(self.g_best)
    
    def cal_fitness(self,pos):
        return self.func(pos)
    
    def update(self):
        """ 参数更新 """

        for par in self.particles:
            # 随机参数
            r1 = np.random.rand()
            r2 = np.random.rand()

            # 获取粒子当前的状态
            cur_pos = par.get_pos()
            cur_vel = par.get_vel()
            cur_pbest = par.get_pbest()
            cur_pbest_fit = par.get_pbest_fit()

            # 速度更新
            next_vel = self.W * cur_vel + self.C1 * r1 * (cur_pbest - cur_pos) + self.C2 * r2 * (self.g_best - cur_pos)
            if self.v_limit_flag:
                next_vel = np.clip(next_vel, self.v_min - self.epslion, self.v_max + self.epslion)
            # 位置更新
            next_pos = next_vel + cur_pos
            if self.x_limit_flag:
                next_pos = np.clip(next_pos, self.x_min - self.epslion, self.x_max + self.epslion)
            
            # 计算next_pos的适应度
            next_pos_fit = self.cal_fitness(next_pos)

            # 更新粒子信息
            par.set_pos(next_pos)
            par.set_vel(next_vel)
            if next_pos_fit < cur_pbest_fit:
                par.set_pbest(next_pos)
                par.set_pbest_fit(next_pos_fit)
            
            # 更新全局最佳位置
            if next_pos_fit < self.gbest_fitness:
                self.gbest_fitness = next_pos_fit
                self.g_best = next_pos

## Test Code ##
# def test():
#     def fit(x):
#         """ 测试优化函数x^2 + y^2 """
#         return np.sum(np.power(x,2.0))
#     C1, C2, W,dim = 2, 2, 1, 4
#     pso = PSO(C1 = C1, C2 = C2, W = W, dim = dim, p_num = 100, func = fit)
#     max_iter = 10000
#     for i in range(max_iter):
#         pso.update()
#         print(f'iter is {i} global best pos is {pso.g_best}, global fitness is {pso.gbest_fitness}')
#     print(f"last best pos is {pso.g_best}, best fitness is {pso.gbest_fitness}")

# test()