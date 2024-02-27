# -*- coding: utf-8 -*-
# @File    : reward.py
# @Date    : 2022-11-28
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
from pathlib import Path
from collections import namedtuple

from parse_config import parse_config
from constant import EPS


Limit = namedtuple('NormalizationLimit', ['upper', 'lower'])
Beta = namedtuple('BetaParams', ['b1', 'b2', 'b3'])


class Reward:
    def __init__(self, config_file: Path = Path('./config/reward_config.yaml')):

        config = parse_config(config_path=config_file)
        self.beta = Beta(config.beta1, config.beta2, config.beta3)

        self.limit_step = Limit(config.upper_step, config.lower_step)
        self.limit_path = Limit(config.upper_path, config.lower_path)

        self.complete, self.step, self.knock_wall, self.subgoal_dead, self.back_road, self.step_cost = config.rewards
        self.path_sum_reward = 0

    def link_func(self):
        def func(x, y, z): return self.beta.b1 * x \
            + self.beta.b2 * (self.limit_step.upper - y) \
            + self.beta.b3 * (self.limit_step.upper - z)
            
        # def func(x, y, z): return self.beta.b1 * x \
        #     + self.beta.b2 * (- y) \
        #     + self.beta.b3 * (- z)

        return func

    def path_func(self):
        def func(x, y, z): return self.beta.b1 * x \
            + self.beta.b2 * (self.limit_path.upper - y) \
            + self.beta.b3 * z
        
        # def func(x, y, z): return self.beta.b1 * x \
        #     + self.beta.b2 * (- y) \
        #     + self.beta.b3 * (- z)
        return func

    def max_min_normalize(self, x, max_v, min_v, limit: Limit):
        x_n = (x - min_v) / (max_v - min_v + EPS)
        x_hat = limit.lower + x_n * (limit.upper - limit.lower)

        return x_hat
