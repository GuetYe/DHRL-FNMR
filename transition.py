# -*- coding: utf-8 -*-
# @File    : transition.py
# @Date    : 2022-11-25
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来- 

from collections import namedtuple

# 简单的类
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

EpisodeMetrics = namedtuple("EpisodeMetrics", ['reward', 'subgoal_reward', 'tree_reward',
                                               'bw', 'delay', 'loss', 'length', 'alley', 'steps'])
