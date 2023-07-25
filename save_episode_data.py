# -*- coding: utf-8 -*-
# @File    : save_episode_data.py
# @Date    : 2022-12-21
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来- 
import pickle
import numpy as np
from collections import Counter

from transition import EpisodeMetrics


def save_episodes_data(data_path, episode_metrics: EpisodeMetrics):
    """save data of multicast tree metrics of all episodes.

    Args:
        data_path (Path): path to save
        episode_metrics (EpisodeMetrics): 'reward', 'subgoal_reward', 'tree_reward',
                                        'bw', 'delay', 'loss', 'length', 'alley', 'steps'
    """
    np.save(data_path / "episode_reward", episode_metrics.reward)
    np.save(data_path / "episode_subgoal_reward", episode_metrics.subgoal_reward)
    np.save(data_path / "episode_tree_reward", episode_metrics.tree_reward)
    np.save(data_path / "episode_steps", episode_metrics.steps)
    np.save(data_path / "episode_bw", episode_metrics.bw)
    np.save(data_path / "episode_delay", episode_metrics.delay)
    np.save(data_path / "episode_loss", episode_metrics.loss)
    np.save(data_path / "episode_length", episode_metrics.length)
    

def save_episode_subgoal_counter(data_path, subgoal_counter: Counter):
    """save episode subgoals list to Counter

    Args:
        data_path (str): save data path
        subgoals (list): subgoal list
    """
    
    print(f'[subgoal selected counter] {subgoal_counter}')
    with open(data_path / 'subgoals.pkl', 'wb+') as f:
        pickle.dump(subgoal_counter, f, 0)
