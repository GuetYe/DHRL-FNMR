# -*- coding: utf-8 -*-
# @File    : main.py
# @Date    : 2022-12-07
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
import time
from pathlib import Path
import os
import random
from collections import Counter
import argparse
from collections.abc import Iterable
from itertools import tee

import numpy as np
import torch
import networkx as nx
from networkx.algorithms.approximation import steiner_tree
from matplotlib import pyplot as plt

from parse_config import parse_config
from hierarchical_dqn import HierarchicalDQNAgent
from env import Env

from transition import EpisodeMetrics


def set_seed(seed):
    """set all random seed

    Args:
        seed (int): seed
    """
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def pkl_file_path_yield(pkl_dir: Path, s=0, n: int = 12, step: int = 1):
    """Generates a path to a saved pickle file in ascending order

    Args:
        pkl_dir (Path): pickle file path
        s (int, optional): start index. Defaults to 10.
        n (int, optional): num of pkl. Defaults to 3000.
        step (int, optional): interval. Defaults to 1.

    Yields:
        Path: path of pkl file.
    """
    a = os.listdir(pkl_dir)
    assert n < len(a), "n should small than len(a)"
    # print([x.split('-')[0] for x in a])
    b = sorted(a, key=lambda x: int(x.split('-')[0]))
    for p in b[s:n:step]:
        yield pkl_dir / p


def modify_controller_param(controller, modify_param_name, modify_param):
    if modify_param_name is not None and modify_param is not None:

        if modify_param_name == 'lr':
            controller.lr = modify_param
        elif modify_param_name == 'batchsize':
            controller.batch_size = modify_param
        elif modify_param_name == 'egreedy':
            controller.epsilon_decay = modify_param
        elif modify_param_name == 'gamma':
            controller.gamma = modify_param
        elif modify_param_name == 'updatefrequency':
            controller.update_target_net_frequency = modify_param


def kmb_algorithm(graph, src_node, dst_nodes, weight=None):
    """
        经典KMB算法 组播树

    :param graph: networkx.graph
    :param src_node: 源节点
    :param dst_nodes: 目的节点
    :param weight: 计算的权重
    :return: 返回图形的最小Steiner树的近似值
    """
    terminals = [src_node] + dst_nodes
    st_tree = steiner_tree(graph, terminals, weight)
    return st_tree


def modify_bw_weight(graph):
    """
        将delay取负，越大表示越小
    :param graph: 图
    :return: weight
    """
    _g = graph.copy()
    for edge in graph.edges:
        _g[edge[0]][edge[1]]['bw'] = 1 / (graph[edge[0]][edge[1]]['bw'] + 1)
    return _g


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_end2end_max_bw(route_graph, env):
    """get a list of bottle bandwidth from source to each termination.

    Returns:
        list: a list of bottle bandwidth
    """
    bottle_bw_to_ends = []
    for t in env.nodes_pair.terminations_c:
        p = nx.shortest_path(route_graph,
                             source=t, target=env.nodes_pair.source)
        bbw = env.max_min_metrics.bw.max  # bottle bandwidth

        for e in pairwise(p):  # pairwise iter
            # temp = self.env_graph.graph.edges[e[0], e[1]]['bw']
            index1, index2 = e[0] - 1, e[1] - 1
            temp = env.normal_matrixes.bw[index1, index2]
            if temp < bbw:
                bbw = temp

        bottle_bw_to_ends.append(bbw)

    return np.array(bottle_bw_to_ends)


def get_tree_params(tree, env):
    """
        计算tree的bw, delay, loss 参数和
    :param tree: 要计算的树
    :param graph: 计算图中的数据
    :return: bw, delay, loss, len
    """
    bw, delay, loss = 0, 0, 0
    if isinstance(tree, nx.Graph):
        edges = tree.edges
    elif isinstance(tree, Iterable):
        edges = tree
    else:
        raise ValueError("tree param error")
    num = 0
    for r in edges:
        bw += env.env_graph.graph[r[0]][r[1]]["bw"]
        delay += env.env_graph.graph[r[0]][r[1]]["delay"]
        loss += env.env_graph.graph[r[0]][r[1]]["loss"]
        num += 1
    bw_mean = get_end2end_max_bw(tree, env).mean()
    return bw_mean, delay / num, loss / num, len(tree.edges)


def get_kmb_params(env, start_node, end_nodes):
    graph = env.env_graph.graph.copy()
    _g = modify_bw_weight(graph)
    # kmb算法 计算权重为-bw
    kmb_bw_tree = kmb_algorithm(_g, start_node, end_nodes,
                                weight='bw')
    bw_bw, bw_delay, bw_loss, bw_length = get_tree_params(kmb_bw_tree, env)

    # kmb算法 计算权重为delay
    kmb_delay_tree = kmb_algorithm(graph, start_node, end_nodes,
                                   weight='delay')
    delay_bw, delay_delay, delay_loss, delay_length = get_tree_params(
        kmb_delay_tree, env)

    # kmb算法 计算权重为loss
    kmb_loss_tree = kmb_algorithm(graph, start_node, end_nodes,
                                  weight='loss')
    loss_bw, loss_delay, loss_loss, loss_length = get_tree_params(
        kmb_loss_tree, env)

    # kmb算法 为None
    kmb_hope_tree = kmb_algorithm(graph, start_node, end_nodes, weight=None)
    length_bw, length_delay, length_loss, length_length = get_tree_params(
        kmb_hope_tree, env)

    bw_ = [bw_bw, delay_bw, loss_bw, length_bw]
    delay_ = [bw_delay, delay_delay, loss_delay, length_delay]
    loss_ = [bw_loss, delay_loss, loss_loss, length_loss]
    length_ = [bw_length, delay_length, loss_length, length_length]
    return bw_, delay_, loss_, length_


def plot_compare_figure(rl_result, kmb_result, drl_result, x_label, y_label, title, mode='bar'):
    width = 0.18
    kmb_bw = [kmb_result[i][0] for i in range(len(kmb_result))]
    kmb_delay = [kmb_result[i][1] for i in range(len(kmb_result))]
    kmb_loss = [kmb_result[i][2] for i in range(len(kmb_result))]
    kmb_length = [kmb_result[i][3] for i in range(len(kmb_result))]

    if mode == 'bar':
        plt.bar(range(len(kmb_result)), rl_result, width, label='DHRL-FNMR')
        plt.bar([x + width for x in range(len(drl_result))],
                kmb_bw, width, label='DRL-M4MR')
        plt.bar([x + 2 * width for x in range(len(kmb_result))],
                kmb_bw, width, label='KMB_bw')
        plt.bar([x + 3 * width for x in range(len(kmb_result))],
                kmb_delay, width, label='KMB_delay')
        plt.bar([x + 4 * width for x in range(len(kmb_result))],
                kmb_loss, width, label='KMB_loss')
    else:
        plt.plot(range(len(kmb_result)), kmb_bw,
                 ".-", label='KMB_bw', alpha=0.8)
        plt.plot(range(len(kmb_result)), kmb_delay,
                 ".-", label='KMB_delay', alpha=0.8)
        plt.plot(range(len(kmb_result)), kmb_loss,
                 ".-", label='KMB_loss', alpha=0.8)
        plt.plot(range(len(kmb_result)), rl_result,
                 '*-', label='DHRL-FNMR', alpha=0.8)

    plt.xticks(range(len(kmb_result)), range(1, len(kmb_result) + 1))
    # plt.xticks(range(len(kmb_result)), range(len(kmb_result)), rotation=0, fontsize='small')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.title(title)
    plt.legend(bbox_to_anchor=(0., 1.0), loc='lower left', ncol=4, )

    _path = Path('./images')
    if _path.exists():
        plt.savefig(_path / f'{title}.png', dpi=800,
                    bbox_inches='tight', pad_inches=0)
    else:
        _path.mkdir(exist_ok=True)
        plt.savefig(_path / f'{title}.png', dpi=800,
                    bbox_inches='tight', pad_inches=0)

    plt.close()


def calculate_compare_result(episode, kmb):
    kmb_bw, kmb_delay, kmb_loss, kmb_length = [], [], [], []
    _average, _max, _min = [], [], []

    for data, episode_data in zip(kmb, episode):
        data = np.array(data)
        episode_data = np.array(episode_data)
        # mask = np.where(data != 0)

        # _delta = (episode_data - data[mask]) / data[mask]

        # kmb_bw.append(_delta[0])
        # kmb_delay.append(_delta[1])
        # kmb_loss.append(_delta[2])
        # kmb_length.append(_delta[3])
        if data[0] != 0 or episode_data == data[0]:
            kmb_bw.append((episode_data - data[0]) / (data[0] + 1e-30))
        if data[1] != 0 or episode_data == data[1]:
            kmb_delay.append((episode_data - data[1]) / (data[1] + 1e-30))
        if data[2] != 0 or episode_data == data[2]:
            kmb_loss.append((episode_data - data[2]) / (data[2] + 1e-30))
        if data[3] != 0 or episode_data == data[3]:
            kmb_length.append((episode_data - data[3]) / (data[3] + 1e-30))

    def _cal_metric(metric_list):
        _average.append(round(np.array(metric_list).mean() * 100, 2))
        _max.append(round(np.array(metric_list).max() * 100, 2))
        _min.append(round(np.array(metric_list).min() * 100, 2))

    _cal_metric(kmb_bw)
    _cal_metric(kmb_delay)
    _cal_metric(kmb_loss)
    _cal_metric(kmb_length)

    return _average, _max, _min


def compare_test(config_file: Path = Path('./config/train_config.yaml')):
    config = parse_config(config_file)

    set_seed(config.random_seed)

    hdqn = HierarchicalDQNAgent(meta_controller_config=Path(config.meta_controller_config),
                                meta_controller_model_config=Path(
                                    config.meta_controller_model_config),
                                controller_config=Path(
                                    config.controller_config),
                                controller_model_config=Path(
                                    config.controller_model_config),
                                train_or_eval=config.mode,
                                beta_frames=2e3)

    env = Env(Path(config.config_file),
              Path(config.reward_config_file),
              config.mode)

    hdqn.load_weight(config.in_policy_weight_file,
                     config.ex_policy_weight_file)

    image_save_path = Path(config.image_save_path)
    image_save_path.mkdir(exist_ok=True)

    pkl_step = config.pkl_step
    pkl_num = config.pkl_num
    pkl_start = config.pkl_start
    pkl_cut_num = pkl_start + pkl_num * pkl_step

    start_node = config.start_node
    end_nodes = config.end_nodes

    print(f"start_node: {start_node}")
    print(f"end_nodes: {end_nodes}")

    kmb_bw, kmb_delay, kmb_loss, kmb_length = [], [], [], []
    episode_metrics = EpisodeMetrics([], [], [], [], [], [], [], [], [])
    
    for index, pkl_path in enumerate(
            pkl_file_path_yield(Path(config.pkl_weight_path), s=pkl_start, n=pkl_cut_num, step=pkl_step)):

        state = env.reset(source=start_node,
                            terminations=end_nodes, pkl_file=pkl_path)

        while True:
            subgoal = hdqn.meta_controller_sample_max(state)  # index of subgoal

            if env.subgoal_step(subgoal):
                subgoal_matrix = env.matrixes.subgoal

                while True:
                    action = hdqn.controller_sample_max(state, subgoal_matrix)
                    step_return = env.step(action)
                    state = step_return.next_state
                    
                    if step_return.state_flag == 'part_done' or step_return.state_flag == "all_done":
                        hdqn.meta_controller_state = None
                        break

                if step_return.state_flag == "all_done":
                    bw, delay, loss, length, alley = env.get_route_params()
                    episode_metrics.bw.append(bw)
                    episode_metrics.delay.append(delay)
                    episode_metrics.loss.append(loss)
                    episode_metrics.length.append(length)
                    episode_metrics.steps.append(env.step_counter)
                    episode_metrics.alley.append(alley)

                    # kmb 算法
                    bw, delay, loss, length = get_kmb_params(env, start_node, end_nodes)
                    kmb_bw.append(bw)
                    kmb_delay.append(delay)
                    kmb_loss.append(loss)
                    kmb_length.append(length)
                    break

                # if create a ring, restart
                if step_return.next_state is None:  # step_return.state_flag == "road_back"
                    # print("RING RESTART")
                    break
            else:
                pass
    
    drl_bw = np.load(config.drl_bw_npy)
    drl_delay = np.load(config.drl_delay_npy)
    drl_loss = np.load(config.drl_loss_npy)
    drl_length = np.load(config.drl_length_npy)
    
    plot_compare_figure(episode_metrics.bw, kmb_bw, drl_bw, "traffic", "mean bw", "bw")
    plot_compare_figure(episode_metrics.delay, kmb_delay, drl_delay, 
                        "traffic", "mean delay", "delay")
    plot_compare_figure(episode_metrics.loss, kmb_loss, drl_loss,
                        "traffic", "mean loss", "loss")
    plot_compare_figure(episode_metrics.length, kmb_length, drl_length,
                        "traffic", "mean length", "length")

    bw_average, bw_max, bw_min = calculate_compare_result(episode_metrics.bw, kmb_bw)
    delay_average, delay_max, delay_min = calculate_compare_result(episode_metrics.delay, kmb_delay)
    loss_average, loss_max, loss_min = calculate_compare_result(episode_metrics.loss, kmb_loss)
    length_average, length_max, length_min = calculate_compare_result(episode_metrics.length, kmb_length)

    print(f"bw: [bw_, delay_, loss_, length_]:\n average: {bw_average}, max: {bw_max}, min: {bw_min}")
    print(f"delay: [bw_, delay_, loss_, length_]:\n average: {delay_average}, max: {delay_max}, min: {delay_min}")
    print(f"loss: [bw_, delay_, loss_, length_]:\n average: {loss_average}, max: {loss_max}, min: {loss_min}")
    print(f"length: [bw_, delay_, loss_, length_]:\n average: {length_average}, max: {length_max}, min: {length_min}")



if __name__ == '__main__':
    compare_test()
