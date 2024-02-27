# -*- coding: utf-8 -*-
import os
from pathlib import Path
from collections import namedtuple
import random
from collections.abc import Iterable
from itertools import tee

import networkx
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import networkx as nx
from networkx.algorithms.approximation import steiner_tree
import matplotlib.pyplot as plt
from log import MyLog
from net import *
from env_config import *

np.random.seed(2022)
torch.random.manual_seed(2022)
random.seed(2022)

mylog = MyLog(Path(__file__), filesave=True, consoleprint=True)
logger = mylog.logger
RouteParams = namedtuple("RouteParams", ('bw', 'delay', 'loss'))
plt.style.use("seaborn-whitegrid")


class KMB:

    @staticmethod
    def parse_xml_topology(topology_path):
        """
            parse topology from topology.xml
        :param topology_path: 拓扑的xml文件路径
        :return: topology graph, networkx.Graph()
                 nodes_num,  int
                 edges_num, int
        """
        tree = ET.parse(topology_path)
        root = tree.getroot()
        topo_element = root.find("topology")
        graph = networkx.Graph()
        for child in topo_element.iter():
            # parse nodes
            if child.tag == 'node':
                node_id = int(child.get('id'))
                graph.add_node(node_id)
            # parse link
            elif child.tag == 'link':
                from_node = int(child.find('from').get('node'))
                to_node = int(child.find('to').get('node'))
                graph.add_edge(from_node, to_node)

        nodes_num = len(graph.nodes)
        edges_num = len(graph.edges)

        print('nodes: ', nodes_num, '\n', graph.nodes, '\n',
              'edges: ', edges_num, '\n', graph.edges)
        return graph, nodes_num, edges_num

    def pkl_file_path_yield(self, pkl_dir, n: int = 3000, step: int = 1):
        """
            生成保存的pickle文件的路径, 按序号递增的方式生成
            :param pkl_dir: Path, pkl文件的目录
            :param n: 截取
            :param step: 间隔
        """
        a = os.listdir(pkl_dir)
        assert n < len(a), "n should small than len(a)"
        b = sorted(a, key=lambda x: int(x.split('-')[0]))
        for p in b[:n:step]:
            yield pkl_dir / p


    @staticmethod
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

    @staticmethod
    def spanning_tree(graph, weight=None):
        """
            生成树算法
        :param graph: networkx.graph
        :param weight: 计算的权重
        :return: iterator 最小生成树
        """
        spanning_tree = nx.algorithms.minimum_spanning_tree(graph, weight)
        return spanning_tree

    def get_tree_params(self, tree, graph):
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
            bw += graph[r[0]][r[1]]["bw"]
            delay += graph[r[0]][r[1]]["delay"]
            loss += graph[r[0]][r[1]]["loss"]
            num += 1
        bw_mean = self.env.find_end_to_end_max_bw(tree, self.env.start, self.env.ends_constant).mean()
        return bw_mean, delay / num, loss / num, len(tree.edges)

    @staticmethod
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

    def get_kmb_params(self, graph, start_node, end_nodes):
        """
            获得以 bw 为权重的 steiner tree 返回该树的 bw和
            获得以 delay 为权重的 steiner tree 返回该树的 delay和
            获得以 loss 为权重的 steiner tree 返回该树的 loss和
            获得以 hope 为权重的 steiner tree 返回该树的 长度length
        :param graph: 图
        :param start_node: 源节点
        :param end_nodes: 目的节点
        :return: bw, delay, loss, length
        """
        _g = self.modify_bw_weight(graph)
        # kmb算法 计算权重为-bw
        kmb_bw_tree = self.kmb_algorithm(_g, start_node, end_nodes,
                                         weight='bw')
        bw_bw, bw_delay, bw_loss, bw_length = self.get_tree_params(kmb_bw_tree, graph)

        # kmb算法 计算权重为delay
        kmb_delay_tree = self.kmb_algorithm(graph, start_node, end_nodes,
                                            weight='delay')
        delay_bw, delay_delay, delay_loss, delay_length = self.get_tree_params(kmb_delay_tree, graph)

        # kmb算法 计算权重为loss
        kmb_loss_tree = self.kmb_algorithm(graph, start_node, end_nodes,
                                           weight='loss')
        loss_bw, loss_delay, loss_loss, loss_length = self.get_tree_params(kmb_loss_tree, graph)

        # kmb算法 为None
        kmb_hope_tree = self.kmb_algorithm(graph, start_node, end_nodes, weight=None)
        length_bw, length_delay, length_loss, length_length = self.get_tree_params(kmb_hope_tree, graph)

        bw_ = [bw_bw, delay_bw, loss_bw, length_bw]
        delay_ = [bw_delay, delay_delay, loss_delay, length_delay]
        loss_ = [bw_loss, delay_loss, loss_loss, length_loss]
        length_ = [bw_length, delay_length, loss_length, length_length]
        return bw_, delay_, loss_, length_

    def get_spanning_tree_params(self, graph):
        """
            获得以 bw 为权重的 spanning tree 返回该树的 bw和
            获得以 delay 为权重的 spanning tree 返回该树的 delay和
            获得以 loss 为权重的 spanning tree 返回该树的 loss和
            获得以 hope 为权重的 spanning tree 返回该树的 长度length
        :param graph:
        :return:
        """
        _g = self.modify_bw_weight(graph)
        spanning_bw_tree = self.spanning_tree(_g, weight='bw')
        bw, _, _, _ = self.get_tree_params(spanning_bw_tree, graph)
        spanning_delay_tree = self.spanning_tree(graph, weight='delay')
        _, delay, _, _ = self.get_tree_params(spanning_delay_tree, graph)
        spanning_loss_tree = self.spanning_tree(graph, weight='loss')
        _, _, loss, _ = self.get_tree_params(spanning_loss_tree, graph)
        spanning_length_tree = self.spanning_tree(graph, weight=None)
        _, _, _, length = self.get_tree_params(spanning_length_tree, graph)

        return bw, delay, loss, length

    