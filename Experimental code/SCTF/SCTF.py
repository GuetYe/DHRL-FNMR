#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2023/11/23 9:43
@File:SCTF.py
@Desc:SCTF算法，实现吞吐量翻倍
"""
import copy
import networkx as nx
import json
from networkx.readwrite import json_graph
import random
import pickle


class SCTF:
    def __init__(self,graph,src,stds):
        """
        初始化
        :param graph: 图信息
        :param src: 起点
        :param stds: 终点集
        """
        self.topo = copy.deepcopy(graph)
        self.src = src
        self.stds = copy.deepcopy(stds)
        self.result = {}

    # 一步迭代
    def step(self):
        std = random.choice(self.stds)
        self.stds.remove(std)
        shortest_path = nx.shortest_path(self.topo,self.src,std,weight='weight')
        self.result[(self.src,std)] = shortest_path
        for i,node in enumerate(shortest_path):
            if i>0:
                self.topo[shortest_path[i-1]][node]['weight'] = 0

    def run(self):
        while len(self.stds) > 0:
            self.step()
        return self.result

def run(file_path,src,dsts):
    from my_main import generating_tree, tree_evaluate,Data_process
    # 内部参数
    beta = [1,1,1]
    negative_attr = [1,0,0]
    src = src[0]  # 为了适应列表输入
    with open(file_path, 'rb') as pkl_file:
        graph = pickle.load(pkl_file)
    data_process = Data_process(graph)
    new_graph = data_process.weight(beta, negative_attr)

    model = SCTF(new_graph, src, dsts)
    paths = model.run()
    tree = generating_tree(paths)
    result = tree_evaluate(graph,tree,paths)
    return result


if __name__ == '__main__':
    with open('../utils/topo1_20231123_101349.json') as json_file:
        data = json.load(json_file)

    graph = json_graph.node_link_graph(data)
    print(graph.edges(data=True))
    src = 's1'
    stds = ['s9','s10']

    model = SCTF(graph,src,stds)
    paths = model.run()
    print(paths)






