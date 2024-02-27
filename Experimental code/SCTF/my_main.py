#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2023/11/23 11:49
@File:my_main.py
@Desc:所有功能的实现
"""
"""
实现的功能：
Normalization.py
1.对拓扑图进行归一化
2.将拓扑中的数据根据收益加权为一个权重，并写入到图中
SCTF.py
3.利用这个数据执行算法得到相应的路径
result_display.py
5.将路径合成组播树拓扑图
result_evaluate.py
6.通过拓扑图进行评估
"""
import os
import random
import pickle
import numpy as np
import networkx as nx
import copy
import matplotlib.pyplot as plt
from algorithms.SCTF import SCTF
from algorithms.MAQ import env_display

class Data_process:
    def __init__(self,graph):
        self.graph = graph
        self.attributes = set()
        self.attr_value_dict = {}
        self.new_graph = copy.deepcopy(self.graph)
        self.epsilon = 1e-6

        for u,v,data in self.graph.edges(data=True):
            self.attributes.update(data.keys())

    # 图属性归一化
    def normalization(self):
        for attr in self.attributes:
            attr_value_list = list(nx.get_edge_attributes(self.graph,attr).values())
            attr_value_max = max(attr_value_list)
            attr_value_min = min(attr_value_list)
            for u,v in self.graph.edges:
                attr_value = self.graph[u][v][attr]
                self.new_graph[u][v][attr] = (attr_value-attr_value_min)/(attr_value_max-attr_value_min+self.epsilon)
        return self.new_graph

    # 生成加权的属性
    def weight(self,beta,negative_attr):
        if 'weight' in self.attributes:
            print("attribute ‘weight’ has been exiting")
        else:
            self.new_graph = self.normalization()  # 防止未归一化进行情况下进行加权
            attr_list = sorted(self.attributes)  # 注意这是加权的顺序
            for u,v in self.graph.edges:
                weight = 0.
                for attr_value,beta_value,neg_flag in zip(attr_list,beta,negative_attr):
                    weight += self.new_graph[u][v][attr_value]*(neg_flag==0) * beta_value \
                              + (1-self.new_graph[u][v][attr_value])*(neg_flag==1)* beta_value
                # print('权重:',weight)
                self.new_graph[u][v]['weight'] = weight
        return self.new_graph

# 从路径中生成树
def generating_tree(paths):
    new_graph = nx.DiGraph()
    link_set = set()
    for path in paths.values():
        for i in range(len(path)):
            if i>0 and path[i-1]!=path[i] and (path[i-1],path[i]) not in link_set \
                    and (path[i],path[i-1]) not in link_set:
                new_graph.add_edge(path[i-1],path[i])
                link_set.update((path[i-1],path[i]))
    return new_graph

# 绘制结果图
def graph_display(graph,tree,src,stds,weight='bw'):
    pos = nx.spring_layout(graph)
    edge_weights = nx.get_edge_attributes(graph,weight)
    nx.draw(graph,pos,
            with_labels=True,
            node_size=700,
            node_color="skyblue",
            font_size=10,
            font_color="black",
            font_weight='bold'
            )
    # 显示权重
    nx.draw_networkx_edge_labels(graph,pos,edge_labels={(u,v):f'{weight:.2f}' for (u,v),weight in edge_weights.items()})
    # 高亮显示源点(红色)
    nx.draw_networkx_nodes(graph,pos,nodelist=src,node_color='red',node_size=700)
    nx.draw_networkx_nodes(graph,pos,nodelist=stds,node_color='yellow',node_size=700)
    nx.draw_networkx_edges(graph,pos,edgelist=tree.edges,edge_color='blue',width=2.0)
    plt.show()

# 组播树评估
def tree_evaluate(graph,tree,paths):
    bottleneck_bw_list = []
    for node_list in paths.values():
        bw0 = float('inf')
        for i,node in enumerate(node_list):
            if i>0 and node_list[i-1] != node:
                bottleneck_bw = min(bw0,graph[node_list[i-1]][node]['bw'])
                bw0 = bottleneck_bw
        bottleneck_bw_list.append(bw0)

    delay_cumsum = 0  # 组播树的时延累计求和
    loss_cumsum = 0  # 组播树的丢包求和
    for u,v in tree.edges:
        # print(u,v)
        if u != v:
            delay_cumsum += graph[u][v]['delay']
            loss_cumsum += graph[u][v]['loss']

    bottleneck_bw_mean = np.mean(bottleneck_bw_list)
    tree_len = len(tree.edges)
    link_delay_mean = delay_cumsum/tree_len
    link_loss_mean = loss_cumsum/tree_len
    return bottleneck_bw_mean,link_delay_mean,link_loss_mean,tree_len


if __name__ == '__main__':
    directory_path = 'dataset/topo and data new/node14_data'
    ''' 10节点拓扑 '''
    # 从节点3，发往节点7、8、10的组播
    ''' 14节点拓扑 '''
    # 从节点12，发往节点2、4、11的组播
    ''' 21节点拓扑 '''
    # 表示从节点3，发往节点7、8、13的组播
    src = 12
    stds = [8,3]
    beta = [1, 0, 0]
    negative_attr = [1,0,0]
    file_list = os.listdir(directory_path)
    illegal_document = {'plot_info.pkl', '0-2023-11-20-15-41-12.pkl',
                        '0-2023-11-20-15-41-22.pkl', '0-2023-11-20-15-41-32.pkl'}
    file_set = set(file_list)
    sample_size = 120
    # sample_file = random.sample(file_list-illegal_document,)
    test_file_index = 18

    # 循环处理文件
    for index,file_name in enumerate(file_list):
        # if index == test_file_index:  # 用于选择文件,后面再进行优化
        # 组合完整文件路径
        file_path = os.path.join(directory_path,file_name)
        # graph = read_pkl(file_path)  # 读取topo数据
        with open(file_path,'rb') as pkl_file:
            graph = pickle.load(pkl_file)
        print('-->',graph[2][4]['bw'],graph[4][11]['bw'])
        data_process = Data_process(graph)
        new_graph = data_process.weight(beta,negative_attr)
        env_display(new_graph,[src],stds,weight='bw')
        model = SCTF(new_graph,src,stds)  # 运用算法生成路径
        paths = model.run()
        # print('生成的路径\n',paths)
        tree = generating_tree(paths)  # 通过路径生成树
        # print(tree.edges)
        # print(stds)
        graph_display(new_graph,tree,[src],stds,weight='weight')
        print(new_graph.edges(data=True))
        result = tree_evaluate(graph,tree,paths)   # 评价树的相应的指标,返回（瓶颈路径的平均值，链路平均时延，链路平均丢包率，链路长度）
        print('瓶颈路径的剩余带宽平均值，链路平均时延，链路平均丢包率，链路长度\n',result)






















