#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2023/11/26 16:12
@File:Q_learning.py
@Desc:先实现简单的情况，对每个目标进行学习，然后
"""
import copy
import numpy as np
import os
import pickle
from my_main import Data_process

class MultcastEnv:
    def __init__(self,graph,src,dst,beta):
        """
        环境初始化
        :param graph:当前的地图
        :param src: 源节点
        :param dst: 目的节点
        """
        # 注意这里的graph中有剩余带宽，时延，丢包率等消息，归一化后的结果
        self.graph = graph
        self.src = src
        self.dst = dst   # 目的节点集列表
        self.state = copy.deepcopy(src)  # 当前状态
        self.beta = beta  # 加权值


    def step(self,action):  # 外面调用这个函数来改变当前的位置
        try:
            bw = self.graph.edges[self.state][action]['bw']
            delay = self.graph.edges[self.state][action]['delay']
            loss = self.graph.edges[self.state][action]['loss']
        except:
            cost = float('inf')
            next_state = self.state
            done = False
            return next_state,cost,done

        cost = self.beta[0] * (1/bw) + self.beta[1]*delay + self.beta[2]*loss
        next_state = action
        done = False
        if next_state == self.dst:
            done = True

        return next_state,cost,done

    def reset(self):
        self.state = self.src  # 状态回到起点

class Qlearning:
    "组播Q-learning算法"
    def __init__(self,graph,src,dsts,epsilon,alpha,gamma):
        self.graph = graph  # 地图
        self.src = src  # 源起点
        self.dsts = dsts   # 目的节点集
        self.epsilon = epsilon  # epsilon贪婪策略中的参数
        self.alpha = alpha   # 学习率
        self.gamma = gamma    # 折扣因子

        n_state = len(self.graph.nodes()) * len(self.dsts)   # 状态个数
        n_action = len(self.graph.edges())  # 动作个数
        self.n_action = n_action
        self.Q_table = np.zeros([n_state,n_action])   # 初始化Q(s,a)表格

    def take_action(self,state):  # 选取下一步的操作
        state_index_dict,_ = self.state_index()
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            print(self.Q_table[state_index_dict[state]])
            action = np.argmin(self.Q_table[state_index_dict[state]])
        return action


    # 构建状态和索引之间的映射
    def state_index(self):  # 返回状态索引
        state_index_dict = {}
        index_state_dict = {}
        index = 1
        for src in self.src:
            for dst in self.dsts:
                state_index_dict[(src,dst)] = index
                index_state_dict[index] = (src,dst)
        return state_index_dict,index_state_dict

    def update(self,s0,a0,r,s1):
        td_error = r + self.gamma * self.Q_table[s1].min() - self.Q_table[s0,a0]
        self.Q_table[s0,a0] += self.alpha * td_error


# 生成给定取值数量的状态向量
def state_gen(num_list):
    for num in num_list:
        if 'state_list' in locals():
            state_list1 = state_list.copy()
            state_list2 = np.empty((0,state_list1.shape[1]+1))
            for j in range(num+1):
                new_row = np.column_stack((state_list1,np.full(state_list1.shape[0],j)))
                state_list2 = np.vstack((state_list2,new_row))
            state_list = state_list2
        else:
            state_list = np.array(num+1).reshape(-1,1)
    return state_list


if __name__ == '__main__':
    print(np.where(np.all(state_gen([2,2])==np.array([3,0]),1))[0])
    directory_path = '../dataset/topo and data/node21_data'
    src = [3]
    stds = [7, 8, 10]
    epsilon = 0.1
    alpha = 0.2
    gamma = 1
    beta = [1, 1, 1]
    negative_attr = [1,0,0]
    file_list = os.listdir(directory_path)
    illegal_document = {'plot_info.pkl', '0-2023-11-20-15-41-12.pkl',
                        '0-2023-11-20-15-41-22.pkl', '0-2023-11-20-15-41-32.pkl'}
    file_set = set(file_list)
    sample_size = 120
    # sample_file = random.sample(file_list-illegal_document,)
    test_file_index = 60

    # 循环处理文件
    for index,file_name in enumerate(file_list):
        if index == test_file_index:  # 用于选择文件,后面再进行优化
            # 组合完整文件路径
            file_path = os.path.join(directory_path,file_name)
            # graph = read_pkl(file_path)  # 读取topo数据
            with open(file_path,'rb') as pkl_file:
                graph = pickle.load(pkl_file)
            data_process = Data_process(graph)
            new_graph = data_process.weight(beta,negative_attr)
            model = Qlearning(new_graph,src,stds,epsilon,alpha,gamma)  # 运用算法生成路径
            print(model.take_action((3,7)))






