# -*- coding: utf-8 -*-


import random
import numpy as np
import copy
import pandas as pd
import time
import networkx as nx


class GetPath(object):
    def __init__(self, graph):
        self.graph = graph
        self.all_path_list = self.get_all_paths(graph)
        # print(self.all_path_list)

    def get_multicast_path(self, sw_list, k=10, algorithm=0):
        import setting
        k = setting.k
        graph = self.graph
        #  输入sw_list,如[1,4,6,3],即1-->4, 1-->6, 1-->3的一个组播任务
        if algorithm == 0:
            # 第一步，计算每个单播任务的前k个最优路径(最多保留k个，有些路径可能不足k个)。
            path_value_dic = {}
            time1_1 = time.time()
            for i in range(len(sw_list) - 1):  # sw_list：[源交换机编号、目的交换机1、目的交换机2。。。]
                k_best_path_list = self.get_k_best_path(sw_list[0], sw_list[i + 1] , k)
                path_value_dic.setdefault(sw_list[i + 1], k_best_path_list)
            time1_2 = time.time()
            # print("-----qiudanbo_all_time:", str(time1_2-time1_1))
            # print "path_value_dic", path_value_dic
            # 如L:sw_list=[1,2,5,13],此时path_value_dic的值如下:
            # {11: [[1, 4, 11], [1, 9, 4, 11], [1, 5, 4, 11], [1, 9, 5, 4, 11], [1, 5, 9, 4, 11]], 10: [[1, 4, 9, 10],
            # [1, 5, 9, 10], [1, 5, 4, 9, 10], [1, 11, 4, 9, 10], [1, 4, 5, 9, 10]], 3: [[1, 3]]}

            # 第二步，采用遗传算法找到最优的单播路径集合，构建组播路径
            # graph = self.topology_awareness.graph

            time0 = time.time()
            # print (path_value_dic)
            self.ga_multicast = GA_multicast(path_value_dic, graph)
            time1 = time.time()
            best_multicast_path = self.ga_multicast.run(setting.iteration_num)
            time2 = time.time()
            # print ("-----chushihua: ", str(time1-time0))
            # print ("-----yichuan_run_all:", str(time2-time1))

            # best_multicast_path: ((1, 5, 9), (1, 5, 2), (1, 3))

            return best_multicast_path

    def get_k_best_path(self, src_sw, dst_sw, k):   # 取前k个最佳路径
        all_paths = self.all_path_list[src_sw][dst_sw]
        # all_paths = [[1, 2, 4, 8, 12], [1, 2, 5, 8, 12], [1, 3, 6, 9, 12], [1, 3, 7, 9, 12]]
        if len(all_paths) < k+1:  # 如果可用路径小于等于k个，则不用计算，直接全部采用
            print("len(all_paths) < k+1")
            return all_paths
        else:
            path_dic = self.mfstm_path_dic_selection(all_paths)
            # path_dic: {0: 0.675, 1: 0.554, 2: 0.399, 3: 0.942}
            # print "all_paths:", all_paths
            # print "path_dic", path_dic

            sorted_path_list = sorted(path_dic.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            # sorted_path_list：[(3, 0.942), (0, 0.675), (1, 0.554), (2, 0.399)]
            # print "sorted_path_list", sorted_path_list

            k_best_path = []
            k_best_path_value = []
            for i in range(len(all_paths)):
                if i < k:
                    k_best_path_num = sorted_path_list[i][0]
                    k_best_path.append(all_paths[k_best_path_num])
                    k_best_path_value.append(sorted_path_list[i][1])
            return k_best_path

    def get_all_paths(self, graph):
        """
            return all paths among all nodes.
            eg: cycle_graph nodes:[0,1,2,3], all_paths:
             {0: {0: [0], 1: [[0, 1], [0, 3, 2, 1]], 2: [[0, 1, 2], [0, 3, 2]], 3: [[0, 3], [0, 1, 2, 3]]},
             1: {0: [[1, 0], [1, 2, 3, 0]], 1: [1], 2: [[1, 2], [1, 0, 3, 2]], 3: [[1, 0, 3], [1, 2, 3]]},
             2: {0: [[2, 1, 0], [2, 3, 0]], 1: [[2, 1], [2, 3, 0, 1]], 2: [2], 3: [[2, 3], [2, 1, 0, 3]]},
             3: {0: [[3, 0], [3, 2, 1, 0]], 1: [[3, 0, 1], [3, 2, 1]], 2: [[3, 2], [3, 0, 1, 2]], 3: [3]}}
        """
        _graph = copy.deepcopy(graph)
        paths = {}
        # print "+++++++++++++_graph.edges():", _graph.edges()

        for src in _graph.nodes():
            paths.setdefault(src, {src: [[src]]})
            for dst in _graph.nodes():
                if src == dst:
                    continue
                paths[src].setdefault(dst, [])
                # paths[src][dst] = self.get_all_simple_path(_graph, src, dst)  # 找到所有路径
                paths[src][dst] = self.get_m_shortest_simple_paths(_graph, src, dst)   # 找到setting.m个最短路径
        return paths

    def get_m_shortest_simple_paths(self, graph, src, dst):
        """use the function shortest_simple_paths() in networkx to get all paths from src to dst."""
        import setting
        generator = nx.shortest_simple_paths(graph, source=src, target=dst)
        m = setting.m
        m_shortest_simple_paths = []
        i = 0
        try:
            for path in generator:
                m_shortest_simple_paths.append(path)
                i = i + 1
                if i == m:
                    break
            return m_shortest_simple_paths
        except:
            self.logger.debug("No path between")

    def mfstm_path_dic_selection(self, all_paths, tos_ecn=1):
        path_value = {}
        i = 0
        for path in all_paths:
            path_value.setdefault(i, {"delay": 0, "bw": 0, "loss": 0, "hop":0})
            path_value[i]["delay"] = self.get_delay(path)
            path_value[i]["bw"] = self.get_rbw(path)
            path_value[i]["loss"] = self.get_loss(path)
            path_value[i]["hop"] = self.get_hop(path)
            i = i + 1
        time0 = time.time()
        # path_dic = self.get_mfstm_value(path_value, tos_ecn)  # 基于理想解法
        path_dic2 = self.get_mfstm_value2(path_value, tos_ecn)  # 基于加权求和
        # print "result1:", path_dic
        # print "result2:", path_dic2
        time1 = time.time()
        # print("qiudanbo_judge:", str(time1-time0))

        return path_dic2

    def get_hop(self, path):
        hop = len(path) - 1
        return hop

    def get_loss(self, path):
        graph = self.graph
        loss=0
        for i in range(len(path)-1):
            loss = loss+graph[path[i]][path[i+1]]["loss"]
            average_loss = loss / len(path)
        return average_loss

    def get_delay(self, path):
        graph = self.graph
        delay = 0
        for i in range(len(path)-1):
            delay = delay + graph[path[i]][path[i+1]]["delay"]
            average_delay = delay / len(path)
        return average_delay

    def get_rbw(self, path):
        import setting
        graph = self.graph
        min_rbw = setting.MAX_CAPACITY
        for i in range(len(path)-1):
            rbw = graph[path[i]][path[i+1]]["bw"]
            min_rbw = min(min_rbw, rbw)
        return min_rbw

    def get_mfstm_value2(slef, path_value, tos_ecn):
        import setting
        r = 0.0  # 剩余带宽权重
        d = 0.0  # 链路时延权重
        t = 0.0  # 链路丢包率权重
        h = 1.0  # 链路总跳数的权重

        if tos_ecn == 1:
            r, d, t, h = setting.r1, setting.d1, setting.t1, setting.h1
        elif tos_ecn == 2:
            r, d, t, h = setting.r2, setting.d2, setting.t2, setting.h2
        elif tos_ecn == 3:
            r, d, t, h = setting.r3, setting.d3, setting.t3, setting.h3

        time0 = time.time()
        delay_list = []
        rbw_list = []
        hop_list = []
        loss_list = []

        for key1 in path_value:
            delay_list.append(path_value[key1]["delay"])
            rbw_list.append(path_value[key1]["bw"])
            hop_list.append(path_value[key1]["hop"])
            loss_list.append(path_value[key1]["loss"])

        delay_dif = max(delay_list) - min(delay_list)
        rbw_dif = max(rbw_list) - min(rbw_list)
        hop_dif = max(hop_list) - min(hop_list)
        loss_dif = max(loss_list) - min(loss_list)


        if delay_dif == 0:
            delay_dif = 1
        if rbw_dif == 0:
            rbw_dif = 1
        if hop_dif == 0:
            hop_dif = 1
        if loss_dif ==0:
            loss_dif = 1

        for key1 in path_value:
            path_value[key1]["delay"] = round(float(max(delay_list) - path_value[key1]["delay"])
                                              / delay_dif, 3)
            path_value[key1]["bw"] = round(float(path_value[key1]["bw"] - min(rbw_list))
                                            / rbw_dif, 3)
            path_value[key1]["hop"] = round(float(max(hop_list) - path_value[key1]["hop"])
                                            / hop_dif, 3)
            path_value[key1]["loss"] = round(float(max(loss_list) - path_value[key1]["loss"])
                                            / loss_dif, 3)

        result = {}
        for key1 in path_value:
            value = path_value[key1]["delay"] * d + path_value[key1]["bw"] * r + path_value[key1]["hop"] * h + path_value[key1]["loss"] * t
            result.setdefault(key1, value)

        time1 = time.time()
        # print "total_time2:", str(time1 - time0)

        return result

class GA_multicast(object):
    """遗传算法类"""
    def __init__(self, path_value_dic, graph):
        import setting
        self.crossRate = setting.aCrossRate
        self.mutationRate = setting.aMutationRate
        self.lifeCount = setting.aLifeCount         # 种群中个体的数量（初始组播树的数量）
        self.geneLenght = len(path_value_dic)       # 个体的长度(组播目的节点的数量)
        # self.ga_path_selection = PathSelection()

        self.graph = graph
        self.path_value_dic = path_value_dic
        # 如L:sw_list = [1, 3, 10, 11], 此时path_value_dic的值如下:
        # {11: [[1, 4, 11], [1, 9, 4, 11], [1, 5, 4, 11], [1, 9, 5, 4, 11], [1, 5, 9, 4, 11]], 10: [[1, 4, 9, 10],
        # [1, 5, 9, 10], [1, 5, 4, 9, 10], [1, 11, 4, 9, 10], [1, 4, 5, 9, 10]], 3: [[1, 3]]}
        # print "path_value_dic:", path_value_dic

        self.initPopulation()

    # 初始化种群
    def initPopulation(self):
        self.lives = []  # 初始化的种群，包含所有个体
        for i in range(self.lifeCount):  # 循环lifeCount轮，每轮产生一个新的个体
            live = []
            for j in self.path_value_dic:  # 依次对前往每个目的节点的可选路径集中随机选一个，构成一个组播树
                live_part = random.choice(self.path_value_dic[j])
                live.append(tuple(live_part))
            self.lives.append(tuple(live))
        # print self.lives
        # self.lives 当初始化种群数量为4时，如：[((1, 3), (1, 5, 9, 10), (1, 9, 4, 11)), ((1, 3), (1, 4, 9, 10),
        # (1, 9, 5, 4, 11)), ((1, 3), (1, 4, 5, 9, 10), (1, 5, 9, 4, 11)), ((1, 3), (1, 5, 4, 9, 10), (1, 9, 5, 4, 11))]

    def run(self, n=100, tos_ecn=1):
        while n > 0:
            # print "——————————迭代————————"
            self.next(tos_ecn)
            n = n - 1
            # print n
            # print "best_multicast_path", self.lives_sorted_list[0][0]
            # print "best_value:", self.lives_sorted_list[0][1]
            # print "lives_sorted_list", self.lives_sorted_list
        return self.lives_sorted_list[0][0]

    def next(self, tos_ecn):
        import setting
        """产生下一代"""
        # self.judge()
        self.judge2(tos_ecn)
        newLives = []
        # self.lives_score_dic：{((1, 3), (1, 5, 9, 10), (1, 9, 4, 11))：0.002，...}
        self.lives_sorted_list = sorted(self.lives_score_dic.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        # self.lives_sorted_list = sorted(self.lives_score_dic.items(), key=lambda kv: (kv[1], kv[0]))  # 从小到大
        # lives_sorted_list: 如：[(((2, 3), (1, 5, 7, 10), (1, 9, 4, 11)), 28),
        #                         (((1, 3), (1, 5, 9, 10), (1, 9, 4, 11)), 40),...]

        retain_num = int(len(self.lives_sorted_list)*setting.retainRate)
        for i in range(retain_num):
            newLives.append(self.lives_sorted_list[i][0])
            # newLives: [((2, 3), (1, 5, 7, 10), (1, 9, 4, 11)),((1, 3), (1, 5, 9, 10), (1, 9, 4, 11)),... ]
        while len(newLives) < self.lifeCount:
            # print len(newLives)
            newlive = self.newChild()
            # print "newlive:", newlive
            newLives.append(newlive)
            # print "newLives", newLives
        self.lives = newLives

    def newChild(self):
        """产生新后的(选择出两个父个体，有概率进行交叉，交叉后的再有概率进行变异)"""
        parent1 = self.getOne()

        # 按概率交叉
        rate = random.random()
        if rate < self.crossRate:
            parent2 = self.getOne()
            live_new = self.cross(parent1, parent2)
        else:
            live_new = parent1

        # 按概率突变
        rate = random.random()
        if rate < self.mutationRate:
            live_new = self.mutate(live_new)

        return live_new

    def mutate(self, live):
        # live: ((1, 3), (1, 5, 9, 10), (1, 9, 4, 11))

        # path_value_dic: {11: [[1, 4, 11], [1, 9, 4, 11], [1, 5, 4, 11], [1, 9, 5, 4, 11], [1, 5, 9, 4, 11]],
        #                  10: [[1, 4, 9, 10],[1, 5, 9, 10], [1, 5, 4, 9, 10], [1, 11, 4, 9, 10], [1, 4, 5, 9, 10]],
        #                   3: [[1, 3]]}

        index = random.randint(0, self.geneLenght - 1)
        # print "path_value_dic:", self.path_value_dic
        new_value = tuple(random.choice(self.path_value_dic[live[index][-1]]))
        live_list = list(live)
        live_list[index] = new_value
        return tuple(live_list)

    def cross(self, parent1, parent2):
        """交叉（在index2中随机选择一段，替换到index1中）"""
        # parent: ((1, 3), (1, 5, 9, 10), (1, 9, 4, 11))
        index1 = random.randint(0, self.geneLenght - 1)
        index2 = random.randint(index1, self.geneLenght - 1)
        newLive = list(parent1)
        for i in range(index1, index2+1):
            newLive[i] = parent2[i]
        return tuple(newLive)

    def getOne(self):
        """基于斐波那契数列的选择方法，live被选中的概率与对应的live_score值成正比"""
        r = random.uniform(0, self.value_sum)
        for live in self.lives:
            r = r - self.lives_score_dic[live]
            if r <= 0:
                return live

    def judge(self):
        """计算每个个体的适应度"""
        self.value_sum = 0.0
        self.lives_score_dic = {}
        for live in self.lives:  # live:为一个组播树
            # live，如：((1, 3), (1, 5, 9, 10), (1, 9, 4, 11))
            live_score = self.matchFun(live)
            self.lives_score_dic.setdefault(live, live_score)
            # self.lives_score_dic：{((1, 3), (1, 5, 9, 10), (1, 9, 4, 11))：0.002，...}
            self.value_sum = self.value_sum + live_score

    def get_multicast_average_delay(self, live):
        edge_set = self.get_multicast_edge_set(live)
        graph = self.graph
        total_delay = 0
        for edge in edge_set:
            edge0, edge1 = edge.split(",")
            total_delay = total_delay + graph[int(edge0)][int(edge1)]["delay"]
        multicast_average_dealy = total_delay / len(edge_set)
        return multicast_average_dealy

    def get_multicast_average_bw(self, live):
        total_bw = 0
        for path in live:
            total_bw = total_bw + self.get_rbw(path)
        multicast_average_bw = total_bw / len(live)
        return multicast_average_bw

    def get_rbw(self, path):
        import setting
        graph = self.graph
        min_rbw = setting.MAX_CAPACITY
        for i in range(len(path)-1):
            rbw = graph[path[i]][path[i+1]]["bw"]
            min_rbw = min(min_rbw, rbw)
        return min_rbw

    def get_multicast_average_loss(self, live):
        edge_set = self.get_multicast_edge_set(live)
        graph = self.graph
        total_loss = 0
        for edge in edge_set:
            edge0, edge1 = edge.split(",")
            total_loss = total_loss + graph[int(edge0)][int(edge1)]["loss"]
        multicast_average_loss = total_loss / len(edge_set)
        return multicast_average_loss

    def get_multicast_edge_set(self, live):
        edge_set = set()
        for path in live:
            for i in range(len(path)-1):
                one_edge_tup = (str(path[i])+","+str(path[i+1]))
                edge_set.add(one_edge_tup)
        return edge_set

    def judge2(self, tos_ecn):
        """计算每个个体的适应度"""
        self.value_sum = 0.0
        self.lives_score_dic = {}
        lives_value = {}
        i = 0
        for live in self.lives:  # live:为一个组播树
            # live，如：((1, 3), (1, 5, 9, 10), (1, 9, 4, 11))
            lives_value.setdefault(i, {"delay": 0, "bw": 0, "table": 0, "hop": 0})
            lives_value[i]["delay"] = self.get_multicast_average_delay(live)
            lives_value[i]["bw"] = self.get_multicast_average_bw(live)
            lives_value[i]["loss"] = self.get_multicast_average_loss(live)
            lives_value[i]["hop"] = len(self.get_multicast_edge_set(live))
            i = i + 1
        time0 = time.time()
        GA1=GetPath(self.graph)
        # live_num_dic = GA1.get_mfstm_value(lives_value, tos_ecn)  # 基于理想解多属性决策
        live_num_dic = GA1.get_mfstm_value2(lives_value, tos_ecn)   # 基于加权求和
        time1 = time.time()
        # print "judge_time:", str(time1-time0)

        # print("self.lives", self.lives)
        #
        #
        # print ("self.lives", self.lives)
        # print ("lives_value:", lives_value)
        # print ("live_num_dic:", live_num_dic)
        # print ("len(live_num_dic):", len(live_num_dic))
        # live_num_dic: {0: 0.675, 1: 0.554, 2: 0.399, 3: 0.942}

        j = 0
        for live in self.lives:
            live_score = live_num_dic[j]
            self.lives_score_dic.setdefault(live, live_score)
            self.value_sum = self.value_sum + live_score
            j = j + 1
        # print "lives_score_dic:", self.lives_score_dic

def get_multicast_average_delay(graph, live):
    edge_set = get_multicast_edge_set(live)
    total_delay = 0
    for edge in edge_set:
        edge0, edge1 = edge.split(",")
        total_delay = total_delay + graph[int(edge0)][int(edge1)]["delay"]
    multicast_average_dealy = total_delay / len(edge_set)
    return multicast_average_dealy

def get_multicast_average_bw(graph, live):
    total_bw = 0
    for path in live:
        total_bw = total_bw + get_rbw(graph, path)
    multicast_average_bw = total_bw / len(live)
    return multicast_average_bw

def get_rbw(graph, path):
    import setting
    min_rbw = setting.MAX_CAPACITY
    for i in range(len(path)-1):
        rbw = graph[path[i]][path[i+1]]["bw"]
        min_rbw = min(min_rbw, rbw)
    return min_rbw

def get_multicast_average_loss(graph, live):
    edge_set = get_multicast_edge_set(live)
    total_loss = 0
    for edge in edge_set:
        edge0, edge1 = edge.split(",")
        total_loss = total_loss + graph[int(edge0)][int(edge1)]["loss"]
    multicast_average_loss = total_loss / len(edge_set)
    return multicast_average_loss

def get_multicast_edge_set(live):
    edge_set = set()
    for path in live:
        for i in range(len(path)-1):
            one_edge_tup = (str(path[i])+","+str(path[i+1]))
            edge_set.add(one_edge_tup)
    return edge_set


def run(file_path,src,dsts):
    sw_list = src+dsts
    pkl_graph = nx.read_gpickle(file_path)
    ga_getpath = GetPath(pkl_graph)

    best_multicast_path = ga_getpath.get_multicast_path(sw_list)  # 获取最优组播路径
    delay = get_multicast_average_delay(pkl_graph, best_multicast_path)  # 获取组播树各个边的平均时延
    bw = get_multicast_average_bw(pkl_graph, best_multicast_path)  # 获取组播树平均瓶颈带宽
    loss = get_multicast_average_loss(pkl_graph, best_multicast_path)  # 获取组播树各个边的平均丢包率
    length = len(get_multicast_edge_set(best_multicast_path))  # 获取组播树的总跳数（总的边数）

    return bw,delay,loss,length

def main():
    import setting
    ''' 10节点拓扑 '''
    # pickle_path = 'node10_data/4-2023-11-20-15-42-12.pkl'
    # sw_list = [3, 7, 8, 10]  #表示从节点3，发往节点7、8、10的组播需求

    ''' 14节点拓扑 '''
    # pickle_path = 'node14_data/11-2022-03-11-19-42-58.pkl'
    # sw_list = [12, 2, 4, 11]  # 表示从节点12，发往节点2、4、11的组播需求

    ''' 21节点拓扑 '''
    pickle_path = 'node21_data/38-2023-11-24-18-53-52.pkl'
    sw_list = [3, 16, 12, 21]  # 表示从节点3，发往节点7、8、13的组播需求

    pkl_graph = nx.read_gpickle(pickle_path)
    ga_getpath = GetPath(pkl_graph)

    best_multicast_path = ga_getpath.get_multicast_path(sw_list)    # 获取最优组播路径
    average_delay = get_multicast_average_delay(pkl_graph, best_multicast_path)   # 获取组播树各个边的平均时延
    average_bw = get_multicast_average_bw(pkl_graph, best_multicast_path)         # 获取组播树平均瓶颈带宽
    average_loss = get_multicast_average_loss(pkl_graph, best_multicast_path)     # 获取组播树各个边的平均丢包率
    total_hop = len(get_multicast_edge_set(best_multicast_path))                  # 获取组播树的总跳数（总的边数）

    print("    best_multicast_path: ", best_multicast_path)
    print("multicast_average_delay: ", average_delay)
    print("   multicast_average_bw: ", average_bw)
    print(" multicast_average_loss: ", average_loss)
    print("    multicast_total_hop: ", total_hop)


if __name__ == '__main__':
      main()