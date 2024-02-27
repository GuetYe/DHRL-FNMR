# -*- coding: utf-8 -*-

# r,d,t,h分别表示剩余带宽、时延、丢包率、跳数的权重
# r1, d1, t1, h1 = 0.25, 0.25, 0.25, 0.25
r1, d1, t1, h1 = 1, 1, 1, 0

m = 5  # 从dj算法的结果中挑选m个最短路径用于计算最优单播路径
k = 3   # 用于计算组播的best k path

# GA setting
aCrossRate = 0.9     # 交叉概率
aMutationRate = 0.1  # 变异概率
retainRate = 0.2     # 每轮迭代时的精英保留比例
aLifeCount = 20      # 每轮迭代时的种群中个体的总数
iteration_num = 30   # 迭代轮数

MAX_CAPACITY = 30000  # Max capacity of link (kbit/s)
