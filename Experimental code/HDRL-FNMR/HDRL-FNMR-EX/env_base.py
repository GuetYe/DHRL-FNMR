# -*- coding: utf-8 -*-
# @File    : env_base.py
# @Date    : 2022-12-21
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
from networkx import Graph

class EnvGraph:
    def __init__(self, graph: Graph, route: Graph, branch: Graph, path: list, nodes, edges, num_of_nodes, max_degree, num_of_edges) -> None:
        self.graph = graph
        self.route = route
        self.branch = branch
        self.path = path  # path of subgoal
        self.nodes = nodes
        self.edges = edges
        self.num_of_nodes = num_of_nodes
        self.max_degree = max_degree
        self.num_of_edges = num_of_edges


class NodesPair:
    def __init__(self, source, terminations, terminations_c, arrived_termination) -> None:
        self.source = source
        self.terminations = terminations
        self.terminations_c = terminations_c
        self.arrived_termination = arrived_termination


class StateSymbol:
    def __init__(self, source, termination, tree, branch, blank, subgoal) -> None:
        self.source = source
        self.termination = termination
        self.tree = tree
        self.branch = branch
        self.blank = blank
        self.subgoal = subgoal


class StateFlag:
    def __init__(self, alldone, compelte_part, go_on, knock_wall, road_back) -> None:
        self.alldone = alldone
        self.compelte_part = compelte_part
        self.go_on = go_on
        self.knock_wall = knock_wall
        self.road_back = road_back


class Matrix:
    def __init__(self, adj, route, branch, bw, delay, loss, subgoal) -> None:
        self.adj = adj
        self.route = route
        self.branch = branch
        self.bw = bw
        self.delay = delay
        self.loss = loss
        self.subgoal = subgoal


class NormalMatrix:
    def __init__(self, bw, delay, loss) -> None:
        self.bw = bw
        self.delay = delay
        self.loss = loss


class MaxMinMetrics:
    def __init__(self, bw, delay, loss) -> None:
        self.bw = bw
        self.delay = delay
        self.loss = loss


class MaxMinValue:
    def __init__(self, max, min) -> None:
        self.max = max
        self.min = min


class StepReturn:
    def __init__(self, next_state, reward, state_flag, subgoal_flag) -> None:
        self.next_state = next_state
        self.reward = reward
        self.state_flag = state_flag
        self.subgoal_flag = subgoal_flag
