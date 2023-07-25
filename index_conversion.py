# -*- coding: utf-8 -*-
# @File    : index_conversion.py
# @Date    : 2022-11-28
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
import torch


def node_to_index(*node: int):
    res = []
    if len(node) == 1:
        return node[0] - 1
    else:
        for n in node:
            res.append(n - 1)

        return res


def index_to_node(*index: int):

    if len(index) == 1:
        index = index[0]
        if isinstance(index, torch.Tensor):
            return int(index) + 1
        return index + 1
    else:
        res = []
        for i in index:
            res.append(i + 1)

        return res
