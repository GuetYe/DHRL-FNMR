# -*- coding: utf-8 -*-
# @File    : constant.py
# @Date    : 2023-02-13
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
import numpy as np
import torch

EPS = np.finfo(np.float64).eps
INF = np.inf
DTYPE = torch.float32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
