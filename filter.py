# -*- coding: utf-8 -*-
# @File    : filter.py
# @Date    : 2023-02-22
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来- 
import os
import networkx as nx
from pathlib import Path
import shutil

def dataset_filter(pkl_dir: Path):
    """
        Some data is wrong. For example, when the bandwidth is very low, the delay and packet loss rate are also small, 
    resulting in conflicts. Therefore, learning difficulties will be caused every time the selection
    """
    all_pkls = os.listdir(pkl_dir)
    dataset = Path('./dataset')
    dataset.mkdir(exist_ok=True)
    for p in all_pkls:
        pkl_file = pkl_dir / p
        pkl_graph = nx.read_gpickle(pkl_file)
        save_flag = True
        loss_zero = 0
        for n1, n2 in pkl_graph.edges:
            bw = pkl_graph.edges[n1, n2]['bw']
            delay = pkl_graph.edges[n1, n2]['delay']
            loss = pkl_graph.edges[n1, n2]['loss']
            
            if bw == 0 and loss == 0:
                save_flag = False
            if loss != 0:
                loss_zero += 1
                
        if save_flag and loss_zero != 0:  
            shutil.copyfile(pkl_file, dataset / pkl_file.name)


if __name__ == '__main__':
    dataset_filter(Path('./NLIs'))
