import pickle
import numpy as np

# f = open('./node10_data/4-2023-11-20-15-42-12.pkl','rb')
# data = pickle.load(f)
# print(data)
# print(len(data))
import matplotlib.pyplot as plt
import networkx as nx

def read_pickle(pickle_path):
    """
    读取pickle并转化为graph
    :param pickle_path:
    :return:
    """
    pkl_graph = nx.read_gpickle(pickle_path)
    print(pkl_graph.edges.data())
    nx.draw(pkl_graph, with_labels=True)
    plt.show()
    return pkl_graph

# data=read_pickle('node10_data/4-2023-11-20-15-42-12.pkl')

data=read_pickle('node21_data/38-2023-11-24-18-53-52.pkl')
