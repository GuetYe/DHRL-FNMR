# -*- coding: utf-8 -*-
# @File    : plot_figures.py
# @Date    : 2023-03-20
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


plt.style.use('seaborn-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def read_data_path(data_path="./data"):
    """
        dict = {param_name: npy_path}
    """
    data_path = Path(data_path)
    data_path_dict = {"lr": [], "nsteps": [], "batchsize": [], 'egreedy': [], "gamma": [], "update": [],
                      "rewardslist": [], "tau": []}

    for data_doc in data_path.iterdir():
        if data_doc.match('*_lr_*'):
            data_path_dict['lr'].append(data_doc)
        elif data_doc.match('*_nsteps_*'):
            data_path_dict['nsteps'].append(data_doc)
        elif data_doc.match('*_batchsize_*'):
            data_path_dict['batchsize'].append(data_doc)
        elif data_doc.match('*_egreedy_*'):
            data_path_dict['egreedy'].append(data_doc)
        elif data_doc.match('*_gamma_*'):
            data_path_dict['gamma'].append(data_doc)
        elif data_doc.match('*_updatefrequency_*'):
            data_path_dict['update'].append(data_doc)
        elif data_doc.match('*_rewardslist_*'):
            data_path_dict['rewardslist'].append(data_doc)
        elif data_doc.match('*_tau_*'):
            data_path_dict['tau'].append(data_doc)

    return data_path_dict


def get_diff_data_from_multi_file(data_path_list, param_name):
    """
        dict = {metric_name: npy_data}
    """
    data_dict = {"bw": [], "delay": [], "loss": [], 'length': [], "final_reward": [], "episode_reward": [],
                 "steps": [], "legend": []}
    for path in data_path_list:
        data_dict['legend'].append(f"{param_name} " + path.name.split('_')[-1])
        for child in path.iterdir():
            # if child.suffix == 'pkl':
            #     continue
            data = np.load(child, allow_pickle=True)
            if child.match("*bw*"):
                data_dict['bw'].append(data)
            elif child.match('*delay*'):
                data_dict['delay'].append(data)
            elif child.match("*loss*"):
                data_dict['loss'].append(data)
            elif child.match("*length*"):
                data_dict['length'].append(data)
            elif child.match("*tree_reward*"):
                data_dict['final_reward'].append(data)
            elif child.match("*episode_reward*"):
                data_dict['episode_reward'].append(data)
            elif child.match("*steps*"):
                data_dict['steps'].append(data)

    return data_dict


def get_compare_data(data_path="./data"):
    """
        dict = {param_name: metric_name: npy_data}
    """
    data_path_dict = read_data_path(data_path)
    data_npy_dict = {}
    for k in data_path_dict.keys():
        data_npy_dict[k] = get_diff_data_from_multi_file(
            data_path_dict[k], param_name=k)

    return data_npy_dict


def smooth(data_array, weight=0.9):
    # 一个类似 tensorboard smooth 功能的平滑滤波
    # https://dingguanglei.com/tensorboard-xia-smoothgong-neng-tan-jiu/
    last = data_array[0]
    smoothed = []
    for new in data_array:
        smoothed_val = last * weight + (1 - weight) * new
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_line_chart(data_list, para_name, title, x_label, y_label, legend_list):
    if data_list == []:
        return
    y_label_dict = {"bw": '平均瓶颈带宽', "delay": '平均时延', "loss": '平均丢包率', 'length': '长度',
                    "final_reward": "最终奖励值", "episode_reward": "迭代奖励值", "steps": '决策步数'}
    # color = ["#F9D923", "#EB5353", "#36AE7C", "#187498"]
    # color = ["#0848ae", "#AFF289", "#17a858", "#e8710a", "#df208c"]
    # color = ["#0848ae", "#EB5353", "#A09F1B", "#17a858", "#e8710a", "#df208c"]
    # color = ["#8ecfc9", "#ffbe7a", "#fa7f6f", "#82b0d2", '#beb8dc', '#96c37d']
    color = ['purple', 'brown', 'blue', 'orange', 'green', 'red']

    i = 0
    fig, ax = plt.subplots(1, 1)
    # if y_label in ['final_reward', 'episode_reward', 'steps']:
    #     axins = ax.inset_axes((0.4, 0.4, 0.5, 0.4))
    # else:
    axins = None
    ylim0 = float('inf')
    ylim1 = 0
    for data, legend in zip(data_list, legend_list):
        x = range(len(data))
        xlim1 = len(data)
        ylim0 = np.min(data) if np.min(data) < ylim0 else ylim0
        ylim1 = np.max(data) if np.max(data) > ylim1 else ylim1
        ax.plot(x, data, alpha=0.2, color=color[i % 6])
        ax.plot(x, smooth(data), color=color[i % 6], label=legend, alpha=1)
        # ax.plot(x, data, color=color[i % 5], label=legend, alpha=0.4)

        # if axins is not None:  # 局部放大
        #     axins.plot(x[-200:], data[-200:], alpha=0.1, color=color[i % 5])
        #     axins.plot(x[-200:], smooth(data)[-200:], color=color[i % 5])
        i += 1

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label_dict[y_label])
    ax.set_xlim(0, xlim1)
    # ax.set_ylim(ylim0, ylim1)
    # ax.legend(bbox_to_anchor=(0., 1.0), loc='lower left', ncol=len(legend_list))
    ax.legend()

    if axins is not None:
        mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
    _p = Path('./images') / para_name
    _p.mkdir(exist_ok=True)
    plt.savefig(_p / f'{title}.png', dpi=800,
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_all_line_chart_from_diff_param(data_path="./data"):
    data_npy_dict = get_compare_data(data_path)
    for param_name, metric_data in data_npy_dict.items():
        legend = metric_data.pop("legend")
        for metric_name, data in metric_data.items():
            plot_line_chart(data, param_name, metric_name,
                            '训练代数', metric_name, legend)


def plot_time_cost():
    times = ["00:33:48", "02:26:30", "07:54:27"]
    nums = [1, 12, 60]
    times2 = ["00:10:54", "01:49:24", "05:53:14"]

    def convert(times):
        ret = []
        for t in times:
            _temp = t.split(":")
            ret.append(int(_temp[0]) + float(_temp[1]) /
                       60 + float(_temp[2]) / 3600)
        return ret

    print(convert(times))

    plt.plot(nums, convert(times), '-*', label='DRL-M4MR')
    plt.plot(nums, convert(times2), '-.', label='HDRL-FNMR')
    plt.xlabel("网络状态信息/个")
    plt.ylabel("时间开销/小时")
    plt.legend()
    plt.savefig('./images/time.png', dpi=800,
                bbox_inches='tight', pad_inches=0)


def plot_subgoals_select_frequency(subgoals_path='data/[20230325010956]_pkl12/subgoals.pkl'):
    with open(subgoals_path, 'rb') as f:
        subgoals = pkl.load(f)
    x = list(range(len(subgoals)))
    y = [subgoals[i + 1] for i in range(len(subgoals))]
    
    plt.bar(x, y)
    print(x)
    plt.xticks(x, list(range(1, len(x) + 1)))
    plt.xlabel('节点')
    plt.ylabel('节点选择次数/次')
    plt.savefig('./images/selected.png', dpi=800,
                bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    plot_all_line_chart_from_diff_param()
    # plot_time_cost()
    # plot_subgoals_select_frequency()
