# 对比实验的数据结果
# 1. 遗传算法
# 2. Q-learning
# 3. E_SCTF
# 数据包含 ./dataset/topo and new/
# node10_data  120
# node14_data  120
# node21_data  120
# 输出结果 ./result  命名分别为node10_result,node14_result,ode21_result
# multicast_path_average_bw  组播树路径的平均瓶颈带宽(bw)
# multicast_link_average_delay 组播树链路的平均时延(delay)
# multicast_link_average_loss  组播树链路的平均丢包率(loss)
# multicast_link_length   组播树的平均链路长度(length)
# 绘图结果
# 对数据每10个进行平均绘制成柱形图
from SCTF import run as SCTF_run
from Q_laerning import run as Q_learning_run
from multicast_ga.multicast_ga_test.multicast_ga1 import run as GA_run
import os
import numpy as np

def algorithm_run(fun):
    bw,delay,loss,length = fun(file_path,src,dsts)
    bw_list.append(bw)
    delay_list.append(delay)
    loss_list.append(loss)
    length_list.append(length)
    if index-index0 == window:
        mean_bw_list.append(np.mean(bw_list[(index0+1):(index+1)]))
        mean_delay_list.append(np.mean(delay_list[(index0+1):(index+1)]))
        mean_loss_list.append(np.mean(loss_list[(index0+1):(index+1)]))
        mean_length_list.append(np.mean(length_list[(index0+1):(index+1)]))


if __name__=='__main__':

    window = 10
    name = 'node21'  # 结果在node10,node14,node21

    ''' 10节点拓扑 '''
    # 从节点3，发往节点7、8、10的组播
    if name=='node10':
        directory_path = '../dataset/topo and data new/node10_data'
        src = [3]
        dsts = [7,8,10]

    ''' 14节点拓扑 '''
    # 从节点12，发往节点2、4、11的组播
    if name=='node14':
        directory_path = '../dataset/topo and data new/node14_data'
        src = [12]
        dsts = [2,4,11]

    ''' 21节点拓扑 '''
    # 表示从节点3，发往节点7、8、13的组播
    if name=='node21':
        directory_path = '../dataset/topo and data new/node21_data'
        src = [3]
        dsts = [7,8,13]


    file_list = os.listdir(directory_path)
    num_file = len(file_list)
    algorithms = [GA_run,SCTF_run,Q_learning_run]
    total_bw = np.empty((0,num_file))
    total_delay = np.empty((0,num_file))
    total_loss = np.empty((0,num_file))
    total_length = np.empty((0,num_file))
    total_mean_bw = np.empty((0,int(num_file/window)))
    total_mean_delay = np.empty((0,int(num_file/window)))
    total_mean_loss = np.empty((0,int(num_file/window)))
    total_mean_length = np.empty((0,int(num_file/window)))
    for fun in algorithms:
        bw_list = []
        delay_list = []
        loss_list = []
        length_list = []
        mean_bw_list = []
        mean_delay_list = []
        mean_loss_list = []
        mean_length_list = []
        index0 = -1
        for index,file_name in enumerate(file_list):
            # 方法的顺序为
            # 1. 遗传算法
            # 2. Q-learning
            # 3. SCTF
            file_path = directory_path+'/'+file_name
            algorithm_run(fun)
            if index-index0 == window:
                index0 = index
                print(index)
        total_bw = np.vstack((total_bw,bw_list))
        total_delay = np.vstack((total_delay,delay_list))
        total_loss = np.vstack((total_loss,loss_list))
        total_length = np.vstack((total_length,length_list))
        total_mean_bw = np.vstack((total_mean_bw,mean_bw_list))
        total_mean_delay = np.vstack((total_mean_delay,mean_delay_list))
        total_mean_loss = np.vstack((total_mean_loss,mean_loss_list))
        total_mean_length = np.vstack((total_mean_length,mean_length_list))
    np.save('../result/'+name+'_total_bw.npy',total_bw)
    np.save('../result/'+name+'_total_delay.npy',total_delay)
    np.save('../result/'+name+'_total_loss.npy',total_loss)
    np.save('../result/'+name+'_total_length.npy',total_length)
    np.save('../result/'+name+'_total_mean_bw.npy', total_mean_bw)
    np.save('../result/'+name+'_total_mean_delay.npy', total_mean_delay)
    np.save('../result/'+name+'_total_mean_loss.npy', total_mean_loss)
    np.save('../result/'+name+'_total_mean_length.npy', total_mean_length)
    np.savetxt('../result/'+name+'_total_mean_bw.csv', total_mean_bw,delimiter=',')
    np.savetxt('../result/'+name+'_total_mean_delay.csv', total_mean_delay,delimiter=',')
    np.savetxt('../result/'+name+'_total_mean_loss.csv', total_mean_loss,delimiter=',')
    np.savetxt('../result/'+name+'_total_mean_length.csv', total_mean_length,delimiter=',')

























