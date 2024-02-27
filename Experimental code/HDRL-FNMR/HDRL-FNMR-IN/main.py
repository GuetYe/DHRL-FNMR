# -*- coding: utf-8 -*-
# @File    : main.py
# @Date    : 2022-12-07
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
import time
from pathlib import Path
import os
import random
from collections import Counter
import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from parse_config import parse_config
from hierarchical_dqn import HierarchicalDQNAgent
from env import Env
from save_episode_data import save_episodes_data, save_episode_subgoal_counter
from paint import paint_intrinsic_state
from transition import EpisodeMetrics


def set_seed(seed):
    """set all random seed

    Args:
        seed (int): seed
    """
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def pkl_file_path_yield(pkl_dir: Path, s = 0, n: int = 12, step: int = 1):
    """Generates a path to a saved pickle file in ascending order

    Args:
        pkl_dir (Path): pickle file path
        s (int, optional): start index. Defaults to 10.
        n (int, optional): num of pkl. Defaults to 3000.
        step (int, optional): interval. Defaults to 1.

    Yields:
        Path: path of pkl file.
    """
    a = os.listdir(pkl_dir)
    assert n < len(a), "n should small than len(a)"
    # print([x.split('-')[0] for x in a])
    b = sorted(a, key=lambda x: int(x.split('-')[0]))
    for p in b[s:n:step]:
        yield pkl_dir / p

def modify_controller_param(controller, modify_param_name, modify_param):
    if modify_param_name is not None and modify_param is not None: 
        
        if modify_param_name == 'lr':
            controller.lr = modify_param
            for param_group in controller.optimizer.param_groups:
                param_group['lr'] = modify_param
        elif modify_param_name == 'batchsize':
            controller.batch_size = modify_param
        elif modify_param_name == 'egreedy':
            controller.epsilon_decay = modify_param
        elif modify_param_name == 'gamma':
            controller.gamma = modify_param
        elif modify_param_name == 'updatefrequency':
            controller.update_target_net_frequency = modify_param

def main(name,
         modify_param_name=None, 
         modify_param=None,
         modify_controller='intrinsic',
         config_file: Path = Path('./config/train_config.yaml')):
    start_time = time.time()
    config = parse_config(config_file)
    
    set_seed(config.random_seed)

    hdqn = HierarchicalDQNAgent(meta_controller_config=Path(config.meta_controller_config),
                                meta_controller_model_config=Path(config.meta_controller_model_config),
                                controller_config=Path(config.controller_config),
                                controller_model_config=Path(config.controller_model_config),
                                train_or_eval=config.mode,
                                beta_frames=2e3)
    
    env = Env(Path(config.config_file),
              Path(config.reward_config_file),
              config.mode)
    
    if modify_controller == 'intrinsic':
        controller = hdqn.controller
    else:
        controller = hdqn.meta_controller
    modify_controller_param(controller, modify_param_name, modify_param)
        
    writer = SummaryWriter(f"./runs/{name}") if config.mode != 'eval' else None

    data_save_path = Path(config.data_save_path) / name

    if config.mode != 'eval':
        data_save_path.mkdir(exist_ok=True, parents=True)

    image_save_path = Path(config.image_save_path)
    image_save_path.mkdir(exist_ok=True)

    pkl_step = config.pkl_step
    pkl_num = config.pkl_num
    pkl_start = config.pkl_start
    pkl_cut_num = pkl_start + pkl_num * pkl_step

    start_node = config.start_node
    end_nodes = config.end_nodes

    print(f"start_node: {start_node}")
    print(f"end_nodes: {end_nodes}")

    loss_meta_step, loss_controller_step = 0, 0
    all_episode_metrics = EpisodeMetrics([], [], [], [], [], [], [], [], [])
    selected_subgoal_counter = Counter({x: 0 for x in range(1, 15)})

    for episode in range(config.episodes):
        episode_metrics = EpisodeMetrics([], [], [], [], [], [], [], [], [])
        for index, pkl_path in enumerate(
                pkl_file_path_yield(Path(config.pkl_weight_path), s=pkl_start, n=pkl_cut_num, step=pkl_step)):

            state = env.reset(source=start_node, terminations=end_nodes, pkl_file=pkl_path)
            # print(f"[{episode}] reset env")
            reward_temp, subgoal_reward_temp = 0, 0

            while True:
                # meta controller sample subgoal
                subgoal = hdqn.meta_controller_sample(state, episode)  # index of subgoal
                selected_subgoal_counter[int(subgoal) + 1] += 1
                # print("subgoal: ", int(subgoal) + 1)
                # subgoal check
                if env.subgoal_step(11):  # index 11 node 12
                    # print(f'[{episode}] legal subgoal index: {subgoal}')
                    subgoal_matrix = env.matrixes.subgoal
                    
                    while True:
                        action = hdqn.controller_sample(state, subgoal_matrix, episode) 
                        # env.add_state_queue()
                        # 'next_state', 'reward', 'state_flag', 'subgoal_flag'
                        step_return = env.step(action)
                        controller_reward, meta_reward = step_return.reward
                        
                        hdqn.store(state, action, controller_reward, step_return.next_state, 
                                subgoal, subgoal_matrix, meta_reward, step_return.subgoal_flag)
                        hdqn.learn(step_return.subgoal_flag, episode)
                        
                        # if step_return.state_flag == 'road_back':
                        #     # env rollback
                        #     state = env.env_rollback()
                        # elif step_return.state_flag == 'knock_wall':
                        #     state = env.env_rollback(step=1)
                        # else:
                        
                        state = step_return.next_state

                        if config.paint_intrinsic_state and step_return.state_flag != "dead" and step_return.state_flag != 'road_back':
                            paint_intrinsic_state(hdqn.controller_state, name, episode, index, env.step_counter)

                        reward_temp += step_return.reward[0]
                        subgoal_reward_temp += step_return.reward[1] if step_return.reward[1] is not None else 0

                        hdqn.update_extrinsic_and_intrinsic_loss()
                        if hdqn.loss[0] is not None:
                            writer.add_scalar("Optim/meta Loss", hdqn.loss[0], loss_meta_step)
                            loss_meta_step += 1
                        if hdqn.loss[1] is not None:
                            writer.add_scalar("Optim/controller Loss", hdqn.loss[1], loss_controller_step)
                            loss_controller_step += 1
                        
                        # break controller loop with the legal subgoal to selection a new subgoal
                        if step_return.state_flag == 'part_done' or step_return.state_flag == "all_done": # or step_return.state_flag == "road_back":  # env.rollback_counter > env.rollback_threshold 
                            break
                        
                    if step_return.state_flag == "all_done":
                        bw, delay, loss, length, alley = env.get_route_params()
                        tree_reward = env.get_tree_reward()
                        episode_metrics.bw.append(bw)
                        episode_metrics.delay.append(delay)
                        episode_metrics.loss.append(loss)
                        episode_metrics.length.append(length)
                        episode_metrics.reward.append(reward_temp)
                        episode_metrics.subgoal_reward.append(subgoal_reward_temp)
                        episode_metrics.tree_reward.append(tree_reward)
                        episode_metrics.steps.append(env.step_counter)
                        episode_metrics.alley.append(alley)

                        print(f"[{episode}][{index}] reward: {tree_reward}")
                        print(f"[{episode}][{index}] tree_nodes: {env.tree_nodes}")
                        print(f"[{episode}][{index}] route_list: {env.env_graph.route.edges}")
                        # print(f"[{episode}][{index}] branches: {env.branches}")
                        print(f"[{episode}][{index}] step_num: {env.step_counter}")
                        print("=======================================================")
                        break
                    
                    # if create a ring, restart  
                    if step_return.next_state is None:  # step_return.state_flag == "road_back"
                        # print("RING RESTART")
                        break
                else:
                    hdqn.meta_controller_store(state=state, subgoal=subgoal, reward=env.reward.subgoal_dead, next_state=None)
                    hdqn.meta_controller_learn(True, episode)
                    
        if episode_metrics.reward:
            writer.add_scalar('Episode/reward', np.array(episode_metrics.reward).mean(), episode)
            writer.add_scalar('Episode/subgoal_reward', np.array(episode_metrics.subgoal_reward).mean(), episode)
            writer.add_scalar('Episode/tree_reward', np.array(episode_metrics.tree_reward).mean(), episode)
            writer.add_scalar('Episode/steps', np.array(episode_metrics.steps).mean(), episode)
            writer.add_scalar('Episode/alley', np.array(episode_metrics.alley).mean(), episode)
            writer.add_scalar('Episode/bw', np.array(episode_metrics.bw).mean(), episode)
            writer.add_scalar('Episode/delay', np.array(episode_metrics.delay).mean(), episode)
            writer.add_scalar('Episode/loss', np.array(episode_metrics.loss).mean(), episode)
            writer.add_scalar('Episode/length', np.array(episode_metrics.length).mean(), episode)
            
            all_episode_metrics.bw.append(np.array(episode_metrics.bw).mean())
            all_episode_metrics.delay.append(np.array(episode_metrics.delay).mean())
            all_episode_metrics.loss.append(np.array(episode_metrics.loss).mean())
            all_episode_metrics.length.append(np.array(episode_metrics.length).mean())
            all_episode_metrics.reward.append(np.array(episode_metrics.reward).mean())
            all_episode_metrics.tree_reward.append(np.array(episode_metrics.tree_reward).mean())
            all_episode_metrics.subgoal_reward.append(np.array(episode_metrics.subgoal_reward).mean())
            all_episode_metrics.steps.append(np.array(episode_metrics.steps).mean())
            all_episode_metrics.alley.append(np.array(episode_metrics.alley).mean())
            
            # if episode_metrics.tree_reward:
            #     if np.array(episode_metrics.tree_reward).mean() >= max_reward_weight:
            #         hdqn.save(name)
            #         print('saved CNN weights')
                    
            #     max_reward_weight = max(max_reward_weight, np.array(episode_metrics.tree_reward).mean())

        writer.add_scalar("meta_lr", hdqn.meta_controller.optimizer.param_groups[0]['lr'], episode)
        writer.add_scalar("controller_lr", hdqn.controller.optimizer.param_groups[0]['lr'], episode)
        
    hdqn.save(name)
    save_episodes_data(data_save_path, all_episode_metrics)
    save_episode_subgoal_counter(data_save_path, selected_subgoal_counter)

    print(
        f'train over, cost time {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}')


def train_with_param_list(param_name, modify_controller):
    lr_list = [1e-3, 1e-4, 1e-5, 1e-6]
    batch_size_list = [8, 16, 32, 64, 128]
    gamma_list = [0.3, 0.5, 0.7, 0.9]
    e_greedy_list = [100, 500, 1000, 2000]
    update_steps_list = [1, 10, 100, 1000]
    # final, step, knock, subgoal, back
    
    param_list_dict = {
        'lr': lr_list,
        'batchsize': batch_size_list,
        'egreedy': e_greedy_list,
        'gamma': gamma_list,
        'updatefrequency': update_steps_list,
    }
    train_date = time.strftime("%Y%m%d%H%M%S", time.localtime())
    
    if param_name is None:
        name = f"[{train_date}]"
        if args.manualname:
            name += '_' + f'{args.manualname}' 
        if args.manualparam:
            name += '_' + f'{args.manualparam}'
        print(name)
        main(name=name)
        
    else:
        param_list = param_list_dict[param_name]
        print(f"=================={param_name}==================")
        print(f"=================={param_list}==================")
        for param in param_list:
            name = f"[{train_date}]_{param_name}_{param}"
            main(name=name, modify_param_name=param_name, modify_param=param, modify_controller=modify_controller)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train with different parameters")
    
    parser.add_argument('--param', '-p', default=None,
                        help="which param to train",
                        choices=['lr', 'batchsize', 'egreedy', 'gamma', 'updatefrequency'])
    parser.add_argument('--modify_controller', '-c', default='intrinsic', choices=['intrinsic', 'meta'])
    parser.add_argument('--manualname', default="rewardslist")
    parser.add_argument('--manualparam', default="[1.0, 0.1, -0.5, -0.5]")
    args = parser.parse_args()
    
    # main()
    train_with_param_list(args.param, args.modify_controller)
