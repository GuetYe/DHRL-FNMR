# -*- coding: utf-8 -*-
# @File    : hierarchical_dqn.py
# @Date    : 2022-11-2
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-

from pathlib import Path

import torch
import numpy as np

from dqn import DQNAgent
from constant import DTYPE


class HierarchicalDQNAgent:
    def __init__(self,
                 meta_controller_config: Path,
                 meta_controller_model_config: Path,
                 controller_config: Path,
                 controller_model_config: Path,
                 train_or_eval: str = 'train',
                 beta_frames: int = 1e4) -> None:

        self.meta_controller = DQNAgent(config_file=meta_controller_config,
                                        model_config_file=meta_controller_model_config,
                                        train_or_eval=train_or_eval,
                                        beta_frames=beta_frames)
        self.controller = DQNAgent(config_file=controller_config,
                                   model_config_file=controller_model_config,
                                   train_or_eval=train_or_eval,
                                    beta_frames=beta_frames)

        self.meta_controller_state = None
        
        self.controller_state = None
        self.controller_next_state = None
        self.curr_subgoal = None

        self.loss = [None, None]

    def _stack_controller_state(self, state, subgoal_matrix):
        """stack state and subgoal

        Args:
            state (torch.matrix): state
            subgoal_matrix (numpy.matrix): subgoal matrix
        Returns:
            matrix: concatenated matrix
        """
        if state is None: 
            return None
        
        if isinstance(subgoal_matrix, np.ndarray):
            subgoal_matrix = torch.from_numpy(subgoal_matrix).to(DTYPE)

        controller_state = torch.stack(
            [subgoal_matrix] + [m for m in state.squeeze(0)], dim=0)  # [4 or 5, 14, 14]
        return controller_state.unsqueeze(0)  # [1, 4 or 5, 14, 14]

    def save(self, name: str):
        """save neural networks weights

        Args:
            name (str): name you want.
        """
        self.meta_controller.save_weight('meta-' + name)
        self.controller.save_weight("control-" + name)

    def learn(self, subgoalflag: bool, epoch):
        """update Q value
        call DQNAgent.learn()
        
        Args:
            subgoalflag (bool): flag of subgoal
        """
        self.controller.learn(epoch)
        
        self.meta_controller_learn(subgoalflag, epoch)
    
    def controller_learn(self):
        """update controller Q value
        call DQNAgent.learn()
        """
        self.controller.learn()
    
    def meta_controller_learn(self, subgoalflag, epoch):
        """update meta controller Q value
        call DQNAgent.learn()
        """
        if subgoalflag and self.meta_controller_state is None:
            self.meta_controller.learn(epoch)

    def store(self, state, action, reward, next_state, subgoal, subgoal_matrix, meta_reward, subgoal_flag):
        """store transitions to memory pool
        call DQNAgent.store()

        Args:
            state (matrix): state
            action (int): action
            reward (float): reward  
            next_state (matrix): next state 
            subgoal (int): subgoal
            subgoal_matrix (matrix): subgoal matrix
            meta_reward (float): reward of subgoal
            subgoal_flag (bool): subgoal flag
        """
        intrinsic_state = self.controller_state
        if next_state is None:
            intrinsic_next_state = None
        else:
            intrinsic_next_state = self._stack_controller_state(
                next_state, subgoal_matrix)
        self.controller_next_state = intrinsic_next_state
        
        self.controller.store(intrinsic_state, action, reward,
                              intrinsic_next_state)

        if subgoal_flag:
            self.meta_controller.store(state, subgoal, meta_reward, next_state)
            self.meta_controller_state = None
            
    def meta_controller_store(self, state, subgoal, reward, next_state):
        """meta controller store

        Args:
            state (matrix): state
            action (int): action
            reward (float): reward  
            next_state (matrix): next state 
        """
        self.meta_controller.store(state, subgoal, reward, next_state)
        self.meta_controller_state = None

    def meta_controller_sample(self, state, epoch):
        """meta controller sample

        Args:
            state (matrix): state
            epoch (int): train epoch

        Returns:
            curr subgoal: index of subgoal
        """
        if self.meta_controller_state is None:
            self.meta_controller_state = state
            self.curr_subgoal = self.meta_controller.sample(state, epoch)
        return self.curr_subgoal
    
    def meta_controller_sample_max(self, state):
        """meta controller sample max Q action

        Args:
            state (matrix): state
            epoch (int): train epoch

        Returns:
            curr subgoal: index of subgoal
        """
        if self.meta_controller_state is None:
            self.meta_controller_state = state
            self.curr_subgoal = self.meta_controller.sample_max(state)
        return self.curr_subgoal

    def controller_sample(self, controller_state, subgoal_matrix, epoch):
        """sample action

        Args:
            state (matrix): state
            epoch (epoch): epoch of train
            subgoal_matrix (matrix): subgoal matrix
        Returns:
            int: index of action
        """
        controller_state = self._stack_controller_state(
            controller_state, subgoal_matrix)

        self.controller_state = controller_state

        action = self.controller.sample(controller_state, epoch)
        return action
    
    def controller_sample_max(self, controller_state, subgoal_matrix):
        """sample max Q action

        Args:
            state (matrix): state
            epoch (epoch): epoch of train
            subgoal_matrix (matrix): subgoal matrix
        Returns:
            int: index of action
        """
        controller_state = self._stack_controller_state(
            controller_state, subgoal_matrix)

        self.controller_state = controller_state

        action = self.controller.sample_max(controller_state)
        return action

    def update_extrinsic_and_intrinsic_loss(self):
        """update meta controller loss and controller loss.

        Returns:
            tuple: pair of loss
        """
        self.loss = (self.meta_controller.loss, self.controller.loss)
        
    def load_weight(self, controller_policy_weight_path,
                    meta_policy_weight_path):
        if isinstance(controller_policy_weight_path, str):
            controller_policy_weight_path = Path(controller_policy_weight_path)
        if isinstance(meta_policy_weight_path, str):
            meta_policy_weight_path = Path(meta_policy_weight_path)
            
        self.controller.load_weight(controller_policy_weight_path)
        self.meta_controller.load_weight(meta_policy_weight_path)
