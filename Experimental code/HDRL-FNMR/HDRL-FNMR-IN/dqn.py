# -*- coding: utf-8 -*-
# @File    : dqn.py
# @Date    : 2022-11-25
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-

import torch
import torch.nn as nn

import math
from pathlib import Path
import numpy as np
from collections import namedtuple
import pickle

from model import IntrinsicModel, ExtrinsicModel
from parse_config import parse_config
from replymemory import ExperienceReplayMemory, PrioritizedReplayMemory
from transition import Transition
from constant import DTYPE, DEVICE


BatchVars = namedtuple('BatchVars',
                       ['state_batch', 'action_batch', 'reward_batch',
                        'non_final_next_states', 'non_flag', 'non_final_mask',
                        'indices', 'weights'])


class DQNAgent:
    def __init__(self,
                 config_file: Path = None,
                 model_config_file: Path = None,
                 train_or_eval: str = 'train',
                 beta_frames: int = 1e4,
                 ) -> None:

        config = parse_config(config_file)

        self.lr = config.lr
        self.batch_size = config.batch_size
        self.tau = config.tau
        self.action_num = config.action_nums
        self.gamma = config.reward_decay
        
        if config.name == 'intrinsic':
            Model = IntrinsicModel
        elif config.name == 'extrinsic':
            Model = ExtrinsicModel
            
        self.policy_net = Model(model_config_file, config.state_channels, config.action_nums)
        self.target_net = Model(model_config_file, config.state_channels, config.action_nums)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2500, ], gamma=0.1)
        self.loss_func = nn.SmoothL1Loss()
        
        if config.use_per:
            self.memory_pool = PrioritizedReplayMemory(size=int(config.memory_capacity), beta_frames=beta_frames)
        else:
            self.memory_pool = ExperienceReplayMemory(capacity=config.memory_capacity)

        self.update_target_count = 0
        self.update_target_net_frequency = config.update_refquency

        self.use_eps_decay = config.use_decay_e_greedy
        self.epsilon_start, self.epsilon_final, self.epsilon_decay = config.start, config.end, config.decay
        self.epsilon = 0.1
        
        self.set_model_mode(train_or_eval)

        self.loss = None

    def sample(self, s, epoch):
        """choose action according to e_greedy

        Args:
            s : state
            epoch (int): epoch of train

        Returns:
            int: action
        """
        if self.use_eps_decay:
            epsilon = self.decay_eps_greedy(epoch)
        else:
            epsilon = self.epsilon

        temp = np.random.rand()
        if temp >= epsilon:  # greedy policy
            action_value = self.policy_net.forward(s.to(DEVICE))
            # shape [1, actions_num]; max -> (values, indices)
            act_node = torch.max(action_value, 1)[1]  # 最大值的索引
        else:  # random policy
            act_node = torch.from_numpy(np.random.choice(
                np.array(range(self.action_num)), size=1)).to(DEVICE)
        # array([indice], dtype)
        act_node = act_node[0]
        return act_node

    def sample_max(self, s):
        """choose action of the max action-state value

        Args:
            s : state of agent

        Returns:
            int: action index
        """
        action_value = self.policy_net.forward(s.to(DEVICE))
        # shape [1, actions_num]; max -> (values, indices)
        act_node = torch.max(action_value, 1)[1]  # 最大值的索引
        return act_node

    def store(self, s, a, r, s_):
        self.memory_pool.push(s, a, r, s_)

    def _get_batch_vars(self, epoch):
        transitions, indices, weights = self.memory_pool.sample(
            self.batch_size, epoch)
        batch = Transition(*zip(*transitions))
        # 竖着放一起 (B, Hin)
        state_batch = torch.cat(batch.state).to(DEVICE)
        action_batch = torch.cat(batch.action).to(DEVICE)
        reward_batch = torch.cat(batch.reward).to(DEVICE)

        # 计算非最终状态的掩码并连接批处理元素(最终状态将是模拟结束后的状态）
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=DEVICE, dtype=torch.bool)

        try:
            non_final_next_states = torch.cat(
                [s for s in batch.next_state if s is not None]).to(DEVICE)
            non_flag = False
        except Exception as e:
            non_final_next_states = None
            non_flag = True

        batch_vars = BatchVars(state_batch, action_batch, reward_batch,
                               non_final_next_states, non_flag, non_final_mask,
                               indices, weights)
        return batch_vars

    def _calculate_loss(self, batch_vars):
        """
            计算损失
        :param batch_vars: state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask, indices, weight
        :return: loss
        """

        batch_state = batch_vars.state_batch
        batch_action = batch_vars.action_batch
        batch_reward = batch_vars.reward_batch
        non_final_next_states = batch_vars.non_final_next_states
        non_flag = batch_vars.non_flag
        non_final_mask = batch_vars.non_final_mask
        indices = batch_vars.indices
        weights = batch_vars.weights

        # 从policy net中根据状态s，获得执行action的values
        _out = self.policy_net(batch_state.to(DEVICE))
        state_action_values = torch.gather(_out, 1, batch_action.type(torch.int64))

        with torch.no_grad():
            # 如果non_final_next_states是全None，那么max_next_state_values就全是0
            max_next_state_values = torch.zeros(self.batch_size, device=DEVICE).unsqueeze(1)
            if not non_flag:
                # 从target net中根据非最终状态，获得相应的value值
                max_next_action = self.target_net(non_final_next_states).max(dim=1)[1].view(-1, 1)
                max_next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, max_next_action)
            # 计算期望的Q值
            expected_state_action_values = (max_next_state_values * self.gamma) + batch_reward

        if isinstance(self.memory_pool, PrioritizedReplayMemory):
            self.memory_pool.update_priorities(indices, (
                state_action_values - expected_state_action_values).detach().squeeze(1).abs().cpu().numpy().tolist())
            loss = self.loss_func(state_action_values.to(DTYPE).to(DEVICE), expected_state_action_values.to(DTYPE).to(DEVICE)) * weights.unsqueeze(1).to(DTYPE)
            loss = loss.mean()
        else:
            loss = self.loss_func(state_action_values.to(DTYPE).to(DEVICE), expected_state_action_values.to(DTYPE).to(DEVICE))

        return loss

    def learn(self, epoch):
        if len(self.memory_pool) < self.batch_size:
            return

        batch_vars = self._get_batch_vars(epoch)
        loss = self._calculate_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        self._update_target_model()
        self.loss = loss

    def decay_eps_greedy(self, epoch):
        """decay epsilon greedy

        Args:
            epoch (int): train epoch

        Returns:
            int: eps
        """
        eps = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(
            -1. * epoch / self.epsilon_decay)

        return eps

    def _update_target_model(self):
        self.update_target_count += 1
        if self.update_target_count % self.update_target_net_frequency == 0:
            self._soft_update()

    def _soft_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def save_weight(self, name: str):
        path = Path(f'./saved_agents/{name}')
        path.mkdir(exist_ok=True, parents=True)

        torch.save(self.policy_net.state_dict(),
                   path / f'policy_net-{name}.pt')
        torch.save(self.target_net.state_dict(),
                   path / f'target_net-{name}.pt')
        torch.save(self.optimizer.state_dict(),
                   path / f'optim-{name}.pt')

    def load_weight(self, fname_model: Path):
        if Path.exists(fname_model):
            self.policy_net.load_state_dict(
                torch.load(fname_model, map_location=DEVICE))
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            raise FileNotFoundError

    def set_model_mode(self, train_or_eval='train'):
        """    set model mode

        Args:
            train_or_eval (str): mode of model. Defaults to 'train'.
        """
        assert train_or_eval in ['train', 'eval']
        if train_or_eval == 'train':
            self.policy_net.train().to(DEVICE)
            self.target_net.train().to(DEVICE)

        elif train_or_eval == 'eval':
            self.policy_net.eval().to(DEVICE)
            self.target_net.eval().to(DEVICE)

    def save_replay(self):
        pickle.dump(self.memory_pool, open(
            f'./saved_agents/exp_replay_agent-{self.experiment_time}.pkl', 'wb'))

    def load_replay(self, fname: Path):
        if Path.exists(fname):
            self.memory_pool = pickle.load(open(fname, 'rb'))
