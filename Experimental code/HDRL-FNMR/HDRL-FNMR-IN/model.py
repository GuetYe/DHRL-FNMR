# -*- coding: utf-8 -*-
# @File    : model.py
# @Date    : 2022-11-25
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from parse_config import parse_config


class IntrinsicModel(nn.Module):
    def __init__(self, config_file=Path('./config/model_config.yaml'),
                 state_channels=1,
                 action_nums=1):
        super().__init__()

        config = parse_config(config_file)

        self.conv1 = nn.Conv2d(in_channels=state_channels,
                               out_channels=config['conv1']['out_channels'],
                               kernel_size=config['conv1']['kernel_size'],
                               stride=config['conv1']['stride'],
                               padding=config['conv1']['padding'])
        
        self.conv2 = nn.Conv2d(in_channels=config['conv2']['in_channels'],
                               out_channels=config['conv2']['out_channels'],
                               kernel_size=config['conv2']['kernel_size'],
                               stride=config['conv2']['stride'],
                               padding=config['conv2']['padding'])
        
        self.conv3 = nn.Conv2d(in_channels=config['conv3']['in_channels'],
                               out_channels=config['conv3']['out_channels'],
                               kernel_size=config['conv3']['kernel_size'],
                               stride=config['conv3']['stride'],
                               padding=config['conv3']['padding'])

        self.fc1 = nn.Linear(config['fc1']['in_channels'],
                             action_nums)
        
        self.apply(weight_init)

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x))
        x2 = F.leaky_relu(self.conv2(x1))
        x3 = F.leaky_relu(self.conv3(x2))
        
        x3 = x3.view(x.shape[0], -1)
        Qvalue = self.fc1(x3)

        return Qvalue


class ExtrinsicModel(nn.Module):
    def __init__(self, config_file=Path('./config/model_config.yaml'),
                 state_channels=1,
                 action_nums=1):
        super().__init__()

        config = parse_config(config_file)

        self.conv1 = nn.Conv2d(in_channels=state_channels,
                               out_channels=config['conv1']['out_channels'],
                               kernel_size=config['conv1']['kernel_size'],
                               stride=config['conv1']['stride'],
                               padding=config['conv1']['padding'])
        
        self.conv2 = nn.Conv2d(in_channels=config['conv2']['in_channels'],
                               out_channels=config['conv2']['out_channels'],
                               kernel_size=config['conv2']['kernel_size'],
                               stride=config['conv2']['stride'],
                               padding=config['conv2']['padding'])
        
        self.conv3 = nn.Conv2d(in_channels=config['conv3']['in_channels'],
                               out_channels=config['conv3']['out_channels'],
                               kernel_size=config['conv3']['kernel_size'],
                               stride=config['conv3']['stride'],
                               padding=config['conv3']['padding'])

        self.fc1 = nn.Linear(config['fc1']['in_channels'],
                             action_nums)
        
        self.apply(weight_init)

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x))
        x2 = F.leaky_relu(self.conv2(x1))
        x3 = F.leaky_relu(self.conv3(x2))
        
        x3 = x3.view(x.shape[0], -1)
        Qvalue = self.fc1(x3)

        return Qvalue
    
    # def __init__(self, config_file=Path('./config/model_config.yaml'),
    #              state_channels=1,
    #              action_nums=1):
    #     super().__init__()

    #     config = parse_config(config_file)

    #     self.conv1 = nn.Conv2d(in_channels=state_channels,
    #                            out_channels=config['conv1']['out_channels'],
    #                            kernel_size=config['conv1']['kernel_size'],
    #                            stride=config['conv1']['stride'],
    #                            padding=config['conv1']['padding'])

    #     self.conv2 = nn.Conv2d(in_channels=config['conv1']['out_channels'],
    #                            out_channels=config['conv2']['out_channels'],
    #                            kernel_size=config['conv2']['kernel_size'],
    #                            stride=config['conv2']['stride'],
    #                            padding=config['conv2']['padding'])

    #     self.fc1 = nn.Linear(config['fc1']['in_channels'],
    #                          config['fc1']['out_channels'])

    #     self.fc2 = nn.Linear(config['fc2']['in_channels'],
    #                          config['fc2']['out_channels'])

    #     self.adv = nn.Linear(config['adv']['in_channels'],
    #                          action_nums)

    #     self.val = nn.Linear(config['val']['in_channels'],
    #                          config['val']['out_channels'])

    #     # self.apply(weight_init)

    # def forward(self, x):
    #     x1 = F.leaky_relu(self.conv1(x))
    #     # x2 = F.leaky_relu(self.conv2(x1))
        
    #     x2 = x1.view(x.shape[0], -1)
    #     # x2 = x2.view(x.shape[0], -1)
        
    #     # x3 = F.leaky_relu(self.fc1(x2))
    #     x4 = F.leaky_relu(self.fc2(x2))

    #     advantage_function = self.adv(x4)
    #     # sate_value = self.val(x3)

    #     # return sate_value + advantage_function - advantage_function.mean()
    #     return advantage_function


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.0)
