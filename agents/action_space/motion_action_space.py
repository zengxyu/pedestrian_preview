#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 3/25/22 11:20 PM 
    @Description    :
        
===========================================
"""
from abc import ABCMeta

import numpy as np
from gym.spaces import Box


class AbstractMotionActionSpace(metaclass=ABCMeta):
    def __init__(self):
        pass

    def get_v_w(self, **kwargs):
        pass


class MotionDiscreteActionSpace(AbstractMotionActionSpace):
    def __init__(self, **kwargs):
        super().__init__()
        v = kwargs["v"]
        w = kwargs["w"]
        v_n = kwargs["v_n"]
        w_n = kwargs["w_n"]
        self.v_list = np.linspace(v[0], v[1], v_n)
        self.w_list = np.linspace(w[0], w[1], w_n)
        self.n = len(self.v_list) * len(self.w_list)
        self.actions = [[v, w] for v in self.v_list for w in self.w_list]

    def get_v_w(self, action_index):
        return self.actions[action_index]


class MotionContinuousActionSpace(Box, AbstractMotionActionSpace):
    def __init__(self, **kwargs):
        self.v_range = kwargs["v"]
        self.w_range = kwargs["w"]
        low = np.array([self.v_range[0], self.w_range[0]])
        high = np.array([self.v_range[1], self.w_range[1]])
        self.scale = (high - low) / 2
        self.loc = (high + low) / 2
        super().__init__(-1 * np.ones_like(low), 1 * np.ones_like(high))

    def get_v_w(self, action):
        action = action * self.scale + self.loc
        return action
