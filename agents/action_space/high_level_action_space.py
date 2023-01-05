#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 4/4/22 11:04 PM 
    @Description    :
        
===========================================
"""
from abc import ABCMeta
from typing import Union, SupportsFloat

import numpy as np
from gym.spaces import Box


class AbstractHighLevelActionSpace(metaclass=ABCMeta):
    def __init__(self):
        pass

    def to_force(self, **kwargs):
        pass


class AbstractDiscreteHighLevelActionSpace(AbstractHighLevelActionSpace):
    def __init__(self):
        super().__init__()
        self.n = None

    def to_force(self, **kwargs):
        pass


class AbstractContinuousHighLevelActionSpace(Box, AbstractHighLevelActionSpace):
    def __init__(self, low: Union[SupportsFloat, np.ndarray], high: Union[SupportsFloat, np.ndarray]):
        super().__init__(low, high)

    def to_force(self, **kwargs):
        pass


class ContinuousVWActionSpace(AbstractContinuousHighLevelActionSpace):
    def __init__(self, **kwargs):
        self.v_range = kwargs["v"]
        self.w_range = kwargs["w"]
        low = np.array([self.v_range[0], self.w_range[0]])
        high = np.array([self.v_range[1], self.w_range[1]])
        self.scale = (high - low) / 2
        self.loc = (high + low) / 2
        super().__init__(-1 * np.ones_like(low), 1 * np.ones_like(high))

    def to_force(self, action):
        action = action * self.scale + self.loc
        return action
