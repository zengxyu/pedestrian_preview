#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 5/12/22 11:29 PM 
    @Description    :
        
===========================================
"""
from abc import ABCMeta


class BaseObstacle(metaclass=ABCMeta):
    def __init__(self):
        self.cur_position = None
        self.theta = None
        self.type = None

    def create(self, **kwargs):
        pass

    def get_cur_position(self):
        pass


class BaseObstacleGroup(metaclass=ABCMeta):
    def __init__(self):
        self.obstacles = []
        self.obstacle_ids = []

    def create(self, **kwargs):
        pass

    def get_obstacles(self):
        return self.obstacles

    def get_obstacle_ids(self):
        return self.obstacle_ids
