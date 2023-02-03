#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 5/13/22 1:00 AM 
    @Description    :
        
===========================================
"""
from typing import List

import numpy as np

from environment.robots.base_obstacle import BaseObstacle, BaseObstacleGroup
from environment.robots.npc import Npc


class ObstacleCollections:
    def __init__(self, args):
        self.args = args
        self.obstacles: List[BaseObstacle] = []
        self.obstacle_ids: List[int] = []
        self.dynamic_obstacle_ids: List[int] = []
        self.static_obstacle_ids: List[int] = []

    def step(self):
        for obstacle in self.obstacles:
            obstacle.small_step()

    def add(self, obstacle_group: BaseObstacleGroup, dynamic: bool):
        self.obstacles.extend(obstacle_group.get_obstacles())
        self.obstacle_ids.extend(obstacle_group.get_obstacle_ids())
        if dynamic:
            self.dynamic_obstacle_ids.extend(obstacle_group.get_obstacle_ids())
        else:
            self.static_obstacle_ids.extend(obstacle_group.get_obstacle_ids())

    def get_obstacle_ids(self):
        return self.obstacle_ids

    def get_dynamic_ids(self):
        return self.dynamic_obstacle_ids

    def get_positions(self):
        positions = []
        for obstacle in self.obstacles:
            position = obstacle.get_cur_position()
            positions.append(position)
        return np.array(positions)

    def clear(self):
        self.obstacles: List[BaseObstacle] = []
        self.obstacle_ids: List[int] = []
