#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 5/16/22 8:12 PM 
    @Description    :
        
===========================================
"""
from environment.gen_scene.common_sampler import *
from environment.gen_scene.gen_map_util import *


def create_corridor_map(configs):
    grid_resolution = configs["grid_res"]

    width = np.array(configs["corridor_width"][0]
                     + np.random.random_sample() * (configs["corridor_width"][1] - configs["corridor_width"][0])
                     ) + 2.0 * configs["thickness"]
    length = np.array(configs["corridor_height"][0]
                      + np.random.random_sample() * (configs["corridor_height"][1] - configs["corridor_height"][0])
                      ) + 2.0 * configs["thickness"]

    occupancy_map = np.zeros((int(width / grid_resolution), int(length / grid_resolution)), dtype=bool)

    # outer wall
    occupancy_map[:, [0, -1]] = True
    occupancy_map[[0, -1], :] = True

    return occupancy_map, [start_goal_sampler, static_obs_sampler, distant_two_points_sampler]
