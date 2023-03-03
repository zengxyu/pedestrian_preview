#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 5/18/22 9:46 PM 
    @Description    :
        
===========================================
"""

from environment.gen_scene.common_sampler import *


def create_open_map(configs):
    grid_resolution = configs["grid_res"]

    width = np.array(configs["width_range"][0]
                     + np.random.random_sample() * (configs["width_range"][1] - configs["width_range"][0])
                     ) + 2.0 * configs["thickness"]
    length = np.array(configs["height_range"][0]
                      + np.random.random_sample() * (configs["height_range"][1] - configs["height_range"][0])
                      ) + 2.0 * configs["thickness"]

    occupancy_map = np.zeros((int(width / grid_resolution), int(length / grid_resolution)), dtype=bool)

    return occupancy_map
