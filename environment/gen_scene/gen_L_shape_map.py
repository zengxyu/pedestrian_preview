#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 5/12/22 12:57 PM 
    @Description    :
        
===========================================
"""

import itertools
import logging
import sys

from environment.gen_scene.gen_map_util import *
from utils.math_helper import compute_distance


def create_L_map(configs):
    grid_resolution = 2.0 * configs["thickness"]

    hallway_width = np.array(
        configs["limit"][0][0]
        + np.random.random_sample() * (configs["limit"][0][1] - configs["limit"][0][0])
    )
    short_L = np.array(
        configs["limit"][1][0]
        + np.random.random_sample() * (configs["limit"][1][1] - configs["limit"][1][0])
    )
    long_L = np.array(
        configs["limit"][2][0]
        + np.random.random_sample() * (configs["limit"][2][1] - configs["limit"][2][0])
    )
    hallway_width = int(hallway_width / grid_resolution)
    short_L = int(short_L / grid_resolution)
    long_L = int(long_L / grid_resolution)

    flip_axis0 = np.random.choice([True, False])
    flip_axis1 = np.random.choice([True, False])

    occupancy_map = np.zeros((short_L + hallway_width, long_L + hallway_width), dtype=bool)

    occupancy_map[:, 0] = True
    occupancy_map[0, :] = True
    occupancy_map[hallway_width, hallway_width:] = True
    occupancy_map[-1, :hallway_width] = True
    occupancy_map[:hallway_width, -1] = True
    occupancy_map[hallway_width:, hallway_width] = True

    if flip_axis0:
        occupancy_map = np.flip(occupancy_map, axis=0)

    if flip_axis1:
        occupancy_map = np.flip(occupancy_map, axis=1)

    def two_points_sampler(occ_map, margin=5):
        start_x = np.random.randint(hallway_width + int(short_L / 2) + margin, hallway_width + short_L - margin)
        start_y = np.random.randint(margin, hallway_width - margin)
        while occ_map[start_x, start_y]:
            start_x = np.random.randint(hallway_width + int(short_L / 2) + margin, hallway_width + short_L - margin)
            start_y = np.random.randint(margin, hallway_width - margin)

        end_x = np.random.randint(margin, hallway_width - margin)
        end_y = np.random.randint(hallway_width + int(long_L / 2) + margin, hallway_width + long_L - margin)
        while occ_map[end_x, end_y]:
            end_x = np.random.randint(margin, hallway_width - margin)
            end_y = np.random.randint(hallway_width + int(long_L / 2) + margin, hallway_width + long_L - margin)

        start = [start_x, start_y]
        end = [end_x, end_y]

        if flip_axis0:
            start = [hallway_width + short_L - 1 - start[0], start[1]]
            end = [hallway_width + short_L - 1 - end[0], end[1]]

        if flip_axis1:
            start = [start[0], hallway_width + long_L - 1 - start[1]]
            end = [end[0], hallway_width + long_L - 1 - end[1]]

        return np.array(start), np.array(end)

    return occupancy_map, two_points_sampler
