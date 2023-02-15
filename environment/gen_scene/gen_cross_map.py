#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 5/16/22 8:14 PM 
    @Description    :
        
===========================================
"""
import logging

import numpy as np


def create_cross_map(configs):
    grid_resolution = 2.0 * configs["thickness"]

    hallway_width = np.array(
        configs["hallway_width"][0]
        + np.random.random_sample() * (configs["hallway_width"][1] - configs["hallway_width"][0])
    ) + 2.0 * configs["thickness"]
    short_L = np.array(
        configs["short_L"][0]
        + np.random.random_sample() * (configs["short_L"][1] - configs["short_L"][0])
    ) + 2.0 * configs["thickness"]
    long_L = np.array(
        configs["long_L"][0]
        + np.random.random_sample() * (configs["long_L"][1] - configs["long_L"][0])
    ) + 2.0 * configs["thickness"]
    hallway_width = int(hallway_width / grid_resolution)
    short_L = int(short_L / grid_resolution)
    long_L = int(long_L / grid_resolution)

    occupancy_map = np.zeros((2 * short_L + hallway_width, 2 * long_L + hallway_width), dtype=bool)

    occupancy_map[:short_L, [long_L, hallway_width + long_L]] = True
    occupancy_map[-short_L:, [long_L, hallway_width + long_L]] = True
    occupancy_map[[0, -1], long_L: hallway_width + long_L] = True

    occupancy_map[[short_L, hallway_width + short_L], :long_L] = True
    occupancy_map[[short_L, hallway_width + short_L], -long_L:] = True
    occupancy_map[short_L: hallway_width + short_L, [0, -1]] = True

    logging.debug("Cross environment : width:{}; height:{}".format(short_L, long_L))

    def robot_start_end_sampler(**kwargs):
        occupancy_map = kwargs["occupancy_map"]
        margin = kwargs["margin"]
        corner_choice = np.random.choice([0, 1, 2, 3], size=2, replace=False)
        corner_1st = corner_choice[0]
        corner_2nd = corner_choice[1]

        dist_to_corner = min(short_L - margin, long_L - margin, 10)
        # sample start position
        start_x, start_y = sample_point_from_corner(corner_1st, occupancy_map, margin, hallway_width, short_L, long_L,
                                                    dist_to_corner)
        while occupancy_map[start_x, start_y]:
            start_x, start_y = sample_point_from_corner(corner_1st, occupancy_map, margin, hallway_width, short_L,
                                                        long_L,
                                                        dist_to_corner)

        # sample goal position
        end_x, end_y = sample_point_from_corner(corner_2nd, occupancy_map, margin, hallway_width, short_L, long_L,
                                                dist_to_corner)
        while occupancy_map[end_x, end_y]:
            end_x, end_y = sample_point_from_corner(corner_2nd, occupancy_map, margin, hallway_width, short_L, long_L,
                                                    dist_to_corner)

        start = [start_x, start_y]
        end = [end_x, end_y]
        logging.debug("Robot - start:{}; end:{}".format(start, end))
        return [np.array(start), np.array(end)], True

    def pedestrian_start_end_sampler(**kwargs):
        occ_map = kwargs["occupancy_map"]
        margin = kwargs["margin"]
        margin = 5
        corner_choice = np.random.choice([0, 1, 2, 3], size=2, replace=False)
        corner_1st = corner_choice[0]
        corner_2nd = corner_choice[1]

        dist_to_corner = min(short_L - margin, long_L - margin)
        # sample start position
        start_x, start_y = sample_point_from_corner(corner_1st, occ_map, margin, hallway_width, short_L, long_L,
                                                    dist_to_corner)
        while occ_map[start_x, start_y]:
            start_x, start_y = sample_point_from_corner(corner_1st, occ_map, margin, hallway_width, short_L, long_L,
                                                        dist_to_corner)

        # sample goal position
        end_x, end_y = sample_point_from_corner(corner_2nd, occ_map, margin, hallway_width, short_L, long_L,
                                                dist_to_corner)
        while occ_map[end_x, end_y]:
            end_x, end_y = sample_point_from_corner(corner_2nd, occ_map, margin, hallway_width, short_L, long_L,
                                                    dist_to_corner)

        start = [start_x, start_y]
        end = [end_x, end_y]
        logging.debug("Pedestrian - start:{}; end:{}".format(start, end))
        return [np.array(start), np.array(end)], True

    return occupancy_map, [robot_start_end_sampler, static_obs_sampler, pedestrian_start_end_sampler]


def sample_point_from_corner(corner_choice, occ_map, margin, hallway_width, short_L, long_L, distance_to_corner):
    # top
    if corner_choice == 0:
        point_x = np.random.randint(short_L + margin, hallway_width + short_L - margin)
        point_y = np.random.randint(margin, distance_to_corner)
        while occ_map[point_x, point_y]:
            point_x = np.random.randint(short_L + margin, hallway_width + short_L - margin)
            point_y = np.random.randint(margin, distance_to_corner)
    # down
    elif corner_choice == 1:
        point_x = np.random.randint(short_L + margin, hallway_width + short_L - margin)
        point_y = np.random.randint(2 * long_L + hallway_width - distance_to_corner,
                                    2 * long_L + hallway_width - 1 - margin)
        while occ_map[point_x, point_y]:
            point_x = np.random.randint(short_L + margin, hallway_width + short_L - margin)
            point_y = np.random.randint(2 * long_L + hallway_width - distance_to_corner,
                                        2 * long_L + hallway_width - 1 - margin)
    # left
    elif corner_choice == 2:
        point_x = np.random.randint(margin, distance_to_corner)
        point_y = np.random.randint(long_L + margin, hallway_width + long_L - margin)
        while occ_map[point_x, point_y]:
            point_x = np.random.randint(margin, distance_to_corner)
            point_y = np.random.randint(long_L + margin, hallway_width + long_L - margin)
    # right
    else:
        point_x = np.random.randint(2 * short_L + hallway_width - distance_to_corner,
                                    2 * short_L + hallway_width - 1 - margin)
        point_y = np.random.randint(long_L + margin, hallway_width + long_L - margin)
        while occ_map[point_x, point_y]:
            point_x = np.random.randint(2 * short_L + hallway_width - distance_to_corner,
                                        2 * short_L + hallway_width - 1 - margin)
            point_y = np.random.randint(long_L + margin, hallway_width + long_L - margin)
    return point_x, point_y
