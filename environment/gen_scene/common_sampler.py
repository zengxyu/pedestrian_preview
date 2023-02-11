#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 5/16/22 8:13 PM 
    @Description    :
        
===========================================
"""
import logging
import random

import numpy as np

from environment.gen_scene.gen_map_util import is_door_neighbor, convolve_map
from utils.math_helper import compute_distance, compute_yaw, swap_value

corner_pairs = [[0, 2], [2, 0], [1, 3], [3, 1]]


def sg_corner_sampler(**kwargs):
    # 0 1
    # 3 2
    occupancy_map = kwargs["occupancy_map"]
    w, h = occupancy_map.shape

    margin = kwargs["margin"]
    window_w = 5

    def sample_from_corner(corner_index):
        """compute the range for each corner"""
        if corner_index == 0:
            x_low, x_high, y_low, y_high = margin, margin + window_w, margin, margin + window_w
        elif corner_index == 1:
            x_low, x_high, y_low, y_high = margin, margin + window_w, h - window_w - margin - 1, h - window_w - 1
        elif corner_index == 2:
            x_low, x_high, y_low, y_high = w - window_w - margin - 1, w - window_w - 1, h - window_w - margin - 1, h - window_w - 1
        else:
            x_low, x_high, y_low, y_high = w - window_w - margin - 1, w - window_w - 1, margin, margin + window_w

        [x, y], sample_success = sample_from_range(x_low, x_high, y_low, y_high)
        return np.array([x, y]), sample_success

    def sample_from_range(x_low, x_high, y_low, y_high):
        """
        sample point from given range
        :param x_low:
        :param x_high:
        :param y_low:
        :param y_high:
        :return: x, y
                sample_success
        """
        x = np.random.randint(x_low, x_high)
        y = np.random.randint(y_low, y_high)
        count = 0
        while occupancy_map[x][y] and count < 10:
            x = np.random.randint(x_low, x_high)
            y = np.random.randint(y_low, y_high)
            count += 1
        return [x, y], count < 10

    sample_success, start_point, goal_point = False, None, None
    count = 0
    while not sample_success and count < 50:
        opposite_corners = corner_pairs[np.random.randint(0, len(corner_pairs))]
        start_point, sample_success1 = sample_from_corner(opposite_corners[0])
        goal_point, sample_success2 = sample_from_corner(opposite_corners[1])
        count += 1
        sample_success = sample_success1 and sample_success2
    return [start_point, goal_point], sample_success


def point_sampler(occupancy_map):
    """sample from free cells in occupancy map"""
    indx, indy = np.where(np.invert(occupancy_map))
    ind = np.random.choice(range(len(indx)))
    indx, indy = indx[ind], indy[ind]
    return [indx, indy]


def distant_point_sampler(occupancy_map, from_point=None, distance=100):
    """
    sample a point which keep distance from from_point with distance more than 100
    :param occupancy_map:
    :param from_point:
    :param distance:
    :return: sampled point
    """
    x, y = point_sampler(occupancy_map)
    if from_point is not None:
        x_start, y_start = from_point
        while np.sqrt(np.square(x - x_start) + np.square(y - y_start)) < distance:
            x, y = point_sampler(occupancy_map)
    return [x, y]


def sg_in_distance_sampler(**kwargs):
    """
    sample start and goal in distance
    """
    occupancy_map = kwargs["occupancy_map"]
    distance_ratio = kwargs["distance_ratio"]
    distance = distance_ratio * min(occupancy_map.shape[0], occupancy_map.shape[1])
    x_start, y_start = point_sampler(occupancy_map)
    x_end, y_end = point_sampler(occupancy_map)
    counter = 0
    while np.sqrt(np.square(x_end - x_start) + np.square(y_end - y_start)) < distance and counter < 100:
        x_start, y_start = point_sampler(occupancy_map)
        x_end, y_end = point_sampler(occupancy_map)
        counter += 1
    sample_success = np.sqrt(np.square(x_end - x_start) + np.square(y_end - y_start)) > distance
    return [[x_start, y_start], [x_end, y_end]], sample_success


def in_line_left(om_center, theta, point):
    x = point[0]
    y = point[1]
    line_left = np.tan(theta) * (x - om_center[0] + 5) + om_center[1] - y < 0
    return line_left


def in_line_right(om_center, theta, point):
    x = point[0]
    y = point[1]
    line_left = np.tan(theta) * (x - om_center[0] - 5) + om_center[1] - y > 0
    return line_left
