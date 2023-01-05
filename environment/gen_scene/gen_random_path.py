#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 3/27/22 3:39 PM 
    @Description    : For low level training
        
===========================================
"""
import numpy as np

from utils.math_helper import compute_yaw


def generate_random_path(waypoints_num, waypoints_distance_range, waypoint_theta_range):
    """
    generate path with random waypoints,
    the distance between waypoints is equal to waypoints_distance,
    the number of waypoints is equal to waypoints_num
    :param waypoints_num:
    :param waypoint_theta_range:
    :param waypoints_distance_range:
    :return:
    """
    cur_point = np.array([0., 0.])

    paths = []
    for i in range(waypoints_num):
        cur_point = cur_point + create_random_target_point(waypoints_distance_range, waypoint_theta_range)
        paths.append(cur_point)

    start_position = paths[0]
    goal_position = paths[-1]
    start_yaw = compute_yaw(paths[0], paths[1])
    return np.array(paths), np.array(start_position), start_yaw, np.array(goal_position)


def create_random_target_point(distance_range, theta_range):
    # distance from 0.6 to 1.4, theta from 0 to 2 * np.pi
    distance = np.random.random() * (distance_range[1] - distance_range[0]) + distance_range[0]
    theta = np.random.random() * (theta_range[1] - theta_range[0]) + theta_range[0]
    x = distance * np.cos(theta)
    y = distance * np.sin(theta)
    return np.array([x, y])
