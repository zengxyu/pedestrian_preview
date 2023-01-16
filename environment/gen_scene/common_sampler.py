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


def static_obs_sampler(**kwargs):
    door_map = kwargs["door_map"]
    robot_om_path = kwargs["robot_om_path"]
    start_index = kwargs["start_index"]
    end_index = kwargs["end_index"]
    radius = kwargs["radius"]
    sample_from_path = kwargs["sample_from_path"]
    occupancy_map = kwargs["occupancy_map"]

    logging.debug("Sample static obstacles;")
    logging.debug("length of robot om path:{};".format(len(robot_om_path)))

    if sample_from_path:
        # if the surrounding is occupied
        [pivot_om_point, _], sample_success = sample_waypoint_from_path(door_map=door_map,
                                                                        robot_om_path=robot_om_path,
                                                                        start_index=start_index,
                                                                        end_index=end_index,
                                                                        sur_radius=radius
                                                                        )
    else:
        pivot_om_point = sample_waypoint_out_path(occupancy_map=occupancy_map, robot_om_path=robot_om_path)
        sample_success = True
    return pivot_om_point, sample_success


def sample_waypoint_out_path(occupancy_map, robot_om_path):
    path_map = np.zeros_like(occupancy_map)
    for p in robot_om_path:
        path_map[p[0], p[1]] = True

    path_map = convolve_map(path_map, window=5)
    occupancy_map_copy = np.logical_or(path_map, occupancy_map)
    indx, indy = np.where(np.invert(occupancy_map_copy))
    ind = np.random.choice(range(len(indx)))
    indx, indy = indx[ind], indy[ind]
    return indx, indy


def start_goal_sampler(**kwargs):
    occupancy_map = kwargs["occupancy_map"]
    indx, indy = np.where(np.invert(occupancy_map))
    # 0,1,2,3
    length = len(indx)
    start_index = np.random.randint(0, 5)
    end_index = np.random.randint(length - 5, length - 1)
    start = indx[start_index], indy[start_index]
    end = indx[end_index], indy[end_index]
    return [np.array(start), np.array(end)], True


corner_pairs = [[0, 2], [2, 0], [1, 3], [3, 1]]


def start_goal_sampler2(**kwargs):
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


def dynamic_obs_sampler(**kwargs):
    """

    :param occupancy_map:
    :param om_path:
    :param kwargs:
    :return:
    """
    # 动态障碍物(起点，终点)
    # 有两种采样方法，一种是对角采样， 一种是保持距离采样
    # 用哪个函数如何决定

    logging.debug("sampling dynamic obstacles;")
    # logging.info("kwargs:{}".format(kwargs))

    # [start_position, end_position], no_points_available = distant_two_points_sampler(**kwargs)
    [start_position, end_position], sample_success = opposite_two_points_sampler(**kwargs)
    return [start_position, end_position], sample_success


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


def distant_start_end_sampler(**kwargs):
    occupancy_map = kwargs["occupancy_map"]
    x_start, y_start = point_sampler(occupancy_map)
    distance = 0.3 * min(occupancy_map.shape[0], occupancy_map.shape[1])

    x_end, y_end = point_sampler(occupancy_map)
    while np.sqrt(np.square(x_end - x_start) + np.square(y_end - y_start)) < distance:
        x_end, y_end = point_sampler(occupancy_map)
    return [[x_start, y_start], [x_end, y_end]], True


def distant_two_points_sampler(**kwargs):
    """
    sample two distant points which keep distance = given distance_thresh
    :param kwargs:
    :return: [start_position, end_position],
            distance_less_than_thresh: no point sampled available
    """
    occupancy_map, door_map, robot_om_path = kwargs["occupancy_map"], kwargs["door_map"], kwargs["robot_om_path"]
    kept_distance = kwargs["kept_distance"]
    kept_distance_to_start = kwargs["kept_distance_to_start"]
    count_outer = 0
    logging.debug("Sample two distant points...")
    start_position, end_position = None, None

    distance_less_than_thresh = True
    while distance_less_than_thresh and count_outer < 50:
        logging.debug("Sample start point...")
        start_position = point_sampler(occupancy_map)
        end_position = point_sampler(occupancy_map)
        distance_less_than_thresh = compute_distance(start_position, end_position) < kept_distance

        distance_to_start_less_than_thresh = compute_distance(start_position, robot_om_path[0]) < kept_distance_to_start
        distance_to_end_less_than_thresh = compute_distance(end_position, robot_om_path[-1]) < kept_distance_to_start

        distance_less_than_thresh = distance_less_than_thresh or distance_to_start_less_than_thresh or distance_to_end_less_than_thresh
        count_outer += 1
        logging.debug("No available end point, resample start point...")
    sample_success = not distance_less_than_thresh
    return [start_position, end_position], sample_success


def opposite_two_points_sampler(**kwargs):
    """
    randomize two points on the two sides of the path, and keep distance = given kept_distance
    :param kwargs:
    :return:
    """
    occupancy_map, door_map, robot_om_path = kwargs["occupancy_map"], kwargs["door_map"], kwargs["robot_om_path"]
    start_index, end_index, radius = kwargs["start_index"], kwargs["end_index"], kwargs["radius"]
    kept_distance = kwargs["kept_distance"]
    kept_distance_to_start = kwargs["kept_distance_to_start"]

    # sample one waypoint from path
    candidate_waypoints, candidate_yaws = sample_waypoints_from_path(door_map, robot_om_path, start_index, end_index,
                                                                     radius)

    if len(candidate_waypoints) == 0:
        logging.error("Sample waypoints from path -- Failed")
        return [None, None], False

    # if the size of candidate waypoints > 0, 开始采样起点和终点
    om_path_start = robot_om_path[0]
    om_path_end = robot_om_path[-1]
    candidate_indexes = np.arange(0, len(candidate_waypoints), 1).tolist()

    while len(candidate_indexes) > 0:
        candidate_index = np.random.choice(candidate_indexes)
        candidate_indexes.remove(candidate_index)
        candidate_point = candidate_waypoints[candidate_index]
        candidate_yaw = candidate_yaws[candidate_index]
        candidate_point = np.array(candidate_point)

        [start_point, end_point], sample_two_points_success = distant_two_points_sampler_along_theta(occupancy_map,
                                                                                                     om_path_start,
                                                                                                     om_path_end,
                                                                                                     candidate_point,
                                                                                                     candidate_yaw,
                                                                                                     kept_distance,
                                                                                                     kept_distance_to_start)
        if sample_two_points_success:
            logging.debug("Sample start and goal point for dynamic obstacles -- Success; ")
            return [start_point, end_point], sample_two_points_success

    logging.error("Sample start and goal point for dynamic obstacles -- Failed")
    return [None, None], False


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


def distant_two_points_sampler_along_theta(occupancy_map: np.array, om_path_start: np.array, om_path_end: np.array,
                                           om_center: np.array, pivot_yaw: float, kept_distance: int,
                                           kept_distance_to_start: int):
    """
    sample point along theta, one on the left of the om_center, the other one on the right of the om_center
    :param om_path_start:
    :param pivot_yaw:
    :param occupancy_map: np.array
    :param om_center: [pixel]
    :param kept_distance: [pixel]
    :return:
    """
    # sample window
    radius = 17
    window_min_x = max(om_center[0] - radius, 0)
    window_max_x = min(om_center[0] + radius, occupancy_map.shape[1])
    window_min_y = max(om_center[1] - radius, 0)
    window_max_y = min(om_center[1] + radius, occupancy_map.shape[1])
    candidates_left = []
    candidates_right = []

    # add all free points into 2 candidate collections
    for x in range(window_min_x, window_max_x, 1):
        for y in range(window_min_y, window_max_y, 1):
            if occupancy_map[x][y]:
                continue

            if compute_distance([x, y], om_center) < kept_distance / 4:
                continue

            distance_to_start = compute_distance([x, y], om_path_start)
            if distance_to_start < kept_distance_to_start:
                continue

            distance_to_end = compute_distance(([x, y]), om_path_end)
            if distance_to_end < kept_distance_to_start:
                continue

            if in_line_left(om_center, pivot_yaw, [x, y]):
                candidates_left.append([x, y])
            elif in_line_right(om_center, pivot_yaw, [x, y]):
                candidates_right.append([x, y])
            else:
                pass

    logging.debug("There are {} candidates in candidates_left;".format(len(candidates_left)))
    logging.debug("There are {} candidates in candidates_right".format(len(candidates_right)))

    # pair them, check the pair with distance more than kept_distance
    np.random.shuffle(candidates_left)
    np.random.shuffle(candidates_right)

    for point_left in candidates_left:
        for point_right in candidates_right:
            distance = compute_distance(point_left, point_right)
            if distance > kept_distance:
                # Randomly switch start point or end point, randomly switch start point and end point
                start_point, end_point = point_left, point_right
                if np.random.choice([True, False]):
                    start_point, end_point = swap_value(point_left, point_right)
                return [start_point, end_point], True

    return [None, None], False


def sample_waypoint_from_path(door_map, robot_om_path, start_index, end_index, sur_radius):
    logging.debug(
        "Path length:{}; from_start:{}; to_end;{};surr_radius:{} ".format(len(robot_om_path), start_index, end_index,
                                                                          sur_radius))
    if start_index > end_index:
        logging.error("start_index > len(path) - end_index : {} > {}!".format(start_index, end_index))
        return [None, None], False

    # choose candidates that not close to door
    waypoint_indexes = np.arange(start_index, end_index, 1)
    candidate_indexes = []
    for waypoint_ind in waypoint_indexes:
        waypoint_ind = int(waypoint_ind)
        point = robot_om_path[waypoint_ind]
        if not is_door_neighbor(door_map, point, sur_radius):
            candidate_indexes.append(waypoint_ind)

    if len(candidate_indexes) != 0:
        point_index = np.random.choice(candidate_indexes)
        point = robot_om_path[point_index]
        point = np.array([int(point[0]), int(point[1])])
        # compute the path direction at this path point
        point_yaw = compute_yaw(robot_om_path[point_index - 1],
                                robot_om_path[min(point_index + 1, len(robot_om_path) - 1)])
        return [point, point_yaw], True
    return [None, None], False


def sample_waypoints_from_path(door_map, robot_om_path, start_index, end_index, sur_radius):
    logging.debug(
        "Path length:{}; from_start:{}; to_end;{};surr_radius:{} ".format(len(robot_om_path), start_index, end_index,
                                                                          sur_radius))
    if start_index > end_index:
        logging.error("start_index > len(path) - end_index : {} > {}!".format(start_index, end_index))
        return None

    # choose candidates that not close to door
    waypoint_indexes = np.arange(start_index, end_index, 2).tolist()
    candidate_points = []
    candidate_yaws = []
    random.shuffle(waypoint_indexes)
    for waypoint_ind in waypoint_indexes:
        waypoint_ind = int(waypoint_ind)
        point = robot_om_path[waypoint_ind]
        if not is_door_neighbor(door_map, point, sur_radius):
            # point
            point = robot_om_path[waypoint_ind]
            point = np.array([int(point[0]), int(point[1])])

            # compute yaw
            prev_index = max(waypoint_ind - 1, 0)
            next_index = min(waypoint_ind + 1, len(robot_om_path) - 1)
            point_yaw = compute_yaw(robot_om_path[prev_index], robot_om_path[next_index])

            candidate_points.append(point)
            candidate_yaws.append(point_yaw)

    return candidate_points, candidate_yaws
