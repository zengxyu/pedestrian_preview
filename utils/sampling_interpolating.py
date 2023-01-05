#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 1/26/22 8:17 PM 
    @Description    :
        
===========================================
"""

import os.path
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.math_helper import compute_distance


def points_interpolating_by_distance(start_point: np.array, end_point: np.array, delta_dist: float):
    # calculate the line distance given the start point and the end point
    distance = np.linalg.norm(start_point - end_point) + 1e-16

    # if two points are the same point, return the start point
    if distance < delta_dist:
        sampled_points = np.array([start_point])
    else:
        # the ratio that delta_dist in the distance of the whole line
        segment_ratio = delta_dist / distance

        # how many parts that the distance can be split by the delta_dist into
        segment_num = int(distance / delta_dist)

        # np.linspace(0, 10, 11) == 0 1 2 3 4 5 6 7 8 9 10, which contains 11 numbers,
        # but the line is parted into 10 segments
        partials = segment_ratio * np.linspace(0, segment_num, segment_num + 1)

        # expand the dim, to do matrix operation
        start_point = start_point[np.newaxis, :]
        end_point = end_point[np.newaxis, :]
        partials = partials[:, np.newaxis]

        # inter points = start point * (1-p) + end point * p
        sampled_points = np.multiply((1 - partials), start_point) + np.multiply(partials, end_point)

    return sampled_points


def extract_points(env_path: np.ndarray, point_gap=0.1):
    # start_time = time.time()
    points = []
    start_pos = env_path[0, :]
    for i in range(1, len(env_path)):
        end_pos = env_path[i, :]
        line_length = (
                              (start_pos[0] - end_pos[0]) ** 2 + (start_pos[1] - end_pos[1]) ** 2
                      ) ** 0.5
        factor = point_gap / line_length
        factors = [[factor]]
        while True:
            temp = factors[-1][0] + factor
            if temp > 1:
                break
            else:
                factors.append([temp])
        end_factors = np.array(factors)
        start_factors = 1 - end_factors
        ps = start_pos * start_factors + end_pos * end_factors
        points.extend(list(ps))
        start_pos = points[-1]
    return np.array(points)


def equidistant_sampling_from_path(path: np.array, delta_dist: float):
    """
    sample points from path per delta_dist
    :param path:
    :param delta_dist:
    :return:
    """
    if len(path) <= 1:
        return path
    cursor = 1
    start_point = path[0]

    sampled_path = []
    sampled_path.append(start_point)

    has_next = True
    while has_next:
        end_point = path[cursor]
        # i = 1
        # check the points with max index in this range
        # while cursor + i < len(path) and np.linalg.norm(end_point - start_point) < 2 * delta_dist:
        #     end_point = path[cursor + i]
        #     i += 1
        sampled_points = points_interpolating_by_distance(start_point, end_point, delta_dist)
        sampled_path.extend(sampled_points[1:])

        start_point = sampled_points[-1]
        cursor = cursor + 1
        has_next = cursor < len(path)

    sampled_path.append(path[-1])
    # sampled_path = np.array(sampled_path)
    # print("sampled_path:", np.array(sampled_path))
    # print("sampled_path before deleting items:\n", np.array(sampled_path))
    for i in range(len(sampled_path) - 1, -1, -1):
        if compute_distance(sampled_path[i], sampled_path[i - 1]) < delta_dist / 2:
            del sampled_path[-1]
        else:
            break
    # print("sampled_path after deleting items:\n", np.array(sampled_path))
    # sampled_path.append(path[-1])
    # start_points = np.array(sampled_path[:-1])
    # end_points = np.array(sampled_path[1:])
    # distances = np.linalg.norm(end_points - start_points, axis=1)
    # print("distances:\n", np.array(distances))

    return np.array(sampled_path)


if __name__ == '__main__':
    path = [[0.8, 1.],
            [0.8, 0.9],
            [0.7, 0.8],
            [0.71, 0.81],
            [0.72, 0.69],
            [0.7, 0.7]
            ]
    path = np.array(path)
    path_result = extract_points(path, 0.05)
    print("path_result:\n", path_result)

    start_points = path_result[:-1]
    end_points = path_result[1:]
    distances = np.linalg.norm(end_points - start_points, axis=1)
    # sample_points = points_interpolating_by_distance(point1, point2, delta_dist=1)
    print("distances:\n", distances)
