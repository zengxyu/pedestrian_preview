# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : navigation_icra 
    @Author  : Xiangyu Zeng
    @Date    : 8/22/22 5:05 PM 
    @Description    :
        
===========================================
"""
import math
from typing import List

import numpy as np
from matplotlib import pyplot as plt


def compute_pairs(split_indexes, coordinates_length):
    split_indexes = np.sort(split_indexes)

    # 先移出相近的元素，
    split_indexes_filtered = remove_close_index(split_indexes)

    # compute index pairs
    indexes_pairs = construct_index_pairs(split_indexes_filtered)

    return indexes_pairs


def remove_close_index(split_indexes):
    # 先移出相近的元素，
    cursor = split_indexes[0]
    split_indexes_filtered = []
    for index in split_indexes:
        if len(split_indexes_filtered) == 0:
            split_indexes_filtered.append(index)
            cursor = index
        elif index <= cursor + 1:
            cursor = index
        else:
            split_indexes_filtered.append(index)
            cursor = index
    return split_indexes_filtered


def construct_index_pairs(indexes):
    #
    indexes_start = np.array(indexes) + 1
    indexes_end = np.roll(indexes, -1)
    indexes_pairs = np.array([indexes_start, indexes_end]).T
    # n * 2
    return indexes_pairs


def compute_angle(coordinates_hits, coordinates_hits_prev, coordinates_hits_next):
    # coordinates_hits_prev = np.roll(coordinates_hits, -2, axis=0)
    # coordinates_hits_next = np.roll(coordinates_hits, 2, axis=0)
    # n * 2
    vec_to_prev = coordinates_hits - coordinates_hits_prev
    vec_to_next = coordinates_hits - coordinates_hits_next
    norm = np.linalg.norm(vec_to_prev, axis=1) * np.linalg.norm(vec_to_next, axis=1)
    # 0 - 1是锐角 -1 ~ 0是钝角
    cos_theta = np.sum(vec_to_prev * vec_to_next, axis=1) / norm

    return cos_theta


class CoordinatesGroup:
    span = 0
    coordinates = []

    def __init__(self, coordinates):
        center = coordinates[int(len(coordinates) / 2)]
        span = np.linalg.norm(coordinates[0] - center) + np.linalg.norm(coordinates[-1] - center)
        self.span = span
        self.coordinates = coordinates
        self.num_points = len(coordinates)
        self.mass_center = np.mean(coordinates, axis=0)
        self.span_center = (coordinates[0] + coordinates[-1]) / 2
        self.center_point = center if np.linalg.norm(coordinates[0] - coordinates[-1]) < 0.2 else self.span_center

    def compute_descriptor(self):
        # 取五个点作为descriptor
        # 起点 终点 中心点
        # corners, center = compute_oriented_bbox(self.coordinates)
        # descriptor = np.concatenate([corners, center[np.newaxis, :]], axis=0)
        # descriptor = descriptor.flatten()
        # 点和点之间的距离
        # 点和点之间的斜率
        # todo three points
        start = self.coordinates[0]
        end = self.coordinates[-1]
        center = self.coordinates[int(self.num_points / 2)]
        descriptor = np.array([start, center, end]).flatten()
        return descriptor


def group_coordinates(coordinates, index_pairs) -> List[CoordinatesGroup]:
    # group coordinates acc to index_pairs
    coordinates_groups = []
    for index_pair in index_pairs:
        start_ind, end_ind = index_pair
        if start_ind < end_ind:
            coords = coordinates[start_ind:end_ind]
        elif start_ind == end_ind:
            coords = coordinates[start_ind:]
        else:
            coords = np.concatenate([coordinates[start_ind:], coordinates[:end_ind]])

        # 创建一个group
        coordinates_group = CoordinatesGroup(coords)
        coordinates_groups.append(coordinates_group)
    return coordinates_groups


def filtered_hit_coordinates(coordinates_list, hits_list):
    coordinates_hits_list = []
    for coordinates, hits in zip(coordinates_list, hits_list):
        coordinates_hits = coordinates[hits == 1]
        coordinates_hits_list.append(coordinates_hits)
    return coordinates_hits_list


# def sample_points_uniformly(coordinates_groups: List[CoordinatesGroup]):
#     for group_i, coordinates_group in enumerate(coordinates_groups):
#         coordinates = coordinates_group.coordinates
#         sampled_points = [coordinates[0]]
#         i = 1
#         while i < coordinates_group.num_points:
#             distance = np.linalg.norm(sampled_points[-1] - coordinates[i])
#             if distance > 0.02:
#                 sampled_points.append(coordinates[i])
#             i += 1
#         coordinates_group2 = CoordinatesGroup(np.array(sampled_points))
#         coordinates_groups[group_i] = coordinates_group2


def smooth_lidar(points, interval, epoch=5):
    new_points = np.zeros_like(points)
    for k in range(epoch):
        new_points = np.zeros_like(points)
        path_len = len(points)
        for i in range(0, path_len):
            new_points[i] = np.mean(points[max(i - interval, 0):min(i + interval, len(points) - 1)], axis=0)
        points = new_points
    return new_points


def compute_dist_pairs(coordinates_hits):
    """
    compute segment pair acc to distance
    :return:
    """
    coordinates_hits_left = np.roll(coordinates_hits, -1, axis=0)
    coordinates_hits_right = np.roll(coordinates_hits, 1, axis=0)

    # compute distance to prev points
    right_btw_distances = np.linalg.norm(coordinates_hits - coordinates_hits_right, axis=1)
    # compute distance to next points
    # left_btw_distances = np.linalg.norm(coordinates_hits - coordinates_hits_left, axis=1)
    # compute radius distance to robot for each points
    # radius_distances = np.linalg.norm(coordinates_hits, axis=1)
    # distance_thresh = np.clip(radius_distances * 0.2, 0.1, 0.2)
    dist_indexes_right = np.where(right_btw_distances > 0.1)
    # dist_indexes_left = np.where(left_btw_distances > 0.1 * radius_distances)
    starts = dist_indexes_right[0]
    ends = np.roll(dist_indexes_right[0], -1)
    assert len(starts) == len(ends)
    dist_pairs = np.array([starts, ends]).T
    # print("starts:{}; ends:{}; dist_pairs:{}".format(starts, ends, dist_pairs))
    # filter out the pairs which only contains one points inside
    dist_pairs = filter_out_small_group(dist_pairs, len(coordinates_hits))

    return dist_pairs


def filter_out_small_group(dist_pairs, length):
    """
        # filter out the pairs which only contains one points inside
    :param dist_pairs:
    :return:
    """
    thresh = 2
    new_dist_pairs = []
    for start, end in dist_pairs:
        if end - start > thresh:
            new_dist_pairs.append([start, end])
        elif end <= start and end + length - 1 - start > thresh:
            new_dist_pairs.append([start, end])
    return np.array(new_dist_pairs)


def visualize_cartesian(xs, ys, title="Plot lidar cartesian positions", c='r'):
    plt.scatter(xs, ys, lw=2, c=c)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
