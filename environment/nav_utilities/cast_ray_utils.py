#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : navigation_icra 
    @Author  : Xiangyu Zeng
    @Date    : 9/3/22 1:26 PM 
    @Description    :
        
===========================================
"""
import numpy as np


def cast_rays(coordinates_hits_list, num_rays):
    """
    # 通过射线投射
    # 90个射线
    :param coordinates_hits_list:
    :param num_rays:
    :return:
    """
    # 计算极坐标
    thetas_list = []
    rs_list = []
    for coordinates_hits in coordinates_hits_list:
        thetas, rs = compute_vectorized_polar_positions(coordinates_hits)
        thetas_list.append(thetas)
        rs_list.append(rs)

    theta_thresh = np.pi * 2 / num_rays / 2

    # 初始化90个bin [0- 2*pi]
    theta_rays = np.linspace(-np.pi, np.pi, num_rays, endpoint=False)
    distances_list = []
    xs_list = []
    ys_list = []
    cartesian_positions_list = []
    for thetas, rs in zip(thetas_list, rs_list):
        distances = compute_ray_hits(theta_rays, thetas, rs, theta_thresh)
        distances_list.append(distances)
        xs, ys = compute_vectorized_cartesian_positions(theta_rays, distances)
        cartesian_positions = np.array([xs, ys]).transpose((1, 0))
        xs_list.append(xs)
        ys_list.append(ys)
        cartesian_positions_list.append(cartesian_positions)
        # visualize_polar(theta_rays, distances)
    # 转为笛卡尔坐标系
    return cartesian_positions_list


def compute_ray_hits(theta_rays, thetas, rs, theta_thresh):
    # 计算 theta_rays 能否击中 thetas
    theta_rays = theta_rays[:, np.newaxis]
    thetas = thetas[np.newaxis, :]
    thetas_diff_matrix = abs(theta_rays - thetas)
    # 九十个角度，为每个角度找到最小值，如果最小值大于某个阈值，将其表示为没有射中
    thetas_min_arg = np.argmin(thetas_diff_matrix, axis=1)
    thetas_min = np.min(thetas_diff_matrix, axis=1)
    # 计算击中的点，算击中的点的距离
    distances = rs[thetas_min_arg]
    distances[thetas_min > theta_thresh] = 1
    # 计算 theta_rays 能否击中 thetas的最近的点的距离小于某个值
    return distances


def compute_vectorized_polar_positions(coordinates_hits):
    """
    cartesian to polar
    :param coordinates_hits:
    :return:
    """
    x, y = coordinates_hits[:, 0], coordinates_hits[:, 1]
    rs = np.sqrt(x ** 2 + y ** 2)
    thetas = np.arctan2(y, x)
    return thetas, rs


def compute_vectorized_cartesian_positions(thetas, distances):
    """
    polar to cartesian
    :param thetas:
    :param distances:
    :return:
    """
    xs = distances * np.cos(thetas)
    ys = distances * np.sin(thetas)
    return xs, ys
