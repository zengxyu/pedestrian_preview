#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : navigation_icra 
    @Author  : Xiangyu Zeng
    @Date    : 9/8/22 12:32 PM 
    @Description    :
        
===========================================
"""
import numpy as np
from matplotlib import pyplot as plt

from environment.nav_utilities.cast_ray_utils import compute_vectorized_polar_positions, \
    compute_vectorized_cartesian_positions
from environment.nav_utilities.coordinates_converter import cvt_vectorized_polar_to_cartesian
from environment.nav_utilities.visualize_utils import visualize_mass_center, visualize_motion_center


def generate_tag_descriptor(coordinates_hits_list, num_rays, theta_thresh, distance_thresh, visualize, save,
                            folder):
    """
    射线投掷
    找到reference_coordinates上射线投掷中的点，计算该点的距离，以射线投掷的角度为该点角度，as the virtual centers of the virtual group
    Assign coordinates at two time steps to the closest virtual group within specified distance
    compute the group centroids of each virtual group

    :param coordinates_hits_list:
    :param num_rays:
    :param theta_thresh:
    :param distance_thresh:
    :param visualize:
    :param save:
    :param folder:
    :return:
    group_centroids: tag_descriptor
    xs_s_t_groups, ys_s_t_groups, virtual_centers: for drawing
    """
    # 转为polar坐标系, convert the cartesian coordinates into polar coordinates
    thetas_list, rs_list = convert_cartesian_to_polar(coordinates_hits_list)

    # compute virtual centers, number: num_rays
    virtual_centers = compute_virtual_center(thetas_list, rs_list, theta_thresh, num_rays)

    # assign coordinates to groups, which have virtual centers respectively, the number is num_rays
    xs_s_t_groups, ys_s_t_groups = group(thetas_list, rs_list, virtual_centers, distance_thresh)

    # compute group centroid
    group_centroids = compute_group_centroids(xs_s_t_groups, ys_s_t_groups, virtual_centers)

    cartesian_virtual_centers = cvt_vectorized_polar_to_cartesian(virtual_centers)

    return np.array(group_centroids), xs_s_t_groups, ys_s_t_groups, cartesian_virtual_centers


def convert_cartesian_to_polar(coordinates_hits_list):
    """
    convert cartesian coordinates to polar coordinates
    :param coordinates_hits_list:
    :return:
    """
    thetas_list = []
    rs_list = []
    for coordinates_hits in coordinates_hits_list:
        thetas, rs = compute_vectorized_polar_positions(coordinates_hits)
        thetas_list.append(thetas)
        rs_list.append(rs)
    return thetas_list, rs_list


def compute_group_centroids(xs_s_t_groups, ys_s_t_groups, centers):
    """
    compute group centroids
    :param xs_s_t_groups:
    :param ys_s_t_groups:
    :param centers:
    :return:
    """
    S = len(xs_s_t_groups)
    T = len(xs_s_t_groups[0])
    # -distance_thresh ~ distance_thresh -> 0 ~ 10
    masses_s_t = []
    for s in range(S):
        masses_t = []
        for t in range(T):
            xs = xs_s_t_groups[s][t]
            ys = ys_s_t_groups[s][t]
            mass_x = np.mean(xs) if len(xs) > 0 else centers[s][1]
            mass_y = np.mean(ys) if len(ys) > 0 else centers[s][1]
            masses_t.append([mass_x, mass_y])
        masses_s_t.append(masses_t)
    return np.array(masses_s_t)


def compute_virtual_center(thetas_list, rs_list, theta_thresh, num_rays):
    """
    compute virtual center
    :param theta_rays:
    :param thetas_list:
    :param rs_list:
    :param theta_thresh:
    :return:
    """
    # divide the coordinates into num_rays groups
    theta_rays = np.linspace(-np.pi, np.pi, num_rays, endpoint=False)

    virtual_centers = []
    for theta_ray in theta_rays:
        thetas_ref = thetas_list[-1]
        rs_ref = rs_list[-1]

        # 找到最近的点所在的索引 a
        thetas_diff_ref = abs(theta_ray - thetas_ref)
        thetas_min_arg = np.argmin(thetas_diff_ref)

        # 找到最近的角度
        center_theta = thetas_ref[thetas_min_arg]

        if abs(center_theta - theta_ray) > theta_thresh:
            center_distance = 1
        else:
            center_distance = rs_ref[thetas_min_arg]

        center_theta = theta_ray
        virtual_centers.append([center_theta, center_distance])

    return np.array(virtual_centers)


def group(thetas_list, rs_list, virtual_centers, distance_thresh):
    """
    assign coordinates_list to groups,
    :param thetas_list: coordinates_list's theta list
    :param rs_list: coordinates_list's distance list
    :param virtual_centers: groups' virtual center
    :param distance_thresh:
    :return:
    """
    xs_s_t_groups = []
    ys_s_t_groups = []
    for i, center in enumerate(virtual_centers):
        center_theta, center_distance = center

        xs_t_groups = []
        ys_t_groups = []

        # 分组
        for thetas_t, distances_t in zip(thetas_list, rs_list):
            # 找到角度范围的点
            # indexes = abs(theta_ray - thetas_t) < theta_thresh
            # 应该用xy坐标算
            xs_t, ys_t = compute_vectorized_cartesian_positions(thetas_t, distances_t)
            center_x, center_y = compute_vectorized_cartesian_positions(center_theta, center_distance)
            x_diff = abs(center_x - xs_t)
            y_diff = abs(center_y - ys_t)

            indexes = (x_diff < distance_thresh) & (y_diff < distance_thresh)

            if np.sum(indexes) > 0:
                # 将这些点的角度和距离提取出来
                xs_group = xs_t[indexes]
                ys_group = ys_t[indexes]
                xs_t_groups.append(xs_group)
                ys_t_groups.append(ys_group)
                #
            else:
                xs_t_groups.append([])
                ys_t_groups.append([])

        xs_s_t_groups.append(xs_t_groups)
        ys_s_t_groups.append(ys_t_groups)
    return xs_s_t_groups, ys_s_t_groups
