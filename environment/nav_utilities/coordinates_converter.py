from math import cos, sin
from typing import List

import numpy as np

from utils.math_helper import cartesian_2_polar, polar_2_cartesian


def cvt_to_bu(om_point, grid_resolution):
    """
    convert occupancy map coordinates to bullet coordinates
    :param om_point:
    :param grid_resolution:
    :return: position in bullet coordinates
    """

    return np.array(om_point) * grid_resolution


def cvt_to_om(bu_point, grid_resolution):
    """
    convert bullet coordinates to occupancy map coordinates
    :param bu_point:
    :param grid_resolution:
    :return: position in occupancy map
    """
    res = np.array(bu_point) / grid_resolution
    res = res.astype(np.int)
    return res


def transfer_world_to_local(center_x, center_y):
    t = np.array([-center_x, -center_y])
    return t


def transfer_local_to_world(center_x, center_y):
    t = np.array([center_x, center_y])
    return t


def transform_world_to_robot(robot_x, robot_y, robot_yaw):
    """
    rotation and translation from world coordinates to robot coordinate frame
    :param robot_x:
    :param robot_y:
    :param robot_yaw:
    :return:
    """
    alpha = robot_yaw - np.pi / 2
    r = np.array([[cos(alpha), sin(alpha)],
                  [- sin(alpha), cos(alpha)]])
    t = np.array([-robot_x, -robot_y])
    return r, t


def transform_world_to_target(target_x, target_y, target_yaw):
    """
    rotation and translation from world coordinates to target coordinates.
    Target coordinate frame takes the path direction as the Y axis
    :param target_x:
    :param target_y:
    :param target_yaw:
    :return:
    """
    alpha = target_yaw - np.pi / 2
    r = np.array([[cos(alpha), sin(alpha)],
                  [- sin(alpha), cos(alpha)]])
    t = np.array([-target_x, -target_y])
    return r, t


def transform_robot_to_image(image_half_w):
    r = np.array([[0, -1], [1, 0]])
    t = np.array([image_half_w, image_half_w])
    return r, t


def transform_point_to_image_coord(point):
    r = np.array([[0, 1], [1, 0]])
    cvt_p = r @ np.array(point)
    cvt_p = np.array([int(cvt_p[0]), int(cvt_p[1])])
    return cvt_p


def cvt_polar_positions_to_reference(polar_positions, reference_yaw):
    polar_positions[:, 0] = (polar_positions[:, 0] - reference_yaw + np.pi / 2) % (2 * np.pi)
    return polar_positions


def cvt_positions_to_reference(positions, reference_position, reference_yaw=np.pi / 2):
    """
    如果是想转成相对于机器人的坐标，但角度仍然是世界坐标系的角度，那么reference_yaw = np.pi / 2
    :param positions:
    :param reference_position:
    :param reference_yaw:
    :return:
    """
    x, y = reference_position[0], reference_position[1]
    r, t = transform_world_to_target(x, y, reference_yaw)
    cvt_hit_positions = []
    for position in positions:
        cvt_hit_position = r @ (position + t)
        cvt_hit_positions.append(cvt_hit_position)
    return np.array(cvt_hit_positions)


def transform_local_to_world(local_position, reference_position, reference_yaw):
    reference_yaw = reference_yaw - np.pi / 2
    r = np.array([[cos(reference_yaw), -sin(reference_yaw)], [sin(reference_yaw), cos(reference_yaw)]])
    cvt_position = r @ local_position + reference_position
    return cvt_position


def cvt_positions(positions, cvt_func):
    polar_positions = np.array([cvt_func(position) for position in positions])
    return polar_positions


def cvt_positions_to_polar(positions):
    polar_positions = np.array([cartesian_2_polar(position) for position in positions])
    return polar_positions


def cvt_polar_to_cartesian(positions):
    # TODO 这个需要加速
    cartesian_positions = np.array([polar_2_cartesian(position) for position in positions])
    return cartesian_positions


def cvt_vectorized_polar_to_cartesian(positions):
    thetas, rs = positions[:, 0], positions[:, 1]
    x = rs * np.cos(thetas)
    y = rs * np.sin(thetas)

    return np.array([x, y]).T
