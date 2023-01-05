from math import acos

import numpy as np

import numpy as np


def swap_value(a, b):
    return b, a


def compute_distance(point1, point2):
    """
    compute distance between two points
    :param point1:
    :param point2:
    :return:
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


def compute_amplitude(point1, point2, line_point1, line_point2):
    distance = compute_distance(point1, point2)
    direction = np.cross(point2 - point1, line_point2 - line_point1)
    return np.sign(direction) * distance


def compute_delta_position(theta, distance):
    """
    compute delta position by theta and distance
    :param theta:
    :param distance:
    :return:
    """
    delta_x = distance * np.cos(theta)
    delta_y = distance * np.sin(theta)
    return np.array([delta_x, delta_y])


def compute_yaw(p1, p2):
    """
    compute the yaw of a vector starting from p1, ends at p2
    :param p1: point 1
    :param p2: point 2
    :return:
    """
    y = p2[1] - p1[1]
    x = p2[0] - p1[0]
    theta = np.arctan2(y, x)
    return theta


def gaussian(x, mean=0, sigma=5, a=1):
    y = a * np.e ** (-(x - mean) ** 2 / (2 * sigma ** 2))
    return y


def linear(waypoint_dist, center_dist, amplitude):
    force = (center_dist - abs(waypoint_dist - center_dist)) / center_dist * amplitude
    return force


def compute_radian_between_two_vectors(v1, v2):
    a = np.dot(v1, v2)
    costheta = a / (np.linalg.norm(v1) * np.linalg.norm(v2))
    theta = np.arccos(costheta)
    return theta


def cartesian_2_polar(v):
    x, y = v
    r = np.sqrt(x ** 2 + y ** 2)
    # arctan2 (-pi, pi)
    theta = np.arctan2(y, x)
    return theta, r


def polar_2_cartesian(v):
    theta, distance = v
    x = distance * np.cos(theta)
    y = distance * np.sin(theta)
    return x, y


def vector_angle(v1, v2):
    rad = acos(np.prod(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return rad


def compute_rotate_radian(v1, v2):
    x1, y1 = v1
    x2, y2 = v2
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    theta = np.arctan2(det, dot)
    return theta


def clockwise_radian(v1, v2):
    x1, y1 = v1
    x2, y2 = v2
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    theta = -np.arctan2(det, dot)
    # theta = theta if theta > 0 else 2 * np.pi + theta
    return theta


def counterclockwise_radian(v1, v2):
    return -clockwise_radian(v1, v2)


def calc_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))


def gaussian2d(xx, yy, obstacle_x, obstacle_y):
    """
        2d gaussian
    :param xx:
    :param yy:
    :param obstacle_x:
    :param obstacle_y:
    :return: The input dim of points == The output dim
    """
    Z = np.exp(-0.1 * ((xx - obstacle_x) ** 2 + (yy - obstacle_y) ** 2))
    return Z


# Functions to create different potential maps
def trench(X, Y):
    offs = -3 * 10 / 4
    Z = -1 * np.exp(-(X - offs) ** 2) - 1 * np.exp(-(Y - offs) ** 2)
    Z = np.where(Z < -1, -1, Z)
    return Z


def mortars(X, Y):
    offs = -3 * 10 / 4
    Z = +3 * np.exp(-0.1 * ((Y - 8) ** 2 + (X) ** 2)) + 3 * np.exp(-0.1 * ((X - 4) ** 2 + (Y + 7) ** 2))
    return Z


if __name__ == '__main__':
    v1 = np.array([0, 1])
    v2 = np.array([1, 0])
    theta = compute_rotate_radian(v1, v2)
    # theta = cartesian_2_polar(v2)
    print(theta)
