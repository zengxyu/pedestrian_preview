#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 5/4/22 10:32 PM 
    @Description    :
        
===========================================
"""
import numpy as np


def get_dist_to_line_segment(point, line_p1, line_p2):
    foot_p = _get_foot_point(point, line_p1, line_p2)
    if ((foot_p[0] - line_p1[0]) > 0) ^ (
            (foot_p[0] - line_p2[0]) > 0
    ):  # 异或符号，符号不同是为1，,说明垂足落在直线中
        dist = np.linalg.norm((foot_p[0] - point[0], foot_p[1] - point[1]))
    else:
        dist = min(
            np.linalg.norm((line_p1[0] - point[0], line_p1[1] - point[1])),
            np.linalg.norm((line_p2[0] - point[0], line_p2[1] - point[1])),
        )
    return dist


def _get_foot_point(point, line_p1, line_p2):
    """
    计算点到线的垂足
    @point, line_p1, line_p2 : [x, y, z]
    """
    x0 = point[0]
    y0 = point[1]
    z0 = 0  # point[2]

    x1 = line_p1[0]
    y1 = line_p1[1]
    z1 = 0  # line_p1[2]

    x2 = line_p2[0]
    y2 = line_p2[1]
    z2 = 0  # line_p2[2]

    denominator = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
    if denominator == 0:
        k = 0
    else:
        k = (
                -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1) + (z1 - z0) * (z2 - z1))
                / denominator
                * 1.0
        )

    xn = k * (x2 - x1) + x1
    yn = k * (y2 - y1) + y1
    zn = 0  # k * (z2 - z1) + z1

    return xn, yn
