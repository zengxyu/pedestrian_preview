#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 5/2/22 9:31 PM
    @Description    :

===========================================
"""
import math


def get_line_length(p1, p2):
    '''计算边长'''
    length = math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2)  # pow  次方
    length = math.sqrt(length)
    return length


def get_triangle_area(p1, p2, p3):
    '''计算三角形面积   海伦公式'''
    p1p2 = get_line_length(p1, p2)
    p2p3 = get_line_length(p2, p3)
    p3p1 = get_line_length(p3, p1)
    s = (p1p2 + p2p3 + p3p1) / 2
    area = s * (s - p1p2) * (s - p2p3) * (s - p3p1)  # 海伦公式
    if area < 0:
        print("area:", area)
        area = 0
    area = math.sqrt(area)
    return area


def get_polygon_area(points):
    # 计算多边形面积
    area = 0
    if (len(points) < 3):
        raise Exception("error")

    p1 = points[0]
    for i in range(1, len(points) - 1):
        p2 = points[i]
        p3 = points[i + 1]

        # 计算向量
        vecp1p2 = [p2[0] - p1[0], p2[1] - p1[1]]
        vecp2p3 = [p3[0] - p2[0], p3[1] - p2[1]]

        # 判断顺时针还是逆时针，顺时针面积为正，逆时针面积为负
        vecMult = vecp1p2[0] * vecp2p3[1] - vecp1p2[1] * vecp2p3[0]  # 判断正负方向比较有意思
        sign = 0
        if (vecMult > 0):
            sign = 1
        elif (vecMult < 0):
            sign = -1

        tri_area = get_triangle_area(p1, p2, p3) * sign
        area += tri_area
    return abs(area)
