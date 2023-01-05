#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : navigation_icra 
    @Author  : Xiangyu Zeng
    @Date    : 8/21/22 3:09 PM 
    @Description    :
        
===========================================
"""
import numpy as np


def compute_oriented_bbox(points):
    ca = np.cov(points, y=None, rowvar=0, bias=1)
    # print("points:", points)
    # print("ca:", ca)
    _, vec = np.linalg.eig(ca)
    tvect = np.transpose(vec)

    ar = np.dot(points, np.linalg.inv(tvect))

    # get the minimum and maximum x and y
    mina = np.min(ar, axis=0)
    maxa = np.max(ar, axis=0)
    diff = (maxa - mina) * 0.5
    # the center is just half way between the min and max xy
    center = mina + diff
    diff += 0.01
    # get the 4 corners by subtracting and adding half the bounding boxes height and width to the center
    corners = np.array([center + [-diff[0], -diff[1]],
                        center + [diff[0], -diff[1]],
                        center + [diff[0], diff[1]],
                        center + [-diff[0], diff[1]]])

    # use the the eigenvectors as a rotation matrix and
    # rotate the corners and the centerback
    corners = np.dot(corners, tvect)
    center = np.dot(center, tvect)

    return corners, center
