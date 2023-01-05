#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 3/30/22 11:15 PM 
    @Description    :
        
===========================================
"""
import logging

from environment.nav_utilities.coordinates_converter import cvt_to_bu


def sample_bu_points(dilated_occ_map, grid_res, two_points_sampler):
    """
    sample two points : start points and end points
    :return:
    """
    logging.debug("Start sampling start points and end points")
    s_om_point, g_om_point = two_points_sampler(dilated_occ_map)
    s_bu_point = cvt_to_bu(s_om_point, grid_res)
    g_bu_point = cvt_to_bu(g_om_point, grid_res)
    logging.debug("Complete sampling start points and end points")

    return s_om_point, s_bu_point, g_om_point, g_bu_point
