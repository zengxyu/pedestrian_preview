#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : navigation_icra 
    @Author  : Xiangyu Zeng
    @Date    : 9/6/22 1:39 PM 
    @Description    :
        
===========================================
"""
from utils.icp import icp


def align_coordinates_by_icp(cartesian_positions_list):
    reference_positions = cartesian_positions_list[1]

    cartesian_positions_prev = cartesian_positions_list[0]

    cartesian_positions_next = cartesian_positions_list[2]

    _, aligned_cartesian_positions0 = icp(reference_positions, cartesian_positions_prev, verbose=False)

    _, aligned_cartesian_positions1 = icp(reference_positions, cartesian_positions_next, verbose=False)

    return [aligned_cartesian_positions0, reference_positions, aligned_cartesian_positions1]
