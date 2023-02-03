#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/6/22 5:59 PM 
    @Description    :
        
===========================================
"""
from agents.action_space.high_level_action_space import ContinuousVWActionSpace
from utils.config_utility import read_yaml

motion_action_map = {"ContinuousVWActionSpace": ContinuousVWActionSpace}


def build_action_space(parser_args):
    action_name = parser_args.running_config["action_space"]

    # get action class
    if motion_action_map.keys().__contains__(action_name):
        action_class = motion_action_map[action_name]
    else:
        raise NotImplementedError

    # get action config
    action_space_config = parser_args.action_spaces_config[action_class.__name__]
    action_space = action_class(**action_space_config)

    return action_space
