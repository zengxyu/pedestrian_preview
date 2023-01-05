#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 1/22/22 5:00 PM 
    @Description    :
        
===========================================
"""
import os

import numpy as np
from pfrl import explorers

from config import read_yaml


def get_explorer_by_name(parser_args, name, action_space=None):
    config = read_yaml(config_dir=parser_args.agents_config_folder, config_name="explorers.yaml")
    if name == "Greedy":
        return explorers.Greedy()
    elif name == "LinearDecayEpsilonGreedy":
        return explorers.LinearDecayEpsilonGreedy(
            start_epsilon=config[name]["start_epsilon"],
            end_epsilon=config[name]["end_epsilon"],
            decay_steps=config[name]["decay_steps"],
            random_action_func=lambda: np.random.randint(action_space.n),
        )
    elif name == "AdditiveGaussian":
        return explorers.AdditiveGaussian(
            scale=config[name]["scale"], low=action_space.low, high=action_space.high
        )
    elif name == "Additive_ou":
        return explorers.AdditiveOU(mu=config[name]["mu"], theta=config[name]["theta"], sigma=config[name]["sigma"])

    raise ValueError("Cannot find explorer by name {}".format(name))
