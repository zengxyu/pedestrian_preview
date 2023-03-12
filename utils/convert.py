#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : pedestrian_preview 
    @Author  : Xiangyu Zeng
    @Date    : 3/11/23 3:46 PM 
    @Description    :
        
===========================================
"""
import os
import pickle

from utils.fo_utility import *
folder_name = "sg_walls"
phase = "train"

env_folder = os.path.join(get_office_evacuation_path(), phase, "envs")
env_name_template = "env_{}"
indexes = [i for i in range(1500, 1700)]
env_paths = [os.path.join(env_folder, env_name_template.format(ind) + ".pkl") for ind in indexes]

for i, ind in enumerate(indexes):
    env_path = env_paths[i]
    occupancy_map, starts, ends = pickle.load(open(env_path, "rb"))
    pickle.dump([occupancy_map, starts, ends], open(env_path, "wb"))
    print("i:{}".format(i))

