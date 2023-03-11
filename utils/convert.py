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
from utils.gen_fixed_envs import display_and_save

folder_name = "sg_walls"
phase = "train"

env_folder = os.path.join(get_office_evacuation_path(), folder_name, phase, "envs")
image_folder = os.path.join(get_office_evacuation_path(), folder_name, phase, "envs_images")

env_name_template = "env_{}"
indexes = [i for i in range(1500, 1700)]
env_paths = [os.path.join(env_folder, env_name_template.format(ind) + ".pkl") for ind in indexes]
image_paths = [os.path.join(image_folder, env_name_template.format(ind) + ".png") for ind in indexes]

for i, ind in enumerate(indexes):
    env_path = env_paths[i]
    image_path = image_paths[i]
    occupancy_map, starts, ends = pickle.load(open(env_path, "rb"))
    # pickle.dump([occupancy_map, starts, ends], open(env_path, "wb"))
    display_and_save(occupancy_map, starts, ends, save=True, save_path=image_path)

