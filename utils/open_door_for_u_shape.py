#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : pedestrian_preview 
    @Author  : Xiangyu Zeng
    @Date    : 3/4/23 9:11 PM 
    @Description    :
        
===========================================
"""
import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np

from environment.gen_scene.gen_map_util import make_exit_door
from environment.gen_scene.world_generator import get_world_config
from utils.config_utility import read_yaml
from utils.fo_utility import get_project_path
from utils.office_1000_generator import display_and_save


def open_door_for_u_shape():
    random_env_folder = os.path.join(get_project_path(), "data", "office_1000", "train", "random_envs")
    out_image_folder = os.path.join(get_project_path(), "data", "office_1000", "train", "random_envs_images")
    for i in range(1000, 1200):
        print("Processing {}".format(i))
        path = os.path.join(random_env_folder, "env_{}.pkl".format(i))
        out_path = os.path.join(out_image_folder, "env_{}.png".format(i))

        occupancy_map, s, e = pickle.load(open(path, "rb"))

        world_name = "office"
        grid_resolution = 0.1
        worlds_config = read_yaml(os.path.join(get_project_path(), "configs"), "worlds_config.yaml")
        world_config = get_world_config(worlds_config, world_name)
        make_exit_door(occupancy_map, world_config, grid_resolution)
        pickle.dump([occupancy_map, s, e], open(path, "wb"))
        ends_tile = np.tile(e, (len(s), 1))
        display_and_save(occupancy_map, s, ends_tile, save=True, save_path=out_path)


if __name__ == '__main__':
    open_door_for_u_shape()
