#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : pedestrian_preview 
    @Author  : Xiangyu Zeng
    @Date    : 3/5/23 12:31 AM 
    @Description    :
        
===========================================
"""
import logging
import os
import pickle
from typing import Dict, List

import numpy as np

from environment.gen_scene.build_office_world import drop_world_walls
from environment.gen_scene.world_generator import get_world_config
from environment.nav_utilities.coordinates_converter import cvt_to_bu
from utils.fo_utility import get_project_path


def check_office1000_goal_outdoor_folder_structure():
    url = "https://pan.dm-ai.com/s/BgxDACLPNb8nQ5f"
    password = "12345678"
    office1000_parent_folder = os.path.join(get_project_path(), "data/office_1000_goal_outdoor")
    folder_structure = "\n-data\n\t-office_1000_goal_outdoor\n\t\t-train\n\t\t\t\t-geodesic_distance\n\t\t\t\t-random_envs\n\t\t\t\t-random_envs_images\n\t\t-test\n\t\t\t\t-geodesic_distance\n\t\t\t\t-random_envs\n\t\t\t\t-random_envs_images"
    warning = "Please download data from url:{}; password:{}; and put it in your project_folder/data; \nYour folder structure should be like : {}".format(
        url, password, folder_structure)
    assert os.path.exists(office1000_parent_folder), warning

    for phase in ["train", "test"]:
        geodesic_distance_folder = os.path.join(office1000_parent_folder, phase, "geodesic_distance")
        random_env_folder = os.path.join(office1000_parent_folder, phase, "random_envs")
        random_envs_images_folder = os.path.join(office1000_parent_folder, phase, "random_envs_images")

    assert os.path.exists(geodesic_distance_folder), warning
    assert os.path.exists(random_env_folder), warning
    assert os.path.exists(random_envs_images_folder), warning


def load_office1000_goal_scene(p, running_config, worlds_config, phase):
    """
    load scene from map path and trajectory path
    """

    # scene_index = np.random.randint(0, 1000)
    # logging.error("Choose scene index:{}".format(scene_index))
    scene_parent_folder = os.path.join(get_project_path(), "data", "office_1000_goal_outdoor", phase, "random_envs")
    geodesic_distance_parent_folder = os.path.join(get_project_path(), "data", "office_1000_goal_outdoor", phase,
                                                   "geodesic_distance")
    obstacle_distance_parent_folder = os.path.join(get_project_path(), "data", "office_1000_goal_outdoor", phase,
                                                   "obstacle_distance")
    potential_maps_parent_folder = os.path.join(get_project_path(), "data", "office_1000_goal_outdoor", phase,
                                                "potential_maps")
    file_names = os.listdir(geodesic_distance_parent_folder)
    file_name = np.random.choice(file_names)
    print("scene file name:{}".format(file_name))
    si = file_name.index("_") + 1
    ei = file_name.index(".")
    scene_index = int(file_name[si:ei])
    # scene_index = np.random.randint(1000, 1200)
    scene_path = os.path.join(scene_parent_folder, "env_{}.pkl".format(scene_index))

    obstacle_distance_path = os.path.join(obstacle_distance_parent_folder, "env_{}.pkl".format(scene_index))
    geodesic_distance_dict_path = os.path.join(geodesic_distance_parent_folder, "env_{}.pkl".format(scene_index))
    potential_maps_path = os.path.join(potential_maps_parent_folder, "env_{}.pkl".format(scene_index))
    # Read occupancy map, starts and ends
    occupancy_map, starts, ends = pickle.load(open(scene_path, 'rb'))
    # read obstacle distance map
    obstacle_distance_map = pickle.load(open(obstacle_distance_path, 'rb'))
    # read geodesic_distance_map
    geodesic_distance_dict_dict = pickle.load(open(geodesic_distance_dict_path, 'rb'))
    potential_map_x, potential_map_y, potential_map = pickle.load(open(potential_maps_path, 'rb'))

    world_name = running_config["world_name"]
    # get method to create specified scene map
    world_config = get_world_config(worlds_config, world_name)

    indexes = np.random.randint(0, len(starts), size=running_config["num_agents"])

    bu_starts = []
    bu_ends = []
    geodesic_distance_dict_list: List[Dict] = []
    geodesic_distance_map_list: List[np.array] = []
    for i in indexes:
        start = starts[i]
        end = ends[0]

        # 取出对应终点的测地距离dict
        geodesic_distance_dict = geodesic_distance_dict_dict[tuple(end)]

        # 将测地距离dict保存到list中
        geodesic_distance_dict_list.append(geodesic_distance_dict)

        # 将测地距离dict转为测地距离map
        geo_distance_map = np.zeros_like(occupancy_map).astype(float)
        for key in geodesic_distance_dict.keys():
            distance = geodesic_distance_dict[key]
            geo_distance_map[key] = distance
        geodesic_distance_map_list.append(geo_distance_map)
        # geodesic_distance_dict[]
        # sample start position and goal position
        bu_start = cvt_to_bu(start, running_config["grid_res"])
        bu_end = cvt_to_bu(end, running_config["grid_res"])
        bu_starts.append(bu_start)
        bu_ends.append(bu_end)

    obstacle_ids = drop_world_walls(p, occupancy_map.copy(), running_config["grid_res"], world_config)

    # maps, obstacle_ids, bu_starts, bu_goals
    return occupancy_map, geodesic_distance_dict_list, geodesic_distance_map_list, obstacle_distance_map, potential_map_x, potential_map_y, potential_map, obstacle_ids, bu_starts, bu_ends


def load_office1000_goal_outdoor(p, running_config, worlds_config, phase):
    """
    load scene from map path and trajectory path
    """

    # scene_index = np.random.randint(0, 1000)
    # logging.error("Choose scene index:{}".format(scene_index))
    scene_parent_folder = os.path.join(get_project_path(), "data", "office_1000_goal_outdoor", phase, "random_envs")
    geodesic_distance_parent_folder = os.path.join(get_project_path(), "data", "office_1000_goal_outdoor", phase,
                                                   "geodesic_distance")

    file_names = os.listdir(geodesic_distance_parent_folder)
    file_name = np.random.choice(file_names)
    print("scene file name:{}".format(file_name))
    si = file_name.index("_") + 1
    ei = file_name.index(".")
    scene_index = int(file_name[si:ei])
    scene_path = os.path.join(scene_parent_folder, "env_{}.pkl".format(scene_index))

    geodesic_distance_dict_path = os.path.join(geodesic_distance_parent_folder, "env_{}.pkl".format(scene_index))

    # Read occupancy map, starts and ends
    occupancy_map, starts, ends = pickle.load(open(scene_path, 'rb'))
    # read geodesic_distance_map
    geodesic_distance_dict_dict = pickle.load(open(geodesic_distance_dict_path, 'rb'))

    world_name = running_config["world_name"]
    # get method to create specified scene map
    world_config = get_world_config(worlds_config, world_name)

    indexes = np.random.randint(0, len(starts), size=running_config["num_agents"])

    bu_starts = []
    bu_ends = []
    geodesic_distance_list: List[Dict] = []
    for i in indexes:
        start = starts[i]
        end = ends[0]
        geodesic_distance_list.append(geodesic_distance_dict_dict[tuple(end)])
        # geodesic_distance_dict[]
        # sample start position and goal position
        bu_start = cvt_to_bu(start, running_config["grid_res"])
        bu_end = cvt_to_bu(end, running_config["grid_res"])
        bu_starts.append(bu_start)
        bu_ends.append(bu_end)

    obstacle_ids = drop_world_walls(p, occupancy_map.copy(), running_config["grid_res"], world_config)

    # maps, obstacle_ids, bu_starts, bu_goals
    return occupancy_map, geodesic_distance_list, obstacle_ids, bu_starts, bu_ends
