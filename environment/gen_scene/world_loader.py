import logging
import os
from typing import Dict

import cv2
import numpy as np

from environment.gen_scene.build_office_world import drop_walls
from environment.nav_utilities.coordinates_converter import cvt_to_bu

import json


def load_scene(p, running_config, world_config, map_path, coordinates_path):
    """
    load scene from map path and trajectory path
    """
    ratio = 0.25 / running_config["grid_res"]
    # read occupancy map from map_path
    occupancy_map = read_occupancy_map(map_path, ratio=ratio)

    start_coordinates, goal = read_start_coordinates(start_coordinates_path=coordinates_path, ratio=ratio)
    goals = np.array([goal for i in range(len(start_coordinates))])
    # dilated_occ_map = dilate_image(occupancy_map, env_config["dilation_size"])
    # 随机采样起点终点
    # [start, end], sample_success = start_goal_sampler(occupancy_map=dilated_occ_map, margin=env_config["dilation_size"])
    #
    config = world_config["configs_all"]
    agent_ids = drop_walls(p, occupancy_map.copy(), running_config["grid_res"], config)
    agent_starts = cvt_to_bu(start_coordinates, running_config["grid_res"])
    agent_goals = cvt_to_bu(goals, running_config["grid_res"])
    # maps, obstacle_ids, bu_starts, bu_goals
    return occupancy_map, agent_ids, agent_starts, agent_goals


def read_occupancy_map(map_path: str, ratio: float):
    if not os.path.exists(map_path):
        logging.error("Map path:{} not exist!".format(map_path))

    lines = np.loadtxt(map_path, delimiter=",")
    map_data_copy = np.array(lines)
    # flip
    map_data_copy = np.where(map_data_copy < 0, 1, map_data_copy)
    h, w = map_data_copy.shape

    # 给左右上下建立围墙
    new_map = np.zeros(shape=(h + 2, w + 2))
    new_map[1:-1, 1:-1] = map_data_copy
    # 建围墙
    new_map[:, [0, -1]] = True
    new_map[[0, -1], :] = True
    # 开门
    # 开右边的门
    door_size = 6
    new_map[-1 - door_size:-1, -1] = False
    # 开顶部的门
    new_map[0, 1:1 + door_size] = False

    # local occupancy resolution 0.2m/cell, convert to 0.1m/cell
    new_map = cv2.resize(new_map, (int(new_map.shape[1] * ratio), int(new_map.shape[0] * ratio)))
    new_map = np.where(new_map >= 0.7, 1, 0)
    new_map = np.transpose(new_map, (1, 0))
    return new_map


def read_start_coordinates(start_coordinates_path: str, ratio: float):
    if not os.path.exists(start_coordinates_path):
        logging.error("Start coordinate path:{} not exist!".format(start_coordinates_path))

    file = open(start_coordinates_path, 'r')
    lines = file.readlines()
    coordinates_groups = []
    for line in lines:
        one_group = json.loads(line.strip())
        coordinates_groups.append(one_group)

    num_groups = len(coordinates_groups)

    # select one of the start coordinates groups
    selected_group_index = np.random.randint(0, num_groups)
    selected_group: Dict = coordinates_groups[0]

    start_coordinates = np.array(list(selected_group.values()))[0] * ratio
    goal = np.array([-1, -1])

    return start_coordinates, goal
