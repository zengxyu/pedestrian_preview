import logging
import os
import pickle
from typing import Dict

import cv2
import numpy as np

from environment.gen_scene.build_office_world import drop_walls, drop_world_walls
from environment.gen_scene.common_sampler import point_sampler
from environment.gen_scene.compute_door import compute_door
from environment.nav_utilities.coordinates_converter import cvt_to_bu

import json

from utils.fo_utility import get_project_path, get_data_path
from utils.image_utility import dilate_image


def load_p2v_scene(p, running_config, world_config, map_path, coordinates_path, num_agents):
    """
    load scene from map path and trajectory path
    """
    ratio = 0.25 / running_config["grid_res"]
    # read occupancy map from map_path
    occupancy_map = read_occupancy_map(map_path, ratio=ratio)

    dilated_occupancy_map = dilate_image(occupancy_map, running_config["dilation_size"])

    start_coordinates, goal = read_start_coordinates(start_coordinates_path=coordinates_path, ratio=ratio)
    chosen_indexes = np.random.choice(range(len(start_coordinates)), num_agents).tolist()
    starts = start_coordinates[chosen_indexes]
    # goals = np.array([goal for i in range(len(chosen_start_coordinates))])
    # dilated_occ_map = dilate_image(occupancy_map, env_config["dilation_size"])
    # 随机采样起点终点
    # [start, end], sample_success = start_goal_sampler(occupancy_map=dilated_occ_map, margin=env_config["dilation_size"])

    # starts = [point_sampler(dilated_occupancy_map) for i in range(num_agents)]
    goal = np.array([10, -10])
    goals = [goal for i in range(num_agents)]
    # compute door
    config = world_config["configs_all"]
    agent_ids = drop_world_walls(p, occupancy_map.copy(), running_config["grid_res"], config)
    agent_starts = cvt_to_bu(starts, running_config["grid_res"])
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
    selected_group_index = np.random.randint(1, num_groups)
    print(selected_group_index)
    selected_group: Dict = coordinates_groups[selected_group_index]

    start_coordinates = np.array(list(selected_group.values()))[0] * ratio
    # 60, 20
    # 10, 0
    goal = np.array([10, -10])

    return start_coordinates, goal


if __name__ == '__main__':
    p2v_map_folder = os.path.join(get_data_path(), "p2v", "env1", "maps")
    p2v_occupancy_map_folder = os.path.join(get_data_path(), "p2v", "env1", "occupancy_map")
    if not os.path.exists(p2v_occupancy_map_folder):
        os.makedirs(p2v_occupancy_map_folder)

    # 地图读入路径
    p2v_map_path = os.path.join(p2v_map_folder, "map.txt")
    # 地图写出路径
    p2v_occupancy_map_path = os.path.join(p2v_occupancy_map_folder, "occupancy_map.pkl")
    ratio = 0.25 / 0.1
    occupancy_map = read_occupancy_map(p2v_map_path, ratio)
    p2v_occ_map_file = open(p2v_occupancy_map_path, "wb")
    pickle.dump(occupancy_map, p2v_occ_map_file)






















