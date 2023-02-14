from typing import Dict

import numpy as np
from pybullet_utils.bullet_client import BulletClient

from environment.gen_scene.build_office_world import drop_walls

import logging as logger

from environment.gen_scene.sampler_mapping import get_sampler_class, SamplerClassMapping
from environment.gen_scene.worlds_mapping import WorldMapClassMapping, get_world_creator_func
from environment.nav_utilities.coordinates_converter import cvt_to_bu
from utils.image_utility import dilate_image

from traditional_planner.a_star.astar import AStar


def get_world_config(worlds_config, world_name):
    component_configs = worlds_config[world_name]
    component_configs.update(worlds_config["configs_all"])
    return component_configs


def load_environment_scene(p: BulletClient, running_config: Dict, worlds_config: Dict, agent_sg_sampler_config: Dict):
    """
    load scene
    :param p:
    :param running_config:
    :param worlds_config:
    :return:
    """
    logger.info("Create a building...")
    world_name = running_config["world_name"]
    # get method to create specified scene map
    create_world = get_world_creator_func(world_name)

    world_config = get_world_config(worlds_config, world_name)

    occupancy_map = create_world(world_config)

    agent_sg_sampler_class = get_sampler_class(agent_sg_sampler_config["sampler_name"])
    agent_sg_sampler_params = agent_sg_sampler_config["sampler_params"]

    # dilate image
    dilated_occ_map = dilate_image(occupancy_map, running_config["dilation_size"])

    bu_starts = []
    bu_ends = []
    for i in range(running_config["num_agents"]):
        # sample start position and goal position
        [start, end], sample_success = agent_sg_sampler_class(dilate_occupancy_map=dilated_occ_map,
                                                              occupancy_map=occupancy_map,
                                                              **agent_sg_sampler_params)

        # check the connectivity between the start and end position
        if not sample_success or not check_connectivity(dilated_occ_map, start, end):
            return load_environment_scene(p, running_config, worlds_config, agent_sg_sampler_config)

        bu_start = cvt_to_bu(start, running_config["grid_res"])
        bu_end = cvt_to_bu(end, running_config["grid_res"])
        bu_starts.append(bu_start)
        bu_ends.append(bu_end)

    # create office entity in py bullet simulation environment
    obstacle_ids = drop_walls(p, occupancy_map.copy(), running_config["grid_res"], world_config)
    bu_starts = np.array(bu_starts)
    bu_ends = np.array(bu_ends)
    return occupancy_map, obstacle_ids, bu_starts, bu_ends


def check_connectivity(dilated_occ_map, start, end):
    """
    check the connectivity between start and end position in dilated_occ_map
    TODO
    可以用更粗颗粒度的occ_map来检查连通性，AStar算法遍历更快,
    或者直接计算一个连通图，这样连通图上的任何两点之间都存在可达路径
    """
    om_path = AStar(dilated_occ_map).search_path(tuple(start), tuple(end))
    is_connective = om_path is not None and len(om_path) > 10
    return is_connective


def compute_door_map(dilated_occupancy_map):
    """
    compute door map, 只有当dilation size是5的时候work
    :param dilated_occupancy_map:
    :return:
    """
    radius = 6
    door_map = np.zeros_like(dilated_occupancy_map)
    for i in range(0, dilated_occupancy_map.shape[0]):
        for j in range(0, dilated_occupancy_map.shape[1]):
            h_patch = dilated_occupancy_map[max(i - radius, 0): min(i + radius, dilated_occupancy_map.shape[0]), j]
            # max(j - 6, 0):min(j + 6, dilated_occupancy_map.shape[1])]
            v_patch = dilated_occupancy_map[i, max(j - radius, 0):min(j + radius, dilated_occupancy_map.shape[1])]

            is_h_door = np.sum(h_patch) == 0 and np.sum(v_patch) > 2 * radius * 0.7
            is_v_door = np.sum(v_patch) == 0 and np.sum(h_patch) > 2 * radius * 0.7

            if is_h_door or is_v_door:
                door_map[i, j] = True
    return door_map
