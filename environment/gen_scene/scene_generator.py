from typing import Dict

import numpy as np
from pybullet_utils.bullet_client import BulletClient

from environment.gen_scene.build_office_world import drop_walls
from environment.gen_scene.common_sampler import start_goal_sampler, distant_start_end_sampler
from environment.gen_scene.gen_corridor_map import create_corridor_map
from environment.gen_scene.gen_cross_map import create_cross_map
from environment.gen_scene.gen_office_map import create_office_map

import logging as logger

from environment.gen_scene.gen_simple_map import create_simple_environment
from environment.nav_utilities.coordinates_converter import cvt_to_bu
from utils.image_utility import dilate_image

from traditional_planner.a_star.astar import AStar


def load_environment_scene(p: BulletClient, env_config: Dict, worlds_config: Dict):
    """
    load scene
    :param p:
    :param env_config:
    :param worlds_config:
    :return:
    """
    logger.info("Create a building...")
    scene_name = env_config["scene_name"]

    if scene_name == "office":
        # create a new office
        component_configs = worlds_config["office"]
        component_configs.update(worlds_config["configs_all"])
        occupancy_map, samplers = create_office_map(component_configs)

    elif scene_name == "corridor":
        # create a tunnel
        component_configs = worlds_config["corridor"]
        component_configs.update(worlds_config["configs_all"])
        occupancy_map, samplers = create_corridor_map(component_configs)

    elif scene_name == "cross":
        # create a cross
        component_configs = worlds_config["cross"]
        component_configs.update(worlds_config["configs_all"])
        occupancy_map, samplers = create_cross_map(component_configs)

    elif scene_name == "empty":
        component_configs = worlds_config[scene_name]
        component_configs.update(worlds_config["configs_all"])
        occupancy_map, samplers = create_simple_environment(component_configs)

    else:
        raise NotImplementedError

    # dilate image
    dilated_occ_map = dilate_image(occupancy_map, env_config["dilation_size"])
    door_occ_map = compute_door_map(dilated_occ_map)

    # sample start position and goal position
    # start_goal_sampler = samplers[0]

    bu_starts = []
    bu_ends = []
    for i in range(env_config["num_agents"]):
        # [_, end], sample_success = start_goal_sampler(occupancy_map=dilated_occ_map,
        #                                               margin=env_config["dilation_size"])

        [start, end], sample_success = distant_start_end_sampler(occupancy_map=dilated_occ_map,
                                                               margin=env_config["dilation_size"])

        # check the connectivity between the start and end position
        if not sample_success or not check_connectivity(dilated_occ_map, start, end):
            load_environment_scene(p, env_config, worlds_config)

        bu_start = cvt_to_bu(start, env_config["grid_res"])
        bu_end = cvt_to_bu(end, env_config["grid_res"])
        bu_starts.append(bu_start)
        bu_ends.append(bu_end)

    # map results
    maps = {"occ_map": occupancy_map, "dilated_occ_map": dilated_occ_map, "door_map": door_occ_map}

    # create office entity in py bullet simulation environment
    obstacle_ids = drop_walls(p, occupancy_map.copy(), env_config["grid_res"], component_configs)

    return maps, samplers, obstacle_ids, bu_starts, bu_ends


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
