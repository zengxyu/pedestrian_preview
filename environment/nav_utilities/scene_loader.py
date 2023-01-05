from typing import Dict

import numpy as np
from pybullet_utils.bullet_client import BulletClient

from environment.gen_scene.build_office_world import drop_walls
from environment.nav_utilities.coordinates_converter import cvt_to_bu
from environment.gen_scene.gen_corridor_map import create_corridor_map
from environment.gen_scene.gen_cross_map import create_cross_map
from environment.gen_scene.gen_office_map import create_office_map

import logging as logger

from environment.gen_scene.gen_simple_map import create_simple_environment
from utils.image_utility import dilate_image


def load_environment_scene(p: BulletClient, high_env_config: Dict, world_config: Dict, grid_resolution: float):
    """
    load office
    :param p:
    :param high_env_config:
    :param world_config:
    :param grid_resolution:
    :return:
    """
    probs = high_env_config["env_probs"] / np.sum(high_env_config["env_probs"])
    building_name = np.random.choice(a=high_env_config["env_types"], p=probs)
    logger.info("Create a building : {}".format(building_name))

    if building_name == "office":
        # create a new office
        component_configs = world_config["office"]
        component_configs.update(world_config["configs_all"])
        occupancy_map, samplers = create_office_map(component_configs)

    elif building_name == "corridor":
        # create a tunnel
        component_configs = world_config["corridor"]
        component_configs.update(world_config["configs_all"])
        occupancy_map, samplers = create_corridor_map(component_configs)

    elif building_name == "cross":
        # create a cross
        component_configs = world_config["cross"]
        component_configs.update(world_config["configs_all"])
        occupancy_map, samplers = create_cross_map(component_configs)

    elif building_name == "simple":
        # create a simple environment
        component_configs = world_config["simple"]
        component_configs.update(world_config["configs_all"])
        occupancy_map, samplers = create_simple_environment(component_configs)

    else:
        raise NotImplementedError

    # dilate image
    dilated_occ_map = dilate_image(occupancy_map, dilation_size=high_env_config["dilation_size"])
    door_occ_map = compute_door_map(dilated_occ_map)
    # create office entity in py bullet simulation environment
    obstacle_ids = drop_walls(p, occupancy_map.copy(), grid_resolution, component_configs)

    return obstacle_ids, occupancy_map, dilated_occ_map, door_occ_map, samplers, building_name


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


def plan_linear_path(p: BulletClient, grid_res=0, start=None, end=None):
    xs = np.arange(start[0], end[1], step=1)
    ys = np.arange(start[0], end[1], step=1)
    om_path = np.array([[x, y] for x, y in zip(xs, ys)])
    bullet_path = cvt_to_bu(om_path, grid_res)

    return bullet_path
