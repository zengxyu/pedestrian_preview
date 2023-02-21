import logging
import os
import pickle
from typing import Dict, List

import numpy as np

from environment.gen_scene.build_office_world import drop_world_walls
from environment.gen_scene.world_generator import get_world_config
from environment.nav_utilities.coordinates_converter import cvt_to_bu
from utils.fo_utility import get_project_path


def load_office1000_scene(p, running_config, worlds_config):
    """
    load scene from map path and trajectory path
    """

    # scene_index = np.random.randint(0, 1000)
    # logging.error("Choose scene index:{}".format(scene_index))
    scene_parent_folder = os.path.join(get_project_path(), "data", "office_1000", "random_envs")
    geodesic_distance_parent_folder = os.path.join(get_project_path(), "data", "office_1000", "geodesic_distance")

    file_names = os.listdir(geodesic_distance_parent_folder)
    file_name = np.random.choice(file_names)
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
        end = ends[i]
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