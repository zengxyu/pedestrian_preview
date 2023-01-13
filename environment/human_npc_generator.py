import logging
from typing import Dict

import numpy as np

from environment.global_planner import plan_a_star_path
from environment.nav_utilities.coordinates_converter import cvt_to_bu, cvt_to_om
from utils.image_utility import dilate_image


def generate_human_npc(dynamic_obs_sampler, env_config: Dict, occ_map: np.array, robot_start: np.array,
                       robot_end: np.array):
    # n_radius, n_from_start, n_to_end [unit: pixel]
    grid_res = env_config["grid_res"]
    # the number of human npc
    num_human_npc = int(env_config["num_human_npc"])

    n_radius = int(env_config["unobstructed_radius"] / grid_res)
    n_from_start = int(env_config["distance_from_start"] / grid_res)
    n_to_end = int(env_config["distance_to_end"] / grid_res)
    # n_static_obstacle_num = env_config["pedestrian_static_num"]

    n_kept_distance = int(env_config["kept_distance"] / grid_res)
    n_kept_distance_to_start = int(env_config["kept_distance_to_start"] / grid_res)
    dilation_size = env_config["dilation_size"]

    robot_occ_start = cvt_to_om(robot_start, grid_res)
    robot_occ_end = cvt_to_om(robot_end, grid_res)

    obs_bu_starts = []
    obs_bu_ends = []
    obs_bu_paths = []
    count = 0
    while len(obs_bu_starts) < num_human_npc:
        count += 1
        [obs_occ_start, obs_occ_end], sample_success = dynamic_obs_sampler(
            occupancy_map=dilate_image(occ_map, 2),
            robot_om_start=robot_occ_start,
            robot_om_end=robot_occ_end,
            kept_distance=n_kept_distance,
            kept_distance_to_start=n_kept_distance_to_start)

        # if sample obstacle start position and end position failed, continue to next loop and resample
        if not sample_success:
            continue

        # plan a global path for this (start, end) pair
        occ_path = plan_a_star_path(dilate_image(occ_map, 2), robot_occ_start, robot_occ_end)
        if len(occ_path) is None:
            continue
        logging.debug("There are now {} sampled obstacles".format(len(obs_bu_starts)))
        obs_bu_path = cvt_to_bu(occ_path, grid_res)
        obs_bu_start = cvt_to_bu(obs_occ_start, grid_res)
        obs_bu_end = cvt_to_bu(obs_occ_end, grid_res)

        obs_bu_paths.append(obs_bu_path)
        obs_bu_starts.append(obs_bu_start)
        obs_bu_ends.append(obs_bu_end)

    if len(obs_bu_starts) == 0 and num_human_npc > 0:
        return generate_human_npc(num_human_npc, env_config, occ_map, robot_start, robot_end)
    return obs_bu_starts, obs_bu_ends, obs_bu_paths
