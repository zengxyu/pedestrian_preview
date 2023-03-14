import logging
from typing import Dict

import numpy as np

from environment.gen_scene.sampler_mapping import get_sampler_class
from environment.global_planner import plan_a_star_path
from environment.nav_utilities.coordinates_converter import cvt_to_bu, cvt_to_om
from utils.image_utility import dilate_image


def generate_human_npc_with_specified_goal(running_config: Dict, occ_map: np.array, goal: np.array,
                                           npc_sg_sampler_config: Dict):
    # n_radius, n_from_start, n_to_end [unit: pixel]
    grid_res = running_config["grid_res"]
    # the number of human npc
    num_human_npc = int(running_config["num_npc"])

    npc_sg_sampler_class = get_sampler_class(npc_sg_sampler_config["sampler_name"])
    npc_sg_sampler_params = npc_sg_sampler_config["sampler_params"]

    npc_starts = []
    npc_ends = []
    npc_paths = []
    occ_end = cvt_to_om(goal, grid_res)
    count = 0
    while len(npc_starts) < num_human_npc:
        count += 1
        [occ_start, _], sample_success = npc_sg_sampler_class(
            occupancy_map=dilate_image(occ_map, 2),
            distance_ratio=npc_sg_sampler_params["distance_ratio"]
        )

        # if sample obstacle start position and end position failed, continue to next loop and resample
        if not sample_success:
            continue

        # plan a global path for this (start, end) pair
        occ_path = plan_a_star_path(dilate_image(occ_map, 2), np.array(occ_start, dtype=np.float),
                                    np.array(occ_end, dtype=np.float))
        if occ_path is None or len(occ_path) is None:
            continue
        logging.debug("There are now {} sampled obstacles".format(len(npc_starts)))
        npc_path = cvt_to_bu(occ_path, grid_res)
        npc_start = cvt_to_bu(occ_start, grid_res)
        # npc_end = cvt_to_bu(occ_end, grid_res)

        npc_paths.append(npc_path.copy())
        npc_starts.append(npc_start)
        npc_ends.append(goal)

    if len(npc_starts) == 0 and num_human_npc > 0:
        return generate_human_npc_with_specified_goal(running_config, occ_map, goal, npc_sg_sampler_config)
    return npc_starts, npc_ends, npc_paths


def generate_human_npc(running_config: Dict, occ_map: np.array, npc_sg_sampler_config: Dict):
    # n_radius, n_from_start, n_to_end [unit: pixel]
    grid_res = running_config["grid_res"]
    # the number of human npc
    num_human_npc = int(running_config["num_npc"])

    npc_sg_sampler_class = get_sampler_class(npc_sg_sampler_config["sampler_name"])
    npc_sg_sampler_params = npc_sg_sampler_config["sampler_params"]

    npc_starts = []
    npc_ends = []
    npc_paths = []
    count = 0
    while len(npc_starts) < num_human_npc:
        count += 1
        [occ_start, occ_end], sample_success = npc_sg_sampler_class(
            occupancy_map=dilate_image(occ_map, 2),
            distance_ratio=npc_sg_sampler_params["distance_ratio"]
        )

        # if sample obstacle start position and end position failed, continue to next loop and resample
        if not sample_success:
            continue

        # plan a global path for this (start, end) pair
        occ_path = plan_a_star_path(dilate_image(occ_map, 2), np.array(occ_start, dtype=np.float),
                                    np.array(occ_end, dtype=np.float))
        if occ_path is None or len(occ_path) is None:
            continue
        logging.debug("There are now {} sampled obstacles".format(len(npc_starts)))
        npc_path = cvt_to_bu(occ_path, grid_res)
        npc_start = cvt_to_bu(occ_start, grid_res)
        npc_end = cvt_to_bu(occ_end, grid_res)

        npc_paths.append(npc_path.copy())
        npc_starts.append(npc_start)
        npc_ends.append(npc_end)

    if len(npc_starts) == 0 and num_human_npc > 0:
        return generate_human_npc(running_config, occ_map, npc_sg_sampler_config)
    return npc_starts, npc_ends, npc_paths
