import os
import pickle
from typing import Dict, List

import numpy as np

from environment.gen_scene.build_office_world import drop_world_walls
from environment.gen_scene.common_sampler import point_sampler
from environment.gen_scene.world_generator import get_world_config
from environment.nav_utilities.coordinates_converter import cvt_to_bu
from utils.fo_utility import *
from utils.image_utility import dilate_image


def check_office1000_folder_structure():
    url = "https://pan.dm-ai.com/s/AsHXrJGKe4NLsKH"
    password = "12345678"
    office1000_parent_folder = os.path.join(get_project_path(), "data/office_evacuation")
    spaces = ["\n\t-", "\n\t\t-", "\n\t\t\t-", "-\n\t\t\t\t", "\n\t\t\t\t\t-"]

    parent_folder = "data"
    sub1_folders = ["office_evacuation"]
    sub2_folders = ["sg_walls", "sg_no_walls", "goal_at_door"]
    sub3_folders = ["train", "test"]
    sub4_folders = ["envs", "geodesic_distance", "obstacle_distance", "u_forces", "v_forces"]
    warning1 = "Please download data from url:{}; password:{}; and put it in your project_folder/data; ".format(url,
                                                                                                                password)
    folder_structure = parent_folder
    for sub1_folder in sub1_folders:
        space1 = spaces[0]
        folder_structure += space1
        folder_structure += sub1_folder
        for sub2_folder in sub2_folders:
            space2 = spaces[1]
            folder_structure += space2
            folder_structure += sub2_folder
            for sub3_folder in sub3_folders:
                space3 = spaces[2]
                folder_structure += space3
                folder_structure += sub3_folder
                for sub4_folder in sub4_folders:
                    space4 = spaces[3]
                    folder_structure += space4
                    folder_structure += sub4_folder

    warning2 = "Your folder structure should be like : {}".format(folder_structure)
    assert os.path.exists(office1000_parent_folder), warning1 + warning2

    for sub1_folder in sub1_folders:
        for sub2_folder in sub2_folders:
            for sub3_folder in sub3_folders:
                for sub4_folder in sub4_folders:
                    path = os.path.join(get_project_path(), "data", sub1_folder, sub2_folder, sub3_folder, sub4_folder)
                    warning3 = "\n" + path
                    assert os.path.exists(path), warning1 + warning2 + warning3
                    length = len(os.listdir(path))
                    if sub3_folder == "train":
                        assert length >= 1200, warning1 + warning2 + warning3 + ";length={}".format(length)
                    else:
                        assert length == 240, warning1 + warning2 + warning3 + ";length={}".format(length)


def sample_starts(occupancy_map, running_config):
    dilated_occ_map = dilate_image(occupancy_map, dilation_size=3)
    starts = [np.array(point_sampler(dilated_occ_map))]
    for i in range(running_config["num_agents"] - 1):
        too_close = True
        while too_close:
            start = point_sampler(dilated_occ_map)
            start = np.array(start)
            pre_starts = np.array(starts)
            distances = np.linalg.norm(pre_starts - start[np.newaxis, :], axis=1)
            min_distance = np.min(distances)
            if min_distance < 10:
                too_close = True
            else:
                too_close = False
        starts.append(start)
    return starts


def load_office1000_scene(p, running_config, worlds_config, phase, parent_folder):
    """
    load scene from map path and trajectory path
    """
    phase = "train"
    # scene_index = np.random.randint(0, 1000)
    # logging.error("Choose scene index:{}".format(scene_index))
    envs_folder = os.path.join(parent_folder, phase, "envs")
    geodesic_distance_folder = os.path.join(parent_folder, phase, "geodesic_distance")
    obstacle_distance_folder = os.path.join(parent_folder, phase, "obstacle_distance")
    u_folder = os.path.join(parent_folder, phase, "u_forces")
    v_folder = os.path.join(parent_folder, phase, "v_forces")

    # 确定场景
    file_names = os.listdir(geodesic_distance_folder)
    file_name = np.random.choice(file_names)
    print("scene file name:{}".format(file_name))
    si = file_name.index("_") + 1
    ei = file_name.index(".")
    # index = int(file_name[si:ei])
    index = np.random.randint(600, 1700)
    env = os.path.join(envs_folder, "env_{}.pkl".format(index))

    obstacle_distance_path = os.path.join(obstacle_distance_folder, "env_{}.pkl".format(index))
    geodesic_distance_dict_path = os.path.join(geodesic_distance_folder, "env_{}.pkl".format(index))
    u_path = os.path.join(u_folder, "env_{}.pkl".format(index))
    v_path = os.path.join(v_folder, "env_{}.pkl".format(index))

    # Read occupancy map, starts and ends
    occupancy_map, starts, ends = pickle.load(open(env, 'rb'))
    # starts = sample_starts(occupancy_map, running_config)
    # read obstacle distance map
    obstacle_distance_map = pickle.load(open(obstacle_distance_path, 'rb'))
    # read geodesic_distance_map
    geodesic_distance_dict_dict = pickle.load(open(geodesic_distance_dict_path, 'rb'))
    force_ux, force_uy, force_u = pickle.load(open(u_path, 'rb'))
    v_map_dict = pickle.load(open(v_path, 'rb'))

    indexes = np.random.randint(0, len(starts), size=running_config["num_agents"])

    bu_starts = []
    bu_ends = []
    geodesic_distance_dict_list: List[Dict] = []

    force_vxs = []
    force_vys = []
    force_vs = []
    for i in indexes:
        start = starts[i]
        if len(ends) == 1:
            end = ends[0]
        else:
            end = ends[i]

        # 取出对应终点的测地距离dict
        geodesic_distance_dict = geodesic_distance_dict_dict[tuple(end)]

        # 将测地距离dict保存到list中
        geodesic_distance_dict_list.append(geodesic_distance_dict)

        # 将
        force_vx, force_vy, force_v = v_map_dict[tuple(end)]
        force_vxs.append(force_vx)
        force_vys.append(force_vy)
        force_vs.append(force_v)

        # sample start position and goal position
        bu_start = cvt_to_bu(start, running_config["grid_res"])
        bu_end = cvt_to_bu(end, running_config["grid_res"])
        bu_starts.append(bu_start)
        bu_ends.append(bu_end)

    world_name = running_config["world_name"]
    # get method to create specified scene map
    world_config = get_world_config(worlds_config, world_name)

    obstacle_ids = drop_world_walls(p, occupancy_map.copy(), running_config["grid_res"], world_config)

    # maps, obstacle_ids, bu_starts, bu_goals
    return occupancy_map, geodesic_distance_dict_list, obstacle_distance_map, force_ux, force_uy, force_u, force_vxs, force_vys, force_vs, obstacle_ids, bu_starts, bu_ends
