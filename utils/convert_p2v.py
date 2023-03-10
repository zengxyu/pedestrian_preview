import os
import pickle

import numpy as np

from environment.gen_scene.common_sampler import sg_opposite_baffle_sampler2, sg_opposite_baffle_sampler4, \
    sg_opposite_baffle_sampler3
from environment.gen_scene.compute_door import compute_door
from environment.gen_scene.world_loader import read_occupancy_map
from utils.fo_utility import *
from utils.gen_fixed_envs import display_and_save
from utils.image_utility import dilate_image


def convert_p2v_to_occupancy_map():
    p2v_map_folder = os.path.join(get_data_path(), "p2v", "env1", "maps")
    p2v_occupancy_map_folder = os.path.join(get_data_path(), "p2v", "env1", "occupancy_map")
    if not os.path.exists(p2v_occupancy_map_folder):
        os.makedirs(p2v_occupancy_map_folder)

    # 地图读入路径
    p2v_map_path = os.path.join(p2v_map_folder, "map.txt")
    # 地图写出路径
    p2v_occupancy_map_path = os.path.join(p2v_occupancy_map_folder, "occupancy_map.pkl")
    # 转换为occupancy map
    ratio = 0.25 / 0.1
    occupancy_map = read_occupancy_map(p2v_map_path, ratio)

    phase = "test"
    p2v_goal_at_door = os.path.join(get_p2v_path(), "goal_at_door", phase)
    p2v_sg_no_walls = os.path.join(get_p2v_path(), "sg_no_walls", phase)
    p2v_sg_walls = os.path.join(get_p2v_path(), "sg_walls", phase)

    p2v_sg_walls_envs_folder = os.path.join(p2v_sg_walls, "envs")
    p2v_goal_at_door_envs_folder = os.path.join(p2v_goal_at_door, "envs")
    p2v_sg_no_walls_envs_folder = os.path.join(p2v_sg_no_walls, "envs")

    p2v_sg_walls_envs_images_folder = os.path.join(p2v_sg_walls, "envs_images")
    p2v_goal_at_door_envs_images_folder = os.path.join(p2v_goal_at_door, "envs_images")
    p2v_sg_no_walls_envs_images_folder = os.path.join(p2v_sg_no_walls, "envs_images")

    if not os.path.exists(p2v_sg_walls_envs_folder):
        os.makedirs(p2v_sg_walls_envs_folder)
    if not os.path.exists(p2v_goal_at_door_envs_folder):
        os.makedirs(p2v_goal_at_door_envs_folder)
    if not os.path.exists(p2v_sg_no_walls_envs_folder):
        os.makedirs(p2v_sg_no_walls_envs_folder)
    if not os.path.exists(p2v_sg_walls_envs_images_folder):
        os.makedirs(p2v_sg_walls_envs_images_folder)
    if not os.path.exists(p2v_goal_at_door_envs_images_folder):
        os.makedirs(p2v_goal_at_door_envs_images_folder)
    if not os.path.exists(p2v_sg_no_walls_envs_images_folder):
        os.makedirs(p2v_sg_no_walls_envs_images_folder)
    # 采样100个起点终点对，其中20个终点从环境内采样，20个终点从门口采样，20个起点终点之间没有动态障碍物
    dilated_occ_map = dilate_image(occupancy_map, dilation_size=5)
    # door_center = compute_door(occupancy_map).tolist()
    door_center = np.array([10, 0])
    # sample start point
    count = 0
    starts = []
    ends = []
    num_starts = 20
    # 采样起点，和终点之间有障碍物
    while len(starts) < num_starts and count < 100:
        # print("start point number:{}".format(len(starts)))
        [start, end], sample_success = sg_opposite_baffle_sampler2(dilate_occupancy_map=dilated_occ_map,
                                                                   occupancy_map=occupancy_map)

        if sample_success:
            starts.append(start)
            ends.append(end)
        count += 1
    starts = np.array(starts)
    ends = np.array(ends)
    # 保存
    file = open(os.path.join(p2v_sg_walls_envs_folder, "env_0.pkl"), "wb")
    pickle.dump([occupancy_map, starts, ends], file)
    file.close()
    image_path = os.path.join(p2v_sg_walls_envs_images_folder, "env_0.png")
    display_and_save(occupancy_map, starts, ends, save=True, save_path=image_path)

    # sample start point
    count = 0
    starts = []
    ends = []
    # 采样起点，和终点之间没有障碍物
    while len(starts) < 2 * num_starts and count < 100:
        # print("start point number:{}".format(len(starts)))
        start, sample_success = sg_opposite_baffle_sampler3(dilate_occupancy_map=dilated_occ_map,
                                                            occupancy_map=occupancy_map,
                                                            goal=door_center)

        if sample_success:
            starts.append(start)
            ends.append(door_center)
        count += 1
    starts = np.array(starts)
    ends = np.array(ends)
    file = open(os.path.join(p2v_goal_at_door_envs_folder, "env_0.pkl"), "wb")
    pickle.dump([occupancy_map, starts, ends], file)
    file.close()
    image_path = os.path.join(p2v_goal_at_door_envs_images_folder, "env_0.png")
    display_and_save(occupancy_map, starts, ends, save=True, save_path=image_path)

    # sample start point
    count = 0
    starts = []
    ends = []
    # 采样起点，和终点之间没有障碍物
    while len(starts) < 3 * num_starts and count < 100:
        # print("start point number:{}".format(len(starts)))
        [start, end], sample_success = sg_opposite_baffle_sampler4(dilate_occupancy_map=dilated_occ_map,
                                                                   occupancy_map=occupancy_map)

        if sample_success:
            starts.append(start)
            ends.append(end)
        count += 1
    starts = np.array(starts)
    ends = np.array(ends)
    file = open(os.path.join(p2v_sg_no_walls_envs_folder, "env_0.pkl"), "wb")
    pickle.dump([occupancy_map, starts, ends], file)
    file.close()
    image_path = os.path.join(p2v_sg_no_walls_envs_images_folder, "env_0.png")
    display_and_save(occupancy_map, starts, ends, save=True, save_path=image_path)


if __name__ == '__main__':
    convert_p2v_to_occupancy_map()
