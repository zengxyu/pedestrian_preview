import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
import argparse
from environment.gen_scene.common_sampler import sg_in_distance_sampler
from environment.gen_scene.world_generator import get_world_config
from environment.gen_scene.worlds_mapping import get_world_creator_func
from environment.nav_utilities.coordinates_converter import cvt_to_om
from global_planning.prm.classes import PRMController, Utils, Obstacle
from utils.config_utility import read_yaml
from utils.fo_utility import get_project_path
from utils.image_utility import dilate_image


def get_obstacles_from_occ_map(occ_map):
    obstacles = []

    MAXX = occ_map.shape[0]
    MAXY = occ_map.shape[1]

    while np.any(occ_map):
        indsx, indsy = np.where(occ_map)
        minx = maxx = indsx[0]
        miny = maxy = indsy[0]

        while np.logical_and.reduce(occ_map[maxx + 1: maxx + 1 + 1, miny: maxy + 1]) and maxx + 1 < MAXX:
            maxx += 1

        while np.logical_and.reduce(occ_map[minx - 1: minx, miny: maxy + 1]) and minx - 1 >= 0:
            minx -= 1

        while np.logical_and.reduce(occ_map[minx: maxx + 1, maxy + 1: maxy + 1 + 1]) and maxy + 1 < MAXY:
            maxy += 1

        while np.logical_and.reduce(occ_map[minx: maxx + 1, miny - 1: miny]) and miny - 1 >= 0:
            miny -= 1

        top_left = [minx - 1, miny - 1]
        bottom_right = [maxx + 1, maxy + 1]
        obs = Obstacle(top_left, bottom_right)
        obstacles.append(obs)
        occ_map[minx: maxx + 1, miny: maxy + 1] = False

    return obstacles


def prm_path_planning(occ_map, num_samples, end, grid_res):
    obstacles = get_obstacles_from_occ_map(occ_map.copy())
    # start = cvt_to_om(start, grid_res)
    start = None
    end = cvt_to_om(end, grid_res)
    # print(occ_map[start[0], start[1]])
    print(occ_map[end[0], end[1]])
    utils = Utils()
    utils.draw_map(obstacles, start, end)

    prm = PRMController(num_samples, obstacles, end, max(occ_map.shape))
    # Initial random seed to try
    initial_random_seed = 0
    prm.run_prm(initial_random_seed)
    return prm


if __name__ == '__main__':
    world_name = "office"
    worlds_config = read_yaml(os.path.join(get_project_path(), "configs"), "worlds_config.yaml")
    world_config = get_world_config(worlds_config, world_name)
    create_world = get_world_creator_func(world_name)
    occupancy_map = create_world(world_config)
    plt.imshow(occupancy_map)
    plt.show()
    [start, end], sample_success = sg_in_distance_sampler(occupancy_map=dilate_image(occupancy_map, 2),
                                                          distance_ratio=0.75)

    prm_path_planning(occupancy_map, 50, start, end)
