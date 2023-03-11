import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np

from utils.fo_utility import *


def compute_min_obstacle_distance(file_name):
    fr = open(file_name, 'rb')
    inf = pickle.load(fr)
    occupancy_map = inf[0]
    xx, yy = np.where(np.invert(occupancy_map))
    free_cells = np.array(np.where(np.invert(occupancy_map))).transpose(1, 0)
    occupied_cells = np.array(np.where(occupancy_map)).transpose(1, 0)
    free_cells = np.expand_dims(free_cells, 1).repeat(len(occupied_cells), 1)
    occupied_cells = np.expand_dims(occupied_cells, 0).repeat(len(free_cells), 0)

    distances = np.linalg.norm(free_cells - occupied_cells, axis=2)
    min_distances = np.min(distances, axis=1)
    obstacle_distance_map = np.zeros_like(occupancy_map).astype(float)
    obstacle_distance_map[xx, yy] = min_distances

    return obstacle_distance_map


def compute_min_obstacle_distances(folder_name, phase, indexes):
    env_parent_folder = os.path.join(get_office_evacuation_path(), folder_name, phase, "envs")
    obs_dist_folder = os.path.join(get_office_evacuation_path(), folder_name, phase, "obstacle_distance")
    image_folder = os.path.join(get_office_evacuation_path(), folder_name, phase, "obstacle_distance_images")
    if not os.path.exists(obs_dist_folder):
        os.makedirs(obs_dist_folder)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    env_name_template = "env_{}.pkl"
    image_name_template = "env_{}.png"

    for i in indexes:
        env_name = env_name_template.format(i)
        env_path = os.path.join(env_parent_folder, env_name)
        print("Computing obstacle distance for {}...".format(env_name))
        out = compute_min_obstacle_distance(file_name=env_path)
        out_path = os.path.join(obs_dist_folder, env_name)
        pickle.dump(out, open(out_path, 'wb'))

        image_name = image_name_template.format(i)
        image_path = os.path.join(image_folder, image_name)
        display_and_save(out, save=True, save_path=image_path)
        print("Save to {}!".format(out_path))
    print("Done!")


def display_and_save(obstacle_distance_map, save, save_path):
    plt.imshow(obstacle_distance_map)

    if save:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()


if __name__ == '__main__':
    folder_name = "sg_walls"
    phase = "train"
    # 要处理从哪个到哪个文件
    indexes = [i for i in range(1500, 1700)]
    compute_min_obstacle_distances(folder_name, phase, indexes)
