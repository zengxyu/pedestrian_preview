import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np

from utils.fo_utility import get_project_path


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


def display_and_save(obstacle_distance_map, save, save_path):
    plt.imshow(obstacle_distance_map)

    if save:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()


if __name__ == '__main__':
    phase = "train"
    env_parent_folder = os.path.join(get_project_path(), "data", "office_1000", phase, "random_envs")
    obstacle_distance_parent_folder = os.path.join(get_project_path(), "data", "office_1000", phase,
                                                   "obstacle_distance")
    image_save_folder = os.path.join(get_project_path(), "data", "office_1000", phase,
                                     "obstacle_distance_images")
    if not os.path.exists(obstacle_distance_parent_folder):
        os.makedirs(obstacle_distance_parent_folder)
    if not os.path.exists(image_save_folder):
        os.makedirs(image_save_folder)

    env_names = os.listdir(env_parent_folder)
    length = len(env_names)
    template = "env_{}.pkl"
    image_save_name_template = "env_{}.png"

    indexes = [a for a in range(1000)]
    for i in indexes:
        env_name = template.format(i)
        env_path = os.path.join(env_parent_folder, env_name)
        print("Computing obstacle distance for {}...".format(env_name))
        out = compute_min_obstacle_distance(file_name=env_path)
        out_path = os.path.join(obstacle_distance_parent_folder, env_name)
        pickle.dump(out, open(out_path, 'wb'))

        image_save_file_name = image_save_name_template.format(i)
        image_save_path = os.path.join(image_save_folder, image_save_file_name)
        display_and_save(out, save=True, save_path=image_save_path)
        print("Save to {}!".format(out_path))
    print("Done!")
