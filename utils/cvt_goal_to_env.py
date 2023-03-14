import os
import pickle

import numpy as np

from utils.gen_fixed_envs import display_and_save
from utils.fo_utility import *


def compute_env_images(dataset_path, folder_name, phase, indexes):
    env_folder = os.path.join(dataset_path, folder_name, phase, "envs")
    image_folder = os.path.join(dataset_path, folder_name, phase, "envs_images")
    env_name_template = "env_{}.pkl"
    image_name_template = "env_{}.png"

    for i in indexes:
        env_name = env_name_template.format(i)
        image_name = image_name_template.format(i)
        env_path = os.path.join(env_folder, env_name)
        image_path = os.path.join(image_folder, image_name)
        print("Computing env image for {}...".format(env_name))
        occupancy_map, starts, ends = pickle.load(open(env_path, "rb"))
        ends_tile = np.tile(ends, (len(starts), 1))
        pickle.dump([occupancy_map, starts, ends_tile], open(env_path, "wb"))
        display_and_save(occupancy_map, starts, ends_tile, save=True, save_path=image_path)
        # print("Save to {}!".format(image_path))


if __name__ == '__main__':
    dataset_path = get_office_evacuation_path()
    folder_name = "goal_at_door"
    phase = "train"
    indexes = [i for i in range(1200, 1700)]
    compute_env_images(dataset_path, folder_name, phase, indexes)
