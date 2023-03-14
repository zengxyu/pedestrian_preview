import os
import pickle

from utils.gen_fixed_envs import display_and_save
from utils.fo_utility import *


def compute_env_images(dataset_path, folder_name, phase, indexes):
    env_parent_folder = os.path.join(dataset_path, folder_name, phase, "envs")
    env_images_folder = os.path.join(dataset_path, folder_name, phase, "envs_images")
    env_name_template = "env_{}.pkl"
    image_name_template = "env_{}.png"

    for i in indexes:
        env_name = env_name_template.format(i)
        env_path = os.path.join(env_parent_folder, env_name)
        print("Computing env image for {}...".format(env_name))
        occupancy_map, starts, ends = pickle.load(open(env_path, "rb"))
        image_name = image_name_template.format(i)
        image_path = os.path.join(env_images_folder, image_name)
        display_and_save(occupancy_map, starts, ends, save=True, save_path=image_path)
        print("Save to {}!".format(image_path))


if __name__ == '__main__':
    dataset_path = get_office_evacuation_path()
    folder_name = "sg_no_walls"
    phase = "train"
    indexes = [i for i in range(0, 1000)]
    compute_env_images(dataset_path, folder_name, phase, indexes)
