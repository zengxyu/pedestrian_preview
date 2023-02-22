import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from environment.gen_scene.sampler_mapping import get_sampler_class
from environment.gen_scene.world_generator import get_world_config
from environment.gen_scene.worlds_mapping import get_world_creator_func
from utils.config_utility import read_yaml
from utils.fo_utility import get_project_path
from utils.image_utility import dilate_image


def generate_env(num_starts):
    # read specified world config
    world_name = "office"
    worlds_config = read_yaml(os.path.join(get_project_path(), "configs"), "worlds_config.yaml")
    world_config = get_world_config(worlds_config, world_name)

    # read specified sampler config
    agent_sampler_name = "agent_sg_opposite_baffle_sampler2"
    samplers_config = read_yaml(os.path.join(get_project_path(), "configs"), "samplers_config.yaml")
    agent_sampler_config = samplers_config[agent_sampler_name]
    agent_sg_sampler_class = get_sampler_class(agent_sampler_config["sampler_name"])
    agent_sg_sampler_params = agent_sampler_config["sampler_params"]

    # create world occupancy map
    create_world = get_world_creator_func(world_name)
    occupancy_map = create_world(world_config)
    # dilate occupancy map
    dilated_occ_map = dilate_image(occupancy_map, dilation_size=5)

    starts = []
    ends = []
    count = 0
    sample_success = False
    while len(starts) < num_starts and count < 100:
        # print("start point number:{}".format(len(starts)))

        # sample start and goal
        [start, end], sample_success = agent_sg_sampler_class(dilate_occupancy_map=dilated_occ_map,
                                                              occupancy_map=occupancy_map,
                                                              **agent_sg_sampler_params)
        # print("sample_success:{}".format(sample_success))

        if sample_success:
            starts.append(start)
            ends.append(end)
        count += 1
    sample_success = len(starts) >= num_starts
    return occupancy_map, np.array(starts), np.array(ends), sample_success


def display_and_save(occupancy_map, starts, ends, save, save_path):
    plt.imshow(occupancy_map)
    plt.scatter(starts[:, 1], starts[:, 0], c='g')
    plt.scatter(ends[:, 1], ends[:, 0], c='g')
    for s, e in zip(starts, ends):
        plt.plot([s[1], e[1]], [s[0], e[0]])

    if save:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()


def generate_n_envs(num_envs, num_starts, parent_folder, image_save_folder):
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
    if not os.path.exists(image_save_folder):
        os.makedirs(image_save_folder)

    save_name_template = "env_{}.pkl"
    image_save_name_template = "env_{}.png"
    envs = []
    i = 0
    indexes = [a for a in range(num_envs)]
    while i < len(indexes):
        index = indexes[i]
        save_file_name = save_name_template.format(index)
        image_save_file_name = image_save_name_template.format(index)
        save_path = os.path.join(parent_folder, save_file_name)
        image_save_path = os.path.join(image_save_folder, image_save_file_name)
        print("Generating {}-th office...".format(index))
        occupancy_map, starts, ends, sample_success = generate_env(num_starts=num_starts)
        if sample_success:
            env = [occupancy_map, starts, ends]
            print("Save env {} to {} ... ".format(save_file_name, save_path))
            pickle.dump(env, open(save_path, 'wb'))
            display_and_save(occupancy_map, starts, ends, save=True, save_path=image_save_path)

            print("Save done!")
            i += 1
        # envs.append([occupancy_map, starts, ends])
    return envs


def store_envs(envs, parent_folder):
    """
    save environments to parent folder
    """
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    save_file_name_template = "env_{}.pkl"
    for i, env in enumerate(envs):
        save_file_name = save_file_name_template.format(i)
        save_path = os.path.join(parent_folder, save_file_name)
        logging.info("Save env {} to {} ... ".format(save_file_name, save_path))
        pickle.dump(env, open(save_path, 'wb'))
        logging.info("Save done!")


def read_env(file_path):
    """
    read single environment from file path
    """
    file = open(file_path, 'rb')
    env = pickle.load(file)
    return env


def read_envs(parent_folder):
    """
    read environments from parent folder
    """
    file_names = os.listdir(parent_folder)
    file_paths = [os.path.join(parent_folder, file_name) for file_name in file_names]
    envs = []
    for file_path in file_paths:
        env = read_env(file_path)
        envs.append(env)
    return envs


def run_read_envs():
    parent_folder = os.path.join(get_project_path(), "data", "random_envs")
    envs = read_envs(parent_folder)
    display_and_save(*envs[0])


def run_store_envs():
    # occupancy_map, starts, ends = generate_env(num_starts=20)
    # envs = generate_n_envs(num_envs=1000, num_starts=20)
    parent_folder = os.path.join(get_project_path(), "data", "office_1000", "test", "random_envs")
    image_save_parent_folder = os.path.join(get_project_path(), "data", "office_1000", "test", "random_envs_images")
    generate_n_envs(num_envs=200, num_starts=20, parent_folder=parent_folder,
                    image_save_folder=image_save_parent_folder)

    # store_envs(envs, parent_folder)


if __name__ == '__main__':
    # test_read_envs()
    run_store_envs()
