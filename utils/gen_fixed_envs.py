import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from environment.gen_scene.gen_office_map import create_office_map
from environment.gen_scene.gen_u_shape_map import create_u_shape_map
from environment.gen_scene.sampler_mapping import get_sampler_class
from environment.gen_scene.world_generator import get_world_config
from utils.config_utility import read_yaml
from utils.fo_utility import get_project_path
from utils.image_utility import dilate_image


def generate_env(num_starts, world_name):
    # read specified world config
    worlds_config = read_yaml(os.path.join(get_project_path(), "configs"), "worlds_config.yaml")
    world_config = get_world_config(worlds_config, world_name)

    # read specified sampler config
    agent_sampler_name = "agent_sg_opposite_baffle_sampler2"
    samplers_config = read_yaml(os.path.join(get_project_path(), "configs"), "samplers_config.yaml")
    agent_sampler_config = samplers_config[agent_sampler_name]
    agent_sg_sampler_class = get_sampler_class(agent_sampler_config["sampler_name"])
    agent_sg_sampler_params = agent_sampler_config["sampler_params"]

    # create world occupancy map
    # create_u_tunnel_in_office_map
    if world_name == "office":
        occupancy_map = create_office_map(world_config)
    elif world_name == "u_shape":
        occupancy_map = create_u_shape_map(world_config)
    else:
        raise NotImplementedError

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


def display_and_save_only_env(occupancy_map, save, save_path):
    plt.imshow(occupancy_map)
    if save:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()


def generate_n_envs():
    # 确定生成场景的名字
    world_name = "u_shape"
    # 生成的场景的数量或者索引号码
    indexes = [a for a in range(200, 240)]
    # 每个场景起点个数
    num_starts = 20

    # 生成的场景保存位置
    parent_folder = "office_1500"
    envs_folder = os.path.join(get_project_path(), "data", parent_folder, "test", "envs")
    envs_images_folder = os.path.join(get_project_path(), "data", parent_folder, "test", "envs_images")

    # 如果文件夹不存在，创建文件夹
    if not os.path.exists(envs_folder):
        os.makedirs(envs_folder)
    if not os.path.exists(envs_images_folder):
        os.makedirs(envs_images_folder)

    # 场景名字模板
    env_name_template = "env_{}.pkl"
    image_name_template = "env_{}.png"

    i = 0
    while i < len(indexes):
        index = indexes[i]
        # 生成的occupancy map保存名字
        save_file_name = env_name_template.format(index)
        # 生成的occupancy map对应图像保存名字
        image_save_file_name = image_name_template.format(index)
        # 保存的位置
        save_path = os.path.join(envs_folder, save_file_name)
        image_save_path = os.path.join(envs_images_folder, image_save_file_name)
        print("Generating {}-th office...".format(index))
        occupancy_map, starts, ends, sample_success = generate_env(num_starts=num_starts, world_name=world_name)

        if sample_success:
            env = [occupancy_map, starts, ends, world_name, ]
            print("Save env {} to {} ... ".format(save_file_name, save_path))
            pickle.dump(env, open(save_path, 'wb'))
            display_and_save(occupancy_map, starts, ends, save=True, save_path=image_save_path)

            print("Save done!")
            i += 1


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
    parent_folder = os.path.join(get_project_path(), "data", "envs")
    envs = read_envs(parent_folder)
    display_and_save(*envs[0])


if __name__ == '__main__':
    generate_n_envs()
