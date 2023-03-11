import os
import pickle
from multiprocessing import Pool
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

from utils.compute_u_forces import compute_u_force, display_u_force
from utils.fo_utility import get_project_path, get_p2v_path, get_office_evacuation_path


def compute_v_force(occupancy_map, geo_distance_map, k1=0.01, k2=0.005, k3=1):
    force_v = k1 * geo_distance_map + k2 * geo_distance_map ** 2 + k3
    force_v[geo_distance_map > 39] = 11

    force_vx = np.zeros_like(force_v)
    force_vx = force_vx.astype(float)
    for i in range(0, force_v.shape[0]):
        for j in range(0, force_v.shape[1]):
            if occupancy_map[i][j] == 1 or occupancy_map[i][j - 1] == 1:
                continue
            force_vx[i][j] = np.sign(geo_distance_map[i][j - 1] - geo_distance_map[i][j]) * force_v[i][j]

    force_vy = np.zeros_like(force_v).astype(float)
    force_vy = force_vy.astype(float)

    for i in range(0, force_v.shape[0]):
        for j in range(0, force_v.shape[1]):
            if occupancy_map[i][j] == 1 or occupancy_map[i - 1][j] == 1:
                continue
            force_vy[i][j] = np.sign(geo_distance_map[i - 1][j] - geo_distance_map[i][j]) * force_v[i][j]

    return force_vx, force_vy, force_v


def compute_v_force_const(occupancy_map, geo_distance_map):
    """

    Args:
        occupancy_map: 占据栅格图
        geo_distance_map: 测地距离图

    Returns:

    """
    force_vx = np.zeros_like(occupancy_map).astype(float)
    force_vy = np.zeros_like(occupancy_map).astype(float)
    force_v = np.ones_like(occupancy_map).astype(float) * 10

    for i in range(0, force_v.shape[0]):
        for j in range(0, force_v.shape[1]):
            if occupancy_map[i][j] == 1 or occupancy_map[i][j - 1] == 1:
                continue
            force_vx[i][j] = np.sign(geo_distance_map[i][j - 1] - geo_distance_map[i][j]) * force_v[i][j]

    for i in range(0, force_v.shape[0]):
        for j in range(0, force_v.shape[1]):
            if occupancy_map[i][j] == 1 or occupancy_map[i - 1][j] == 1:
                continue
            force_vy[i][j] = np.sign(geo_distance_map[i - 1][j] - geo_distance_map[i][j]) * force_v[i][j]

    return force_vx, force_vy, force_v


def display_v_images(force_vx, force_vy, force_v):
    L1, L2 = force_v.shape
    plt.figure(30)
    plt.imshow(np.sqrt(force_vx ** 2 + force_vy ** 2), cmap='gray')
    plt.title('Convolved Map')
    plt.show()

    XX, YY = np.meshgrid(np.arange(1, L1 + 1), np.arange(1, L2 + 1))
    plt.figure(20)
    plt.quiver(XX, YY, force_vx, force_vy)
    plt.title('Convolved Map Vectors')
    plt.show()

    plt.figure(11)
    plt.imshow(force_vx, cmap='gray')
    plt.title('X-Convolved Map')
    plt.show()

    plt.figure(12)
    plt.imshow(force_vy, cmap='gray')
    plt.title('Y-Convolved Map')
    plt.show()


def save_v_images(force_vx, force_vy, force_v, save_folder, i, goal):
    plt.figure(30)
    plt.imshow(np.sqrt(force_vx ** 2 + force_vy ** 2), cmap='gray')
    plt.title('Convolved Map')
    plt.savefig(os.path.join(save_folder, "env_{}_{}_convolved_map.png".format(i, goal)))


def compute_v_force_by_path(env_path, geo_path):
    occupancy_map, _, _, _ = pickle.load(open(env_path, "rb"))
    geo_dict_dict = pickle.load(open(geo_path, "rb"))
    v_map_dict: Dict = {}
    for goal in geo_dict_dict.keys():
        geo_distance_map = np.zeros_like(occupancy_map).astype(float)
        geo_distance_dict: Dict = geo_dict_dict[goal]

        for key in geo_distance_dict.keys():
            geo_distance_map[key] = geo_distance_dict[key]

        force_vx, force_vy, force_v = compute_v_force_const(occupancy_map, geo_distance_map)
        v_map_dict[goal] = [force_vx, force_vy, force_v]
    return v_map_dict


def compute_v_forces(folder_name, phase, indexes):
    env_folder = os.path.join(get_office_evacuation_path(), folder_name, phase, "envs")
    geo_folder = os.path.join(get_office_evacuation_path(), folder_name, phase, "geodesic_distance")

    v_folder = os.path.join(get_office_evacuation_path(), folder_name, phase, "v_forces")
    v_image_folder = os.path.join(get_office_evacuation_path(), folder_name, phase, "v_image_forces")

    if not os.path.exists(v_folder):
        os.makedirs(v_folder)
    if not os.path.exists(v_image_folder):
        os.makedirs(v_image_folder)

    env_name_template = "env_{}.pkl"
    #
    env_paths = [os.path.join(env_folder, env_name_template.format(ind)) for ind in indexes]
    geo_paths = [os.path.join(geo_folder, env_name_template.format(ind)) for ind in indexes]
    v_paths = [os.path.join(v_folder, env_name_template.format(ind)) for ind in indexes]
    for i, ind in enumerate(indexes):
        env_path = env_paths[i]
        geo_path = geo_paths[i]
        v_path = v_paths[i]
        print("Processing path:{}".format(env_path))
        v_map_dict = compute_v_force_by_path(env_path, geo_path)

        # 保存force
        pickle.dump(v_map_dict, open(v_path, "wb"))

        # for goal in geo_distance_map_dict.keys():
        #     force_vx, force_vy, force_v = geo_distance_map_dict[goal]

        # 显示u force
        # display_v_images(force_vx, force_vy, force_v)
        # save_v_images(force_vx, force_vy, force_v, save_folder=v_image_folder, i=i, goal=goal)

        # 保存 u force 图片

    return


def multi_process(folder_name, phase, indexes):
    print('Parent process %s.' % os.getpid())
    # 进程数量
    num_process = 5
    p = Pool(num_process)
    num_batch = int((indexes[-1] + 1 - indexes[0]) / num_process)
    split_env_indexes = [[indexes[0] + i * num_batch, indexes[0] + (i + 1) * num_batch] for i in range(num_process)]
    for start_index, end_index in split_env_indexes:
        split_indexes = [i for i in range(start_index, end_index)]
        p.apply_async(compute_v_forces,
                      args=(folder_name, phase, split_indexes,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')


if __name__ == '__main__':
    folder_name = "sg_walls"
    phase = "train"
    # 要处理从哪个到哪个文件
    indexes = [i for i in range(1200, 1500)]
    # compute_v_forces(folder_name, phase, indexes)

    multi_process(folder_name, phase, indexes)

    print()
