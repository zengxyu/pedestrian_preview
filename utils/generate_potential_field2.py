import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from utils.fo_utility import get_project_path


def generate_static_potential_field(occupancy_map, goal_position, k_att=1, k_rep=5, d_rep=0.5):
    """
    生成基于静态障碍物和目标点的组合法的potential field

    参数:
    occupancy_map -- 占据栅格地图，1表示障碍物，0表示可行区域
    goal_position -- 目标点的位置，格式为 (x, y)
    k_att -- 引力系数，默认为1
    k_rep -- 斥力系数，默认为5
    d_rep -- 斥力距离，默认为0.5

    返回值:
    potential_field -- 生成的potential field，大小与occupancy_map相同
    """
    # 获取地图尺寸
    height, width = occupancy_map.shape

    # 初始化potential field
    potential_field = np.zeros_like(occupancy_map, dtype=float)

    # 计算引力
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    dx = x - goal_position[0]
    dy = y - goal_position[1]
    potential_field += 0.5 * k_att * (dx ** 2 + dy ** 2)
    plt.imshow(potential_field)
    plt.show()
    # 计算斥力
    repulsive_field = np.zeros_like(occupancy_map, dtype=float)
    repulsive_field[occupancy_map == 1] = 1
    for i in range(max(0, int(goal_position[1] - d_rep)), min(height, int(goal_position[1] + d_rep) + 1)):
        for j in range(max(0, int(goal_position[0] - d_rep)), min(width, int(goal_position[0] + d_rep) + 1)):
            repulsive_field[i, j] = k_rep * (
                    (1 / d_rep - 1 / np.sqrt((i - goal_position[1]) ** 2 + (j - goal_position[0]) ** 2)) ** 2)
    potential_field += repulsive_field
    plt.imshow(repulsive_field)
    plt.show()
    # 剔除障碍物位置的影响
    potential_field[occupancy_map == 1] = 100

    return potential_field


def generate_dynamic_potential_field(occupancy_map, obstacle_position, k_rep=10, d_rep=0.5):
    """
    生成基于动态障碍物的斥力potential field

    参数:
    occupancy_map -- 占据
    返回值:
    potential_field -- 生成的potential field，大小与occupancy_map相同
    """
    # 获取地图尺寸
    height, width = occupancy_map.shape

    # 初始化potential field
    potential_field = np.zeros_like(occupancy_map, dtype=float)

    # 计算斥力
    repulsive_field = np.zeros_like(occupancy_map, dtype=float)
    repulsive_field[occupancy_map == 1] = np.inf
    for i in range(max(0, int(obstacle_position[1] - d_rep)), min(height, int(obstacle_position[1] + d_rep) + 1)):
        for j in range(max(0, int(obstacle_position[0] - d_rep)), min(width, int(obstacle_position[0] + d_rep) + 1)):
            repulsive_field[i, j] = k_rep * ((1 / d_rep - 1 / np.sqrt(
                (i - obstacle_position[1]) ** 2 + (j - obstacle_position[0]) ** 2)) ** 2)
    potential_field += repulsive_field

    return potential_field


def generate_total_potential_field(occupancy_map, goal_position, obstacle_positions, k_att=1, k_rep_static=5,
                                   k_rep_dynamic=10, d_rep=0.5):
    """
    生成基于静态障碍物和动态障碍物的组合potential field

    参数:
    occupancy_map -- 占据栅格地图，1表示障碍物，0表示可行区域
    goal_position -- 目标点的位置，格式为 (x, y)
    obstacle_positions -- 动态障碍物的位置列表，每个位置格式为 (x, y)
    k_att -- 引力系数，默认为1
    k_rep_static -- 静态障碍物的斥力系数，默认为5
    k_rep_dynamic -- 动态障碍物的斥力系数，默认为10
    d_rep -- 斥力距离，默认为0.5

    返回值:
    potential_field -- 生成的potential field，大小与occupancy_map相同
    """
    # 生成静态障碍物的potential field
    static_potential_field = generate_static_potential_field(occupancy_map, goal_position, k_att, k_rep_static, d_rep)
    plt.imshow(static_potential_field)
    plt.show()
    # 初始化total potential field
    total_potential_field = np.copy(static_potential_field)

    # 添加动态障碍物的影响
    for obstacle_position in obstacle_positions:
        dynamic_potential_field = generate_dynamic_potential_field(occupancy_map, obstacle_position, k_rep_dynamic,
                                                                   d_rep)
        plt.imshow(dynamic_potential_field)
        plt.show()
        total_potential_field += dynamic_potential_field

    return total_potential_field


if __name__ == '__main__':
    # 定义占据栅格地图（1 表示占据，0 表示空闲）
    random_env_parent_folder = os.path.join(get_project_path(), "data", "office_1000", "train", "random_envs")
    path = os.path.join(random_env_parent_folder, "env_0.pkl")
    file = open(path, "rb")
    occupancy_map, _, _ = pickle.load(file)
    goal_position = np.array([30, 30])
    obstacle_positions = np.array([[10, 10], [5, 10], [10, 5]])
    potential_field = generate_total_potential_field(occupancy_map=occupancy_map, goal_position=goal_position, obstacle_positions=obstacle_positions)


