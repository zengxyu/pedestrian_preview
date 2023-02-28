import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np

from utils.fo_utility import get_project_path


# 定义地图大小


def create_potential_field(occupancy_map):
    # 创建空白 potential field
    potential_field = np.zeros(occupancy_map.shape)
    GRID_HEIGHT = occupancy_map.shape[0]
    GRID_WIDTH = occupancy_map.shape[1]
    # 计算 potential field
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if occupancy_map[y][x] == 0:
                # 如果该格子是空闲的，计算该格子到最近的占据格子的距离并取相反数
                min_distance = float('inf')
                for j in range(GRID_HEIGHT):
                    for i in range(GRID_WIDTH):
                        if occupancy_map[j][i] == 1:
                            distance = np.sqrt((x - i) ** 2 + (y - j) ** 2)
                            if distance < min_distance:
                                min_distance = distance
                potential_field[y][x] = -min_distance
            else:
                # 如果该格子是占据的，将其 potential field 值设置为一个极小值，表示该区域不可通过。
                potential_field[y][x] = float(0)

    # # 平滑 potential field
    # for i in range(10):
    #     for y in range(1, GRID_HEIGHT - 1):
    #         for x in range(1, GRID_WIDTH - 1):
    #             if occupancy_map[y][x] == 0:
    #                 potential_field[y][x] = (potential_field[y - 1][x] + potential_field[y + 1][x] +
    #                                          potential_field[y][x - 1] + potential_field[y][x + 1]) / 4
    return potential_field


if __name__ == '__main__':
    # 定义占据栅格地图（1 表示占据，0 表示空闲）
    random_env_parent_folder = os.path.join(get_project_path(), "data", "office_1000", "train", "random_envs")
    path = os.path.join(random_env_parent_folder, "env_0.pkl")
    file = open(path, "rb")
    occupancy_map, _, _ = pickle.load(file)
    potential_field = create_potential_field(occupancy_map=occupancy_map)
    # 输出 potential field
    plt.imshow(potential_field)
    plt.show()
