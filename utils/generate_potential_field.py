import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np

from utils.fo_utility import get_project_path


# 定义地图大小

def gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y):
    return np.exp(-((x - mu_x) ** 2 / (2 * sigma_x ** 2) + (y - mu_y) ** 2 / (2 * sigma_y ** 2)))


def compute_all_gaussian_distributions(h, w):
    xs = [i for i in range(h)]
    ys = [i for i in range(w)]
    xx, yy = np.meshgrid(xs, ys)
    sigma_x = 5
    sigma_y = 5
    z = {}
    for mu_x in range(h):
        for mu_y in range(w):
            pos = (mu_x, mu_y)
            if pos not in z.keys():
                z[pos] = gaussian(xx, yy, mu_x, mu_y, sigma_x, sigma_y)
    return z


def save_all_gaussian_distributions():
    # all_gaussian_distributions = compute_all_gaussian_distributions(h=70, w=70)
    # random_env_parent_folder = os.path.join(get_project_path(), "data", "office_1000", "train", "random_envs")
    gaussians_folder = os.path.join(get_project_path(), "data", "office_1000", "train", "gaussians")
    if not os.path.exists(gaussians_folder):
        os.makedirs(gaussians_folder)
    gaussian = compute_all_gaussian_distributions(70, 70)
    out_path = os.path.join(get_project_path(), "data", "gaussian.pkl")
    pickle.dump(gaussian, open(out_path, "wb"))


def read_gaussian_distributions():
    path = os.path.join(get_project_path(), "data", "gaussian.pkl")
    gaussians = open(path, "rb")


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
                jj, ii = np.where(occupancy_map)
                distances = np.linalg.norm(np.array([y - jj, x - ii]).transpose((1, 0)), axis=1)
                potential_field[y][x] = -np.min(distances, axis=0)
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
    # random_env_parent_folder = os.path.join(get_project_path(), "data", "office_1000", "train", "random_envs")
    # path = os.path.join(random_env_parent_folder, "env_0.pkl")
    # file = open(path, "rb")
    # occupancy_map, _, _ = pickle.load(file)
    # potential_field = create_potential_field(occupancy_map=occupancy_map)
    # # 输出 potential field
    # plt.imshow(potential_field)
    # plt.show()
    save_all_gaussian_distributions()
