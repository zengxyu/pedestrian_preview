#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : pedestrian_preview 
    @Author  : Xiangyu Zeng
    @Date    : 3/1/23 8:29 PM 
    @Description    :
        
===========================================
"""
import os.path
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from utils.fo_utility import get_project_path, get_data_path
import scipy


def compute_u_kernel(H, W):
    """
    计算u kernel
    Args:
        H:
        W:

    Returns:

    """
    kernel_x = np.zeros((H, W))
    kernel_y = np.zeros((H, W))
    kernel_data = np.zeros((H, W))

    cx = H // 2
    cy = W // 2

    sigma_1 = 3

    ind_i = np.arange(1, H)
    ind_j = np.arange(1, W)
    for i in ind_i:
        for j in ind_j:
            x = i - cx
            y = j - cy
            a = x / abs(x) if x != 0 else 0
            b = y / abs(y) if y != 0 else 0
            L = np.linalg.norm([x, y])
            theta = np.arctan(x / y) if y != 0 else np.pi / 2 * np.sign(x)
            if (a <= 0 and b <= 0) or (a >= 0 and b <= 0) or (x == 0 and y < 0):
                kernel_x[i, j] = -np.exp(-L / sigma_1) * np.cos(theta)
                kernel_y[i, j] = -np.exp(-L / sigma_1) * np.sin(theta)
            else:
                kernel_x[i, j] = np.exp(-L / sigma_1) * np.cos(theta)
                kernel_y[i, j] = np.exp(-L / sigma_1) * np.sin(theta)
            kernel_data[i, j] = np.linalg.norm([kernel_x[i, j], kernel_y[i, j]])

    kernel_x[cx, cy] = 0
    kernel_y[cx, cy] = 0

    return kernel_x, kernel_y, kernel_data


def save_kernel(win_x, win_y, win_data, save=False, save_dir=''):
    save_path = os.path.join(save_dir, "kernel.pkl")
    if save:
        pickle.dump([win_x, win_y, win_data], open(save_path, "wb"))


def display_kernel(win_x, win_y, win_data):
    plt.figure(1)
    plt.imshow(win_data, cmap='gray')
    plt.title('Window Data')
    plt.savefig("win_data.png")
    h, w = win_data.shape
    x, y = np.meshgrid(range(h), range(w))
    plt.figure(2)
    plt.quiver(x, y, win_x, win_y)
    # plt.axis([-1, 1, -1, 1])
    plt.title('Window Vectors')
    plt.savefig("win_vec.png")
    plt.show()


def compute_u_force(kernel_x, kernel_y, occupancy_map):
    """
    为occupancy map 计算 u force
    Args:
        kernel_x:
        kernel_y:
        occupancy_map:

    Returns:

    """
    force_x = convolve2d(occupancy_map, kernel_x, mode='same')
    force_y = convolve2d(occupancy_map, kernel_y, mode='same')
    force_x[occupancy_map == 1] = np.max(force_x)
    force_y[occupancy_map == 1] = np.max(force_y)
    force = np.sqrt(force_x ** 2 + force_y ** 2)

    return force_x, force_y, force


def display_u_force(force_x, force_y, force):
    L1, L2 = force_x.shape
    plt.figure(30)
    plt.imshow(np.sqrt(force_x ** 2 + force_y ** 2), cmap='gray')
    plt.title('Convolved Map')
    plt.show()

    XX, YY = np.meshgrid(np.arange(1, L1 + 1), np.arange(1, L2 + 1))
    plt.figure(20)
    plt.quiver(XX, YY, force_x, force_y)
    plt.title('Convolved Map Vectors')
    plt.show()

    plt.figure(11)
    plt.imshow(force_x, cmap='gray')
    plt.title('X-Convolved Map')
    plt.show()

    plt.figure(12)
    plt.imshow(force_y, cmap='gray')
    plt.title('Y-Convolved Map')
    plt.show()


def save_u_force(force_x, force_y, force, save_folder):
    """
    保存u force
    Args:
        force_x:
        force_y:
        force:
        save_folder:

    Returns:

    """
    L1, L2 = force_x.shape
    plt.figure(30)
    plt.imshow(np.sqrt(force_x ** 2 + force_y ** 2), cmap='gray')
    plt.title('Convolved Map')
    plt.savefig(os.path.join(save_folder, "convolved_map.png"))

    XX, YY = np.meshgrid(np.arange(1, L1 + 1), np.arange(1, L2 + 1))
    plt.figure(20)
    plt.quiver(XX, YY, force_x, force_y)
    plt.title('Convolved Map Vectors')
    plt.savefig(os.path.join(save_folder, "convolved_map_vector.png"))

    plt.figure(11)
    plt.imshow(force_x, cmap='gray')
    plt.title('X-Convolved Map')
    plt.savefig(os.path.join(save_folder, "x_convolved_map.png"))

    plt.figure(12)
    plt.imshow(force_y, cmap='gray')
    plt.title('Y-Convolved Map')
    plt.savefig(os.path.join(save_folder, "y_convolved_map.png"))


def compute_u_forces(env_folder, u_folder, u_image_folder):
    """

    Args:
        env_folder: 环境文件夹
        u_folder: uv文件夹

    Returns:

    """
    kernel_path = os.path.join(get_project_path(), "data", "kernel.pkl")
    kernel, kernel_x, kernel_y = pickle.load(open(kernel_path, "rb"))
    # 环境folder
    env_folder = os.path.join(get_project_path(), "data", parent_folder, phase, "envs")

    uv_folder = os.path.join(get_project_path(), "data", parent_folder, phase, "uv_forces")
    if not os.path.exists(uv_folder):
        os.makedirs(uv_folder)

    filenames = os.listdir(env_folder)
    indexes = [i for i in range(0, 1200)]
    env_name_template = "env_{}"
    env_paths = [os.path.join(env_folder, env_name_template.format(ind) + ".pkl") for ind in indexes]
    u_paths = [os.path.join(u_folder, env_name_template.format(ind) + ".pkl") for ind in indexes]
    u_image_paths = [os.path.join(u_image_folder, env_name_template.format(ind) + ".png") for ind in indexes]
    for i in range(len(filenames)):
        env_path = env_paths[i]
        u_path = u_paths[i]
        u_image_path = u_image_paths[i]
        print("Processing path:{}".format(env_path))
        occupancy_map, _, _ = pickle.load(open(env_path, "rb"))
        force_x, force_y, force = compute_u_force(kernel_x, kernel_y, occupancy_map)
        # 保存force
        pickle.dump([force_x, force_y, force], open(u_path, "wb"))
        # 显示u force
        display_u_force(force_x, force_y, force)
        # 保存 u force 图片
        save_u_force(force_x, force_y, force, save_folder=u_image_path)
        print()


def compute_p2v_forces():
    p2v_occupancy_map_folder = os.path.join(get_data_path(), "p2v", "env1", "occupancy_map")
    p2v_occupancy_map_path = os.path.join(p2v_occupancy_map_folder, "occupancy_map.pkl")
    p2v_occupancy_map_file = open(p2v_occupancy_map_path, "rb")
    # 读取P2V地图
    p2v_occupancy_map = pickle.load(p2v_occupancy_map_file)
    # 计算场
    # 加载卷积核
    win_path = os.path.join(get_project_path(), "data", "win.pkl")
    kernel_x, kernel_y, kernel_data = pickle.load(open(win_path, "rb"))
    force_x, force_y, force = compute_u_force(kernel_x, kernel_y, p2v_occupancy_map)
    # 保存force
    display_u_force(force_x, force_y, force)
    save_u_force(force_x, force_y, force, save_folder=u_image_path)

    return


if __name__ == '__main__':
    # 计算kernel
    u_kernel_x, u_kernel_y, u_kernel = compute_u_kernel(H=31, W=31)
    # 保存kernel
    save = False
    save_dir = os.path.join(get_project_path(), "data")
    save_kernel(u_kernel_x, u_kernel_y, u_kernel)
    display_kernel(u_kernel_x, u_kernel_y, u_kernel)

    # 为文件夹下所有图计算向量场
    print()
    parent_folder = "office_1500_goal_outdoor"
    phase = "test"

    compute_u_forces(env_folder=env_folder, u_folder=uv_folder)
