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
from utils.fo_utility import get_project_path
import scipy


def map_value(H, W, save=False, save_dir=''):
    win_x = np.zeros((H, W))
    win_y = np.zeros((H, W))
    win_data = np.zeros((H, W))

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
                win_x[i, j] = -np.exp(-L / sigma_1) * np.cos(theta)
                win_y[i, j] = -np.exp(-L / sigma_1) * np.sin(theta)
            else:
                win_x[i, j] = np.exp(-L / sigma_1) * np.cos(theta)
                win_y[i, j] = np.exp(-L / sigma_1) * np.sin(theta)
            win_data[i, j] = np.linalg.norm([win_x[i, j], win_y[i, j]])

    win_x[cx, cy] = 0
    win_y[cx, cy] = 0
    save_path = os.path.join(save_dir, "win.pkl")
    if save:
        pickle.dump([win_data, win_x, win_y], open(save_path, "wb"))
        # plt.savefig(win_data, win_x, win_y)

    return win_data, win_x, win_y


def compute_u(win_x, win_y, occupancy_map):
    map_x = convolve2d(occupancy_map, win_x, mode='same')
    map_y = convolve2d(occupancy_map, win_y, mode='same')
    map_x[occupancy_map == 1] = np.max(map_x)
    map_y[occupancy_map == 1] = np.max(map_y)

    return map_x, map_y


def display_windows(win_data, win_x, win_y):
    plt.figure(1)
    plt.imshow(win_data, cmap='gray')
    plt.title('Window Data')
    h, w = win_data.shape
    x, y = np.meshgrid(h, w)
    plt.figure(2)
    plt.quiver(x, y, win_x, win_y)
    plt.axis([-1, 1, -1, 1])
    plt.title('Window Vectors')


def display_and_save_u(map_x, map_y, save, save_folder):
    L1, L2 = map_x.shape
    plt.figure(30)
    plt.imshow(np.sqrt(map_x ** 2 + map_y ** 2), cmap='gray')
    plt.title('Convolved Map')
    if save:
        plt.savefig(save_folder)
    XX, YY = np.meshgrid(np.arange(1, L1 + 1), np.arange(1, L2 + 1))
    plt.figure(20)
    plt.quiver(XX, YY, map_x, map_y)
    plt.title('Convolved Map Vectors')

    if save:
        plt.savefig(save_folder)

    plt.figure(11)
    plt.imshow(map_x, cmap='gray')
    plt.title('X-Convolved Map')

    if save:
        plt.savefig(save_folder)

    plt.figure(12)
    plt.imshow(map_y, cmap='gray')
    plt.title('Y-Convolved Map')
    if save:
        plt.savefig(save_folder)
    plt.show()


def compute_all_u(folder, save_folder):
    win_path = os.path.join(get_project_path(), "data", "win.pkl")
    win_data, win_x, win_y = pickle.load(open(win_path, "rb"))
    # win_data, win_x, win_y = map_value(H=31, W=31)
    filenames = os.listdir(folder)
    paths = [os.path.join(folder, filename) for filename in filenames]
    save_paths = [os.path.join(save_folder, filename) for filename in filenames]

    for path, save_path in zip(paths, save_paths):
        print("Processing path:{}".format(path))
        occupancy_map, _, _ = pickle.load(open(path, "rb"))
        map_x, map_y = compute_u(win_x, win_y, occupancy_map)
        map_ = np.sqrt(map_x ** 2 + map_y ** 2)
        pickle.dump([map_x, map_y, map_], open(save_path, "wb"))
        # denominator = np.sqrt(map_x ** 2 + map_y ** 2)
        # map_x = map_x / denominator
        # map_y = map_y / denominator
        # map_x = map_x
        # display_u(map_x, map_y)


if __name__ == '__main__':
    # map_value(H=31, W=31, save=True, save_dir=os.path.join(get_project_path(), "data"))
    random_env_folder = os.path.join(get_project_path(), "data", "office_1000", "train", "random_envs")
    save_folder = os.path.join(get_project_path(), "data", "office_1000", "train", "potential_maps")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    compute_all_u(folder=random_env_folder, save_folder=save_folder)
