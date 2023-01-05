#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 7/2/22 12:43 PM 
    @Description    :
        
===========================================
"""
import matplotlib.pyplot as plt
import numpy as np

from visualize_utilities.plot_env import save_image

FONT_SIZE = 16


def plot_motion_env(points, section_start_positions, parent_dir, save_name):
    for i in range(len(section_start_positions)):
        plt.text(section_start_positions[i, 0], section_start_positions[i, 1] + 0.05, 'P' + str(i))
    plt.scatter(points[:, 0], points[:, 1], c='#83A3FF')
    plt.plot(points[:, 0], points[:, 1], color='#ED2F49')
    plt.scatter([points[0, 0]], [points[0, 1]], c='r', marker='o', s=200)
    plt.scatter([points[-1, 0]], [points[-1, 1]], c='b', marker='o', s=200)
    plt.title("Training environment for motion agent", fontsize=FONT_SIZE)
    plt.ylabel("y", fontsize=FONT_SIZE)
    plt.xlabel("x", fontsize=FONT_SIZE)
    plt.tight_layout()
    plt.grid()
    save_image(plt, parent_dir=parent_dir, save_name=save_name)
    plt.show()


def plot_speed_distance_curve(distances_list, speeds_list, y_label, parent_dir, save_name):
    """

    :param save_name:
    :param distances_list:
    :param speeds_list:
    :param y_label:
    :param title:
    :param parent_dir:
    :return:
    """
    new_speeds_list = []
    new_distances_list = []
    for distances, speeds in zip(distances_list, speeds_list):
        length = min(len(speeds), len(distances))
        new_speeds = np.array(speeds)[:length]
        new_densities = np.array(distances)[:length]
        new_speeds_list.extend(new_speeds)
        new_distances_list.extend(new_densities)

    distances = np.array(new_distances_list)
    speeds = np.array(new_speeds_list)

    distances_outliers1 = distances[np.logical_and(speeds < 0.05, distances > 0.05)]
    speeds_outliers1 = speeds[np.logical_and(speeds < 0.05, distances > 0.05)]
    distances_outliers2 = distances[np.logical_and(speeds >= 0.05, distances > 0.15)]
    speeds_outliers2 = speeds[np.logical_and(speeds >= 0.05, distances > 0.15)]
    distances_normal = distances[np.logical_and(1- np.logical_and(speeds < 0.05, distances > 0.05), 1- np.logical_and(speeds >= 0.05, distances > 0.15))]
    speeds_normal = speeds[np.logical_and(1- np.logical_and(speeds < 0.05, distances > 0.05), 1- np.logical_and(speeds >= 0.05, distances > 0.15))]
    parameter = np.polyfit(distances_normal, speeds_normal, 1, rcond=0.05)
    x = np.linspace(min(distances_normal), max(distances_normal))
    # y = parameter[0] * x ** 2 + parameter[1] * x + parameter[2]
    y = parameter[0] * x + parameter[1]

    plt.scatter(distances_normal, speeds_normal, c='#83A3ff55', label="Normal", s=50)
    plt.scatter(distances_outliers1, speeds_outliers1, c='#FF000055', label="outlier1", s=50)
    # plt.scatter(distances_outliers2, speeds_outliers2, c='#D66D2055', label="outlier2", s=50)
    plt.scatter(distances_outliers2, speeds_outliers2, c='#9F49D655', label="outlier2", s=50)

    plt.plot(x, y, color='g')
    plt.legend()
    # plt.title(title, fontsize=FONT_SIZE)
    plt.ylabel(y_label, fontsize=FONT_SIZE)
    plt.xlabel("Waypoint distance to robot (m)", fontsize=FONT_SIZE)
    plt.tight_layout()
    plt.grid()
    save_image(plt, parent_dir=parent_dir, save_name=save_name)
    plt.show()


def plot_real_speed_curve(real_speeds, parent_dir, save_name):
    """

    :param speeds:
    :param save_name:
    :param title:
    :param parent_dir:
    :return:
    """
    x = np.arange(0, len(real_speeds))
    plt.plot(x, np.array(real_speeds), color='#83A3ff')

    plt.ylabel("Real velocity (m/s)", fontsize=FONT_SIZE)
    plt.xlabel("Time step", fontsize=FONT_SIZE)
    # plt.legend()
    plt.tight_layout()
    plt.grid()
    save_image(plt, parent_dir=parent_dir, save_name=save_name)
    plt.show()


def plot_planned_speed_curve(planned_speeds, parent_dir, save_name):
    """

    :param speeds:
    :param save_name:
    :param title:
    :param parent_dir:
    :return:
    """
    x = np.arange(0, len(planned_speeds))
    plt.plot(x, np.array(planned_speeds), color='#D66D20')

    plt.ylabel("Planned velocity (m/s)", fontsize=FONT_SIZE)
    plt.xlabel("Time step", fontsize=FONT_SIZE)
    # plt.legend()
    plt.tight_layout()
    plt.grid()
    save_image(plt, parent_dir=parent_dir, save_name=save_name)
    plt.show()


def plot_waypoint_distance_to_robot(distances, parent_dir, save_name):
    """

    :param speeds:
    :param save_name:
    :param title:
    :param parent_dir:
    :return:
    """
    x = np.arange(0, len(distances))
    plt.plot(x, np.array(distances), color='g')
    plt.ylabel("Waypoint distance to robot (m)", fontsize=FONT_SIZE)
    plt.xlabel("Time step", fontsize=FONT_SIZE)
    plt.tight_layout()
    plt.grid()
    save_image(plt, parent_dir=parent_dir, save_name=save_name)
    plt.show()


def plot_waypoint_spacing_distance(waypoint_spacings, parent_dir, save_name):
    """

    :param speeds:
    :param save_name:
    :param title:
    :param parent_dir:
    :return:
    """
    x = np.arange(0, len(waypoint_spacings))
    plt.plot(x, np.array(waypoint_spacings), color='r')
    plt.ylabel("Waypoint spacing (m)", fontsize=FONT_SIZE)
    plt.xlabel("Time step", fontsize=FONT_SIZE)
    plt.tight_layout()
    plt.grid()
    save_image(plt, parent_dir=parent_dir, save_name=save_name)
    plt.show()
