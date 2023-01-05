#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 6/28/22 10:36 AM 
    @Description    :
        
===========================================
"""
import pickle
import time
import os
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Rectangle

from utils.fo_utility import get_project_path


def add_map_patch_to_ax(ax, map):
    ax.set_xlim([0, map.shape[0]])
    ax.set_ylim([0, map.shape[1]])
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i, j]:
                rec = Rectangle((i, j), width=1, height=1, color='#83A3ff')
                ax.add_patch(rec)
            # else:
            #     rec = Rectangle((i, j), width=1, height=1, edgecolor='#FBFFE3', facecolor='w')
            #     ax.add_patch(rec)


def add_rectangle_patch(ax, point, width=1, height=1, color=None, facecolor=None):
    if isinstance(point, Point):
        rec = Rectangle((point.x, point.y), width, height, color=color, facecolor=facecolor)
        ax.add_patch(rec)
    else:
        rec = Rectangle((point[0], point[1]), width, height, color=color, facecolor=facecolor)
        ax.add_patch(rec)


def plot_occupancy_map(map, start_point, end_point, parent_dir="plot_occupancy_map", save_name=None):
    plt.figure(figsize=(5, 5))
    ax = plt.gca()

    add_map_patch_to_ax(ax, map)

    add_rectangle_patch(ax, start_point, 2, 2, facecolor='b')

    add_rectangle_patch(ax, end_point, 2, 2, facecolor='r')

    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()

    save_image(plt, parent_dir=parent_dir, save_name=save_name)

    return ax, plt


def plot_a_star_path(occupancy_map, start_point, end_point, path, close_set=None, open_set=None,
                     parent_dir=None, save_name=None):
    """
    画 a star path
    :param occupancy_map:
    :param start_point:
    :param end_point:
    :param path:
    :param close_set:
    :param open_set:
    :return:
    """

    plt.figure(figsize=(5, 5))
    ax = plt.gca()

    add_map_patch_to_ax(ax, occupancy_map)

    add_rectangle_patch(ax, start_point, 2, 2, facecolor='b')

    add_rectangle_patch(ax, end_point, 2, 2, facecolor='r')

    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()

    if close_set is not None:
        for point in close_set:
            add_rectangle_patch(ax, point, color="#dddddd")
    if open_set is not None:
        for point in open_set:
            add_rectangle_patch(ax, point, color="#cccccc")
    for point in path:
        add_rectangle_patch(ax, point, color="#FF9544")

    save_image(plt, parent_dir=parent_dir, save_name=save_name)


def plot_robot_trajectory(occupancy_map, start_point, end_point, path, deformed_path, robot_trajectory,
                          robot_velocities, obstacle_paths, grid_res, parent_dir=None, save_name=None):
    """
    形变路径，机器人轨迹, 机器人速度(速度控制路径的颜色值)
    :param occupancy_map:
    :param start_point:
    :param end_point:
    :param path:
    :param deformed_path:
    :param robot_trajectory:
    :param parent_dir:
    :param save_name:
    :return:
    """
    plt.figure(figsize=(5, 5))
    ax = plt.gca()

    add_map_patch_to_ax(ax, occupancy_map)

    add_rectangle_patch(ax, start_point, 2, 2, facecolor='b')

    add_rectangle_patch(ax, end_point, 2, 2, facecolor='r')

    ax.plot(path[:, 0], path[:, 1], label="A* planned path", c='orange', linewidth=3)

    # ax.plot(deformed_path[:, 0], deformed_path[:, 1], label="RL deformed path", c='k', linewidth=3)
    robot_velocities = robot_velocities[::4]
    robot_trajectory = robot_trajectory[::4]
    # color_alphas = (robot_velocities - np.min(robot_velocities)) / (np.max(robot_velocities) - np.min(robot_velocities))
    ct = ax.scatter(robot_trajectory[:, 0], robot_trajectory[:, 1], c=robot_velocities, cmap='viridis', s=5)

    plt.axis('equal')
    plt.axis('off')
    fc = plt.colorbar(ct, shrink=0.75)
    ax2 = fc.ax
    ax2.set_title("velocity (m/s)", fontsize=16, y=-0.1)
    plt.legend(loc="lower left", fontsize=16)

    plt.tight_layout()
    save_image(plt, parent_dir=parent_dir, save_name=save_name)
    plt.show()


def plot_lidar_scanning(occupancy_map, turtle_bot_position, hit_cartesian_positions, rays_from, rays_to, hit_vector,
                        parent_dir=None, save_name=None):
    """
    画 a star path
    :param occupancy_map:
    :param turtle_bot_position:
    :param lidar_scan_coordinates:
    :return:
    """

    plt.figure(figsize=(5, 5))
    ax = plt.gca()

    add_map_patch_to_ax(ax, occupancy_map)

    for hit, hit_cartesian_position, ray_from, ray_to in zip(hit_vector, hit_cartesian_positions, rays_from, rays_to):
        if hit:
            ax.plot([ray_from[0], hit_cartesian_position[0]], [ray_from[1], hit_cartesian_position[1]], color="r",
                    alpha=0.1)
        else:
            ax.plot([ray_from[0], ray_to[0]], [ray_from[1], ray_to[1]], color="g", alpha=0.1)

    add_rectangle_patch(ax, [turtle_bot_position[0] - 2, turtle_bot_position[1] - 2], 4, 4, color='k', facecolor='g')

    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()

    save_image(plt, parent_dir=parent_dir, save_name=save_name)

    plt.clf()

    """50束"""
    plt.figure(figsize=(5, 5))
    ax = plt.gca()

    add_map_patch_to_ax(ax, occupancy_map)

    count = 0
    for hit, hit_cartesian_position, ray_from, ray_to in zip(hit_vector, hit_cartesian_positions, rays_from, rays_to):
        count += 1
        if count % 20 == 0:
            if hit:
                ax.plot([ray_from[0], hit_cartesian_position[0]], [ray_from[1], hit_cartesian_position[1]], color="r",
                        alpha=0.8)
            else:
                ax.plot([ray_from[0], ray_to[0]], [ray_from[1], ray_to[1]], color="g", alpha=0.8)

    add_rectangle_patch(ax, [turtle_bot_position[0] - 2, turtle_bot_position[1] - 2], 4, 4, color='k', facecolor='g')

    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()

    save_image(plt, parent_dir=parent_dir, save_name=save_name + "_")


def plot_deformed_path(occupancy_map, path_list, legends, colors, parent_dir=None, save_name=None):
    plt.figure(figsize=(5, 5))
    ax = plt.gca()

    add_map_patch_to_ax(ax, occupancy_map)

    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    for path, legend, color in zip(path_list, legends, colors):
        ax.plot(path[:, 0], path[:, 1], label=legend, c=color, linewidth=3)

    plt.legend(loc="upper right")
    save_image(plt, parent_dir=parent_dir, save_name=save_name)


def get_time_uuid():
    return str(int(round(time.time() * 1000)))


def save_image(plt, parent_dir, save_name):
    plt.draw()

    if parent_dir is None:
        parent_dir = get_time_uuid()
    save_dir = os.path.join(get_project_path(), "visualize_utilities", parent_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if save_name is None:
        save_name = get_time_uuid()
    filename = os.path.join(save_dir, save_name + '.png')
    plt.savefig(filename)


def plot_collision_scene(results_dict, parent_dir=None, save_name=None):
    history_robot_positions = results_dict["history_robot_positions"]
    history_robot_yaws = results_dict["history_robot_yaws"]
    history_robot_velocities = results_dict["history_robot_velocities"]
    history_pedestrians_positions = results_dict["history_pedestrians_positions"]
    history_success = results_dict['history_success']
    history_occupancy_maps = results_dict["history_occupancy_maps"]
    grid_res = 0.1
    plt.figure(figsize=(5, 5))
    for success, robot_positions, pedestrians_positions, occupancy_map in zip(history_success, history_robot_positions,
                                                                              history_pedestrians_positions,
                                                                              history_occupancy_maps):
        if not success:
            ax = plt.gca()

            add_map_patch_to_ax(ax, occupancy_map)

            robot_positions = np.array(robot_positions[::4]) / grid_res
            pedestrians_positions = np.array(pedestrians_positions[::4]) / grid_res

            ax.plot(robot_positions[:, 0], robot_positions[:, 1], label="robot position", linewidth=3)
            ax.scatter(robot_positions[-1, 0], robot_positions[-1, 1], s=100, marker='x')

            for i in range(len(pedestrians_positions[0])):
                ax.plot(pedestrians_positions[:, i, 0], pedestrians_positions[:, i, 1],
                        label="pedestrian_{}".format(i), linewidth=3)
                ax.scatter(pedestrians_positions[-1, i, 0], pedestrians_positions[-1, i, 1], s=100, marker='x')

            plt.text(0, 0.02, "success" if success else "Fail", ha='left', va='bottom', fontsize=10)

            plt.axis('equal')
            plt.axis('off')
            plt.legend(loc="upper right", fontsize=16)

            plt.tight_layout()

            save_image(plt, parent_dir=parent_dir, save_name=save_name)
            plt.cla()
    plt.show()
