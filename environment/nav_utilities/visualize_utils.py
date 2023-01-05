#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : navigation_icra 
    @Author  : Xiangyu Zeng
    @Date    : 9/3/22 2:59 PM 
    @Description    :

===========================================
"""
import os
import time

import numpy as np
from matplotlib import pyplot as plt

from environment.nav_utilities.cast_ray_utils import compute_vectorized_cartesian_positions
from environment.nav_utilities.coordinates_converter import cvt_to_om

lw = 0.5
colors = ['b', 'g']


def visualize_cartesian(xs, ys, title="Plot lidar cartesian positions", c='r', visualize=False):
    if visualize:
        plt.scatter(xs, ys, lw=2, c=c)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        plt.clf()
    print()


def visualize_polar(thetas, dists, title="Plot lidar polar positions", c='r'):
    plt.polar(thetas, dists, 'ro', lw=2, c=c)
    plt.title(title)
    plt.tight_layout()
    plt.ylim(0, 1)
    plt.show()


def visualize_motion_center(xs_s_t_groups, ys_s_t_groups, masses_s_t_groups, centers, distance_thresh, visualize, save,
                            folder):
    if visualize or save:
        # draw circle
        num = 20
        thetas, distances = np.linspace(-np.pi, np.pi, num), np.ones((num,)) * distance_thresh

        labels = ["t-1", "t"]
        for t in range(len(ys_s_t_groups[0])):
            for s in range(len(xs_s_t_groups)):
                xs = xs_s_t_groups[s][t]
                ys = ys_s_t_groups[s][t]
                if len(xs) > 0 and len(ys) > 0:
                    if s == 0:
                        plt.scatter(xs, ys, lw=lw, marker='s', c=colors[t], label=labels[t])
                    else:
                        plt.scatter(xs, ys, lw=lw, marker='s', c=colors[t])

        for s in range(len(xs_s_t_groups)):
            mass_x = masses_s_t_groups[s][:, 0]
            mass_y = masses_s_t_groups[s][:, 1]
            if s == 0:
                plt.plot(mass_x, mass_y, c='gray', label="motion")
                plt.scatter(mass_x[0], mass_y[0], marker="*", label="descriptor t-1", color="r")
                plt.scatter(mass_x[1], mass_y[1], marker="o", label="descriptor t", color="r")
            else:
                plt.plot(mass_x, mass_y, c='gray')
                plt.scatter(mass_x[0], mass_y[0], marker="*", color="r")
                plt.scatter(mass_x[1], mass_y[1], marker="o", color="r")

        for s in range(len(xs_s_t_groups)):

            xs, ys = compute_vectorized_cartesian_positions(thetas, distances)
            xs += centers[0][s]
            ys += centers[1][s]
            if s == 0:
                plt.plot(xs, ys, c='k', label="group circle")
            else:
                plt.plot(xs, ys, c='k')

            plt.text(centers[0][s], centers[1][s], s)

        plt.ylim(-1, 1)
        plt.xlim(-1, 1)
        plt.legend()
        # plt.show()
        if save:
            if not os.path.exists(folder):
                os.makedirs(folder)
            plt.savefig(os.path.join(folder, "{}.png".format(time.time())))

        else:
            plt.show()
        plt.clf()


def visualize_mass_center(xs_s_t_groups, ys_s_t_groups, masses_s_t_groups, centers, distance_thresh, visualize, save,
                          folder):
    if visualize or save:
        # draw circle
        num = 20
        thetas, distances = np.linspace(-np.pi, np.pi, num), np.ones((num,)) * distance_thresh

        labels = ["t-1", "t"]
        for t in range(len(ys_s_t_groups[0])):
            for s in range(len(xs_s_t_groups)):
                xs = xs_s_t_groups[s][t]
                ys = ys_s_t_groups[s][t]
                if len(xs) > 0 and len(ys) > 0:
                    if s == 0:
                        plt.scatter(xs, ys, lw=lw, marker='s', c=colors[t], label=labels[t])
                    else:
                        plt.scatter(xs, ys, lw=lw, marker='s', c=colors[t])

        for s in range(len(xs_s_t_groups)):

            xs, ys = compute_vectorized_cartesian_positions(thetas, distances)
            xs += centers[0][s]
            ys += centers[1][s]
            if s == 0:
                plt.plot(xs, ys, c='k', label="group circle")
            else:
                plt.plot(xs, ys, c='k')

            plt.text(centers[0][s], centers[1][s], s)

        for s in range(len(xs_s_t_groups)):
            center_x0 = centers[0][s]
            center_y0 = centers[1][s]
            if s == 0:
                plt.scatter(center_x0, center_y0, c='r', label="group center")
            else:
                plt.scatter(center_x0, center_y0, c='r')
        plt.ylim(-1, 1)
        plt.xlim(-1, 1)
        plt.legend()
        # plt.show()
        if save:
            if not os.path.exists(folder):
                os.makedirs(folder)
            plt.savefig(os.path.join(folder, "{}.png".format(time.time())))
        elif visualize:
            plt.show()
        plt.clf()


def draw_lidar_coordinates(coordinates, c="r"):
    for i, end_occ_coordinate in enumerate(coordinates):
        plt.plot(np.array([0, end_occ_coordinate[0]]),
                 np.array([0, end_occ_coordinate[1]]), c=c, alpha=0.2)
    plt.show()


def visualize_cartesian_list(coordinates_flatten_list, title, labels, folder, visualize=False, save=False):
    if visualize or save:
        for i, coordinates_flatten in enumerate(coordinates_flatten_list):
            # color = colors[i % len(colors)]
            coordinates_flatten = np.array(coordinates_flatten)
            xs = coordinates_flatten.reshape((-1, 2))[:, 0]
            ys = coordinates_flatten.reshape((-1, 2))[:, 1]
            plt.scatter(xs, ys, lw=0.5, c=colors[i % len(colors)], label=labels[i])
            for x, y in zip(xs, ys):
                plt.plot(np.array([0, x]), np.array([0, y]), c='r', alpha=0.2)
        plt.legend()
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        if save:
            if not os.path.exists(folder):
                os.makedirs(folder)
            plt.savefig(os.path.join(folder, "{}.png".format(time.time())))
        elif visualize:
            plt.show()
        plt.clf()


def visualize_coordinates_list(ax, coordinates_list, labels):
    dark_colors = ["#6C7CD0", "#F3802E"]
    shadow_colors = ["#B5C1E9", "#FCD0B0"]
    for i in range(len(coordinates_list)):
        coordinates = coordinates_list[i]
        label = labels[i]

        xs = coordinates.reshape((-1, 2))[:, 0]
        ys = coordinates.reshape((-1, 2))[:, 1]
        ax.scatter(xs, ys, lw=0.5, c=dark_colors[i], label=label)

        # draw rays
        for x, y in zip(xs, ys):
            ax.plot([0, x], [0, y], c=shadow_colors[i], alpha=0.3)


def visualize_virtual_centroids(ax, coordinates_list, virtual_centroids, labels):
    visualize_coordinates_list(ax, coordinates_list, labels)
    dark_color = "#878787"

    # draw rays
    for x, y in virtual_centroids:
        ax.plot([0, x], [0, y], c=dark_color, alpha=0.3)
    # draw scatter
    ax.scatter(virtual_centroids[:, 0], virtual_centroids[:, 1], lw=0.5, c=dark_color, label=labels[2])


def visualize_virtual_centroids_circle(ax, coordinates_list, virtual_centroids, labels):
    visualize_coordinates_list(ax, coordinates_list, labels)
    dark_color = "#878787"
    # draw circle
    for i, virtual_centroid in enumerate(virtual_centroids):
        plot_circle(ax, virtual_centroid, radius=0.1, c=dark_color, alpha=1)
        ax.text(virtual_centroid[0], virtual_centroid[1], s=str(i))


def plot_circle(ax, position, radius=0.4, c='r', alpha=0.5):
    theta = np.linspace(0, 2 * np.pi, 150)
    a = radius * np.cos(theta) + position[0]
    b = radius * np.sin(theta) + position[1]
    ax.plot(a, b, '--', c=c, alpha=alpha)


def visualize_tag_descriptor(ax, coordinates_list, virtual_centroids, tag_descriptors, labels):
    # visualize_virtual_centroids(ax, coordinates_list, virtual_centroids, labels)
    blue = "#6C7CD0"
    orange = "#F3802E"
    tag_descriptor_t_1 = tag_descriptors[:, 0, :]
    tag_descriptor_t = tag_descriptors[:, 1, :]

    ax.scatter(tag_descriptor_t_1[:, 0], tag_descriptor_t_1[:, 1], c=blue)
    ax.scatter(tag_descriptor_t[:, 0], tag_descriptor_t[:, 1], c=orange)

    return


def visualize_one_tag_descriptor(ax, xs_s_t_groups, ys_s_t_groups, tag_descriptors,
                                 virtual_centroids, labels):
    index = 23
    icp_coordinate_t_1 = np.array([xs_s_t_groups[index][0], ys_s_t_groups[index][0]])
    icp_coordinate_t = np.array([xs_s_t_groups[index][1], ys_s_t_groups[index][1]])
    tag_descriptor = tag_descriptors[index]
    virtual_centroid = virtual_centroids[index]

    gray = "#878787"
    blue = "#6C7CD0"
    orange = "#F3802E"

    radius = 0.15
    # draw circle
    plot_circle(ax, virtual_centroid, radius=radius, c=gray, alpha=1)
    ax.scatter(icp_coordinate_t_1[0], icp_coordinate_t_1[1], c=blue)
    ax.scatter(icp_coordinate_t[0], icp_coordinate_t[1], c=orange)

    offset = 0.35
    plot_circle(ax, np.array([virtual_centroid[0] + offset, virtual_centroid[1]]),
                radius=radius, c=gray, alpha=1)
    ax.scatter([tag_descriptor[0][0] + offset], [tag_descriptor[0][1]], c=blue)
    ax.scatter([tag_descriptor[1][0] + offset], [tag_descriptor[1][1]], c=orange)
    ax.arrow(tag_descriptor[0][0] + 0.1, tag_descriptor[0][1], 0.1, 0, shape="full", width=0.01)
    ax.text(tag_descriptor[0][0] - 0.15, tag_descriptor[0][1] + 0.15, s="group " + str(index))
    return


def visualize_tag_descriptors(cartesian_positions, hit_coordinates_list, icp_coordinates_list, tag_descriptors,
                              xs_s_t_groups, ys_s_t_groups, virtual_centers, save_folder, save_path):
    """
    visualize temporal accumulated group descriptor (TAG descriptor)
    :return:
    """
    # fig = plt.figure(figsize=(8, 12))
    # Create four polar axes and access them through the returned array
    fig, axs = plt.subplots(3, 2, figsize=(8, 8))
    labels = ["scanning at time step t-1", "scanning at time step t"]
    visualize_coordinates_list(ax=axs[0, 0], coordinates_list=cartesian_positions,
                               labels=labels)

    visualize_coordinates_list(ax=axs[0, 1], coordinates_list=icp_coordinates_list,
                               labels=labels)
    labels += ["virtual centroids"]
    visualize_virtual_centroids(ax=axs[1, 0], coordinates_list=icp_coordinates_list,
                                virtual_centroids=virtual_centers, labels=labels)

    visualize_virtual_centroids_circle(ax=axs[1, 1], coordinates_list=icp_coordinates_list,
                                       virtual_centroids=virtual_centers, labels=labels)

    visualize_one_tag_descriptor(ax=axs[2, 0],
                                 xs_s_t_groups=xs_s_t_groups, ys_s_t_groups=ys_s_t_groups,
                                 tag_descriptors=tag_descriptors, virtual_centroids=virtual_centers,
                                 labels=labels)

    visualize_tag_descriptor(ax=axs[2, 1], coordinates_list=icp_coordinates_list, virtual_centroids=virtual_centers,
                             tag_descriptors=tag_descriptors, labels=labels)
    plt.legend()
    # plt.title(title)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    # axs[0, 0].text(-0.05, -0.05, "Raw scanning")
    axs[0, 0].title.set_text('(a). Raw scanning')
    axs[0, 1].title.set_text('(b). Icp alignment')
    axs[1, 0].title.set_text('(c). Virtual centroids')
    axs[1, 1].title.set_text('(d). Assign to groups')
    axs[2, 0].title.set_text('(e). How to compute group centroids')
    axs[2, 1].title.set_text('(f). Group centroids at time step t-1 and t')

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, save_path))
    # plt.show()
