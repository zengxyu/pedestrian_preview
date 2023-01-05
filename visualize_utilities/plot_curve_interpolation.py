#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 6/15/22 10:31 PM 
    @Description    :
        
===========================================
"""
import numpy as np
import matplotlib.pyplot as plt

FONT_SIZE = 16


def add_plt():
    # Create a figure of size 8x6 inches, 80 dots per inch
    plt.figure(figsize=(8, 6), dpi=80)
    plt.rc('font', size=FONT_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=FONT_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=FONT_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONT_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=FONT_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=FONT_SIZE)  # fontsize of the figure title
    return plt


def double_bezier(p1, p2, p3, t):
    parm_1 = (1 - t) ** 2
    parm_2 = 2 * t * (1 - t)
    parm_3 = t ** 2
    px = p1[0] * parm_1 + p2[0] * parm_2 + p3[0] * parm_3
    py = p1[1] * parm_1 + p2[1] * parm_2 + p3[1] * parm_3

    return np.array([px, py])


def linear_interpolation(p1, p2, t):
    parm_1 = (1 - t)
    parm_2 = t
    px = p1[0] * parm_1 + p2[0] * parm_2
    py = p1[1] * parm_1 + p2[1] * parm_2

    return np.array([px, py])


def gaussian(x, mean=0, sigma=5, a=1):
    y = a * np.e ** (-(x - mean) ** 2 / (2 * sigma ** 2))
    return np.array([x, y])


def bezier_points():
    p1 = [0, 0]
    p2 = [1.5, 2]
    p3 = [6, 0]
    t = np.linspace(0, 1, 100)
    points = double_bezier(p1, p2, p3, t)
    control_points = [p1, p2, p3]
    return points, control_points


def linear_points():
    p1 = [0, 0]
    p2 = [3, 2]
    p3 = [6, 0]
    t = np.linspace(0, 1, 100)
    points1 = linear_interpolation(p1, p2, t)
    points2 = linear_interpolation(p2, p3, t)
    points = np.concatenate([points1[0], points2[0]]), np.concatenate([points1[1], points2[1]])
    control_points = [p1, p2, p3]
    return points, control_points


def gaussian_points():
    x1 = 0
    x2 = 3
    x3 = 6
    t = np.linspace(0, 1, 100) * (x3 - x1)
    points = gaussian(t, x2, x2 / 3, a=1)
    control_points = [[x1, 0], [x3, 0]]
    return points, control_points


def plot_plot(points, control_points, title, xticks, yticks, save_path):
    plt = add_plt()
    plt.grid()
    #
    for i, control_point in enumerate(control_points):
        plt.plot(control_point[0], control_point[1], 'x', color='r', markersize=8)
        plt.text(control_point[0] + 0.2, control_point[1], "P" + str(i))

    x, y = points[0], points[1]
    # Plot cosine with a blue continuous line of width 1 (pixels)
    plt.plot(x, y, color="g", linewidth=2.0, linestyle="-")
    plt.ylabel(yticks)
    plt.xlabel(xticks)
    plt.title(title)

    plt.savefig(save_path)

    plt.show()
    plt.clf()
    plt.close()


def plot_gaussian(points, control_points, title, xticks, yticks, save_path):
    plt = add_plt()
    plt.grid()
    #
    for i, control_point in enumerate(control_points):
        plt.plot(control_point[0], control_point[1], 'x', color='r', markersize=8)
        plt.text(control_point[0] + 0.2, control_point[1], "P" + str(i))

    x, y = points[0], points[1]
    # Plot cosine with a blue continuous line of width 1 (pixels)
    plt.plot(x, y, color="g", linewidth=2.0, linestyle="-", label="a = 1, b = 3, c = b/3")
    plt.ylabel(yticks)
    plt.xlabel(xticks)
    plt.title(title)
    # plt.legend(loc="upper right")
    plt.legend()
    plt.savefig(save_path)

    plt.show()
    plt.clf()
    plt.close()


if __name__ == '__main__':
    points, control_points = bezier_points()
    plot_plot(points, control_points, "Bezier curve", 'x', "y", "plot/bezier2.png")
    #
    # points, control_points = linear_points()
    # plot_plot(points, control_points, "Linear interpolation", 'x', "y", "plot/linear.png")

    # points, control_points = gaussian_points()
    # plot_gaussian(points, control_points, "Gaussian function", 'x', "y", "plot/gaussian2.png")
