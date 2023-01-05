#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 7/3/22 11:00 PM 
    @Description    :
        
===========================================
"""
from matplotlib import pyplot as plt

from visualize_utilities.plot_env import save_image

FONT_SIZE = 16


def plot_heat_map(xx, yy, zz, original_band, initial_band, sampled_original_band, optimized_band,
                  sampled_optimized_band, parent_dir, save_name):
    # Prepare the plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('x', fontsize=FONT_SIZE)
    ax.set_ylabel('y', fontsize=FONT_SIZE)

    # Plot the potential map
    cont = ax.contourf(xx, yy, zz, levels=25, cmap='gist_heat')
    fig.colorbar(cont)

    # Plot the initial band
    # if original_band:
    if original_band is not None:
        ax.scatter(original_band[:, 0], original_band[:, 1], c='green', s=10, label='original band')

    if initial_band is not None:
        ax.scatter(initial_band[:, 0], initial_band[:, 1], c='red', marker='o', s=20, label='initial band')

    if sampled_original_band is not None:
        ax.scatter(sampled_original_band[:, 0], sampled_original_band[:, 1], c='purple', marker='x', s=10,
                   label='sampled initial band')

    if optimized_band is not None:
        # Plot the minimized band
        ax.scatter(optimized_band[:, 0], optimized_band[:, 1], c='blue', marker='o', s=10, label='optimized band')

    if sampled_optimized_band is not None:
        ax.scatter(sampled_optimized_band[:, 0], sampled_optimized_band[:, 1], c='yellow', marker='x', s=10,
                   label='sampled optimized band')
    plt.tight_layout()
    # plt.axis('off')
    # Save the plot
    # ax.legend()
    save_image(plt, parent_dir, save_name)
    fig.show()
