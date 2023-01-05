#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 7/5/22 12:16 AM 
    @Description    :
        
===========================================
"""
import numpy as np
from matplotlib import pyplot as plt

from visualize_utilities.plot_env import save_image


def plot_action_distribution(actions, parent_dir=None, save_name=None):
    actions = np.concatenate(actions, axis=0)[:, 1:]
    counts = np.zeros((6, 20))
    forward_distances = np.arange(0, 0.6 + 0.1, 0.1)
    forward_starts = forward_distances[: -1]
    forward_ends = forward_distances[1:]

    amplitude_distances = np.arange(-1, 1 + 0.1, 0.1)
    amplitude_starts = amplitude_distances[: -1]
    amplitude_ends = amplitude_distances[1:]

    # 统计每个格子中, action落在此格子的数量
    for action in actions:
        forward, amplitude = action[0], action[1]
        for i in range(len(counts)):
            for j in range(len(counts[0])):
                forward_start = forward_starts[i]
                forward_end = forward_ends[i]
                amplitude_start = amplitude_starts[j]
                amplitude_end = amplitude_ends[j]
                if forward_start < forward <= forward_end and amplitude_start < amplitude <= amplitude_end:
                    counts[i, j] += 1
    counts = counts / np.sum(counts)
    fig, ax0 = plt.subplots(1, 1)

    # set color bar
    c = ax0.pcolor(counts)
    plt.colorbar(c)

    plt.ylabel("Forward distance (m)", fontsize=16)
    plt.xlabel("Amplitude distance (m)", fontsize=16)
    plt.ylim(0, 6)
    plt.xlim(0, 20)
    yticks = [i for i in range(7)]
    xticks = [i for i in range(22) if i % 2 == 0]
    plt.xticks(xticks)
    plt.yticks(yticks)
    yticks_label = np.around(np.array(yticks) * 0.1, 1)
    ax0.set_yticklabels(yticks_label, fontsize=12)
    xticks_label = np.around(np.array(xticks) * 0.1, 1)
    ax0.set_xticklabels(xticks_label, fontsize=12)

    fig.tight_layout()
    save_image(plt, parent_dir=parent_dir, save_name=save_name)
    plt.show()


if __name__ == '__main__':
    plot_action_distribution(Z=np.random.rand(6, 10))
