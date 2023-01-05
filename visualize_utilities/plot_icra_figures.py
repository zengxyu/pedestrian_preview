# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : navigation_icra 
    @Author  : Xiangyu Zeng
    @Date    : 9/12/22 7:45 PM 
    @Description    :
        
===========================================
"""
import json
import os

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from utils.fo_utility import get_project_path
from visualize_utilities.icra_figure_util import plot_two_stack_bar_chart

json_filename_template = "dynamic_{}+static_{}_speed_{}.json"
colors = [
    "#ffccff",
    "#FFB6C1",
    "#FFBBFF",
    "#00cc99",
    "#33cc00",
    "#0066ff",
    "#eab676",
    "#66cc99",
    "#66ccff",
    "#9999ff",
    "#ffcc99",
    "#ffcccc",
]


def load_rates(filenames):
    keys = ["success_rate", "collision_rate", "timeout_rate"]
    rates = []
    for filename in filenames:
        with open(filename) as f:
            data = json.load(f)
            rates.append([data[k] * 100 for k in keys])
    rates = np.array(rates)
    return rates


def load_navigation_time(filenames):
    data_list = []
    for i, name in enumerate(filenames):
        with open(name) as f:
            data = json.load(f)["navigation_time"]
            data_list.append(data)
            # get common keys
            if i == 0:
                keys = set(data.keys())
            else:
                keys = keys.intersection(set(data.keys()))
    # calculate average time
    value_list = []
    for i, data in enumerate(data_list):
        total_time = 0
        for k in keys:
            total_time += data_list[i][k]
        # 转换成秒的比例
        second_ratio = 0.4
        value_list.append(round(total_time * second_ratio / len(keys), 1))
    return value_list


def line_graph(x_names, label_names, values, colors, markers, x_label, y_label, save_path, fontsize=13):
    values = np.around(values, 2)
    for i, label_name in enumerate(label_names):
        plt.plot(np.arange(0, len(values[i, :])), values[i, :], label=label_name, color=colors[i], alpha=0.75,
                 marker=markers[i],
                 ms=7)
        j = 0
        for x, y in zip(np.arange(0, len(values[i, :])), values[i, :]):
            """求在这一组数据中值大小拍第几"""
            a = abs(np.sort(values[:, j]) - y) <= 0.001
            y_float = np.argwhere(a)[0][0] - 1 - len(values[:, j]) / 2
            plt.text(x, y + 0.02 * y_float, y, ha='left', va='bottom', fontsize=10)
            j += 1

    plt.xticks(np.arange(0, len(x_names)), x_names, fontsize=fontsize)
    plt.ylim(0, np.max(values) + 0.05 * np.max(values))
    plt.yticks(fontsize=fontsize - 2)
    plt.legend(loc='lower right')
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xlabel(x_label, fontsize=fontsize)

    plt.tight_layout()
    plt.grid()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=500)
    plt.show()


def plot_dynamic_num():
    env = "hybrid"
    dynamic_nums = [1, 2, 3, 4]
    filenames_temporal_mass = [
        os.path.join(in_folder, "temporal_mass", env, json_filename_template.format(d_num, 1, 0.3))
        for d_num in dynamic_nums]
    filenames_cnn = [os.path.join(in_folder, "cnn", env, json_filename_template.format(d_num, 1, 0.3))
                     for d_num in dynamic_nums]
    rates_temporal_mass = load_rates(filenames_temporal_mass)
    rates_cnn = load_rates(filenames_cnn)

    x_names = ["{}".format(c) for c in dynamic_nums]
    legend_labels = ["success", "collision", "timeout"]

    plt.figure(figsize=(10, 6))
    ax = plt.subplot(1, 1, 1)
    # ax.plot(x_names, rates_temporal_mass[:, 0], color=colors[-1], linewidth=3)
    offset = 0.3 * 1.1
    r2 = [x + offset for x in np.arange(len(x_names))]
    # ax.plot(r2, rates_cnn[:, 0], color=colors[-2], linewidth=3)
    save_path = os.path.join(out_folder, "dynamic_num_compare.png")
    colors = seaborn.color_palette("Paired")
    plot_two_stack_bar_chart(
        [colors[3], colors[2], colors[4]],
        x_names,
        rates_temporal_mass,
        rates_cnn,
        legend_labels,
        ax,
        None,
        True,
        "lower left",
        save_path,
        xlabel="Dynamic obstacle number",
        ylabel="Performance (%)",
        width=0.3,
        top_names=['Our', '[17]'],
    )
    plt.clf()


def compare_velocity():
    plt.figure(figsize=(10, 6))
    env = "hybrid"
    velocities = [round(0.1 * i, 1) for i in range(2, 6)]
    x_names = ["{} m/s".format(v) for v in velocities]
    legend_labels = ["success", "collision", "timeout"]

    filenames_temporal_mass = [
        os.path.join(in_folder, "temporal_mass", env, json_filename_template.format(2, 1, v))
        for v in velocities]
    filenames_cnn = [os.path.join(in_folder, "cnn", env, json_filename_template.format(2, 1, v))
                     for v in velocities]
    rates_temporal_mass = load_rates(filenames_temporal_mass)
    rates_cnn = load_rates(filenames_cnn)

    ax = plt.subplot(1, 1, 1)
    # ax.plot(x_names, rates_temporal_mass[:, 0], color=colors[-1], linewidth=3)
    offset = 0.3 * 1.1
    r2 = [x + offset for x in np.arange(len(x_names))]
    # ax.plot(r2, rates_cnn[:, 0], color=colors[-2], linewidth=3)

    save_path = os.path.join(out_folder, "velocity_compare.png")
    colors = seaborn.color_palette("Paired")
    plot_two_stack_bar_chart(
        [colors[3], colors[2], colors[4]],
        x_names,
        rates_temporal_mass,
        rates_cnn,
        legend_labels,
        ax,
        None,
        True,
        "lower left",
        save_path,
        xlabel="Dynamic obstacle velocity",
        ylabel="Performance (%)",
        width=0.3,
        top_names=['Our', '[17]'],
    )
    plt.clf()


if __name__ == '__main__':
    in_folder = os.path.join(get_project_path(), "output/test_result/")
    out_folder = os.path.join(get_project_path(), "output/test_result/")

    plot_dynamic_num()
    compare_velocity()

    v = 0.3
    # 导航时间
    filenames = [
        os.path.join(in_folder, "temporal_mass", "hybrid", json_filename_template.format(2, 1, v)),
        os.path.join(in_folder, "spacial", "hybrid", json_filename_template.format(2, 1, v)),
        os.path.join(in_folder, "temporal", "hybrid", json_filename_template.format(2, 1, v)),
        os.path.join(in_folder, "cnn", "hybrid", json_filename_template.format(2, 1, v)),
    ]
    rates = load_rates(filenames)
    navigation_times = load_navigation_time(filenames)
    print("正确率 --\n {}".format(rates))
    print("导航时间 -- {}".format(navigation_times))
