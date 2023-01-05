#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 6/9/22 6:47 PM 
    @Description    :
        
===========================================
"""
import os.path
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np

from utils.fo_utility import get_project_path
from visualize_utilities.bar_grah import bar_graph, line_graph
from visualize_utilities.metric_extractor import read_json, extract_metric
import seaborn as sns

COLORS = sns.color_palette("Paired")


def test_compare_success_rate_of_pedestrian_speeds():
    # 比较同一个方法在不同的环境配置下，如不同的人数，不同的行人速度下，度量的变化
    project_path = get_project_path()
    plot_folder = "plot"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    metrics_folder = os.path.join(project_path, "output", "evaluation")
    metric_jsons = read_json(metrics_folder)

    conditions = {"env_types": ["hybrid", "corridor", "cross", "office"],

                  "pedestrian_max_speed_range": [0.15, 0.25, 0.35, 0.45],

                  "folder_name": ["CATAttentionNetSpacial_no_large_office"],
                  "pedestrian_dynamic_num": [2],
                  }
    metrics = "success_rate"

    values = extract_metric(metric_jsons, conditions, metrics)

    label_names = conditions["env_types"]
    x_names = conditions["pedestrian_max_speed_range"]
    x_label = "Pedestrian speed (m/s)"
    values = np.reshape(values, (len(label_names), len(x_names)))

    colors = [COLORS[i] for i in [5, 7, 9, 3]]
    markers = ['s', '*', 'h', 'o']
    assert len(colors) == len(label_names)
    y_label = "Success Rate (%)"

    save_path = os.path.join(plot_folder, "success_rate_of_different_pedestrian_speeds.png")

    title = "Compare success rate of different pedestrian speeds"
    line_graph(x_names, label_names, values, colors, markers, title, x_label, y_label, save_path)


def test_compare_success_rate_of_pedestrian_numbers():
    # 比较同一个方法在不同的环境配置下，如不同的人数，不同的行人速度下，度量的变化
    project_path = get_project_path()
    plot_folder = "plot"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    metrics_folder = os.path.join(project_path, "output", "evaluation")
    metric_jsons = read_json(metrics_folder)
    conditions = {
        "env_types": ["hybrid", "corridor", "cross", "office"],

        "pedestrian_dynamic_num": [1, 2, 3, 4],

        "folder_name": ["CATAttentionNetSpacial_no_large_office"],
        "pedestrian_max_speed_range": [0.25],

    }

    metrics = "success_rate"

    values = extract_metric(metric_jsons, conditions, metrics)

    label_names = conditions["env_types"]
    x_names = ["1", "2", "3", "4"]
    x_label = "Pedestrian numbers"
    values = np.reshape(values, (len(label_names), len(x_names)))

    colors = [COLORS[i] for i in [5, 7, 9, 3]]
    markers = ['s', '*', 'h', 'o']
    assert len(colors) == len(label_names)
    y_label = "Success Rate (%)"

    save_path = os.path.join(plot_folder, "success_rate_of_different_pedestrian_numbers.png")

    title = "Compare success rate of different pedestrian numbers"
    line_graph(x_names, label_names, values, colors, markers, title, x_label, y_label, save_path)


def compare_success_rate_different_deformation_in_different_envs():
    # 比较同一个方法在不同的环境配置下，如不同的人数，不同的行人速度下，度量的变化
    project_path = get_project_path()
    plot_folder = "plot"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    metrics_folder = os.path.join(project_path, "output", "evaluation")
    metric_jsons = read_json(metrics_folder)

    conditions = {"env_types": ["hybrid", "corridor", "cross", "office"],
                  "folder_name": ["reduce_to_original_and_ideal_target_reward",
                                  "gaussian_single_add_stop_reach_target_2",
                                  "bezier",
                                  ],
                  "pedestrian_dynamic_num": [2],
                  "pedestrian_max_speed_range": [0.35],
                  "algorithm": ["hierarchical_rl"]
                  }
    metrics = "success_rate"

    values = extract_metric(metric_jsons, conditions, metrics)

    x_names = conditions["env_types"]
    label_names = ["linear", "gaussian", "bezier"]
    values = np.reshape(values, (len(x_names), len(label_names)))
    colors = [Color.Red, Color.PinkGray, Color.LIGHTSALMON]
    assert len(colors) == len(label_names)
    y_label = "Success Rate"

    save_path = os.path.join(plot_folder, "success_rate_of_different_deformation_methods.png")

    title = "Compare deformation methods in different environments"
    bar_graph(x_names, label_names, values, colors, title, y_label, save_path)


if __name__ == '__main__':
    # compare_success_rate_of_different_algorithms_in_different_envs()
    # compare_success_rate_different_deformation_in_different_envs()
    # compare_success_rate_of_pedestrian_speeds()
    compare_success_rate_of_pedestrian_numbers()
