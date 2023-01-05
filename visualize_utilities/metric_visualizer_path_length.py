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

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np

from utils.fo_utility import get_project_path
from visualize_utilities.bar_grah import bar_graph
from visualize_utilities.metric_extractor import read_json, extract_metric


def compare_path_length_of_deformation_methods_in_corridor():
    # 比较同一个方法在不同的环境配置下，如不同的人数，不同的行人速度下，度量的变化
    project_path = get_project_path()
    plot_folder = "plot"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    metrics_folder = os.path.join(project_path, "output", "evaluation")
    metric_jsons = read_json(metrics_folder)

    conditions = {
        "pedestrian_dynamic_num": [2, 3, 4],

        "folder_name": ["reduce_to_original_and_ideal_target_reward",
                        "gaussian_single_add_stop_reach_target_2",
                        "bezier",
                        ],
        "env_types": ["corridor"],
        "pedestrian_max_speed_range": [0.35],
        "algorithm": ["hierarchical_rl"]
    }
    metrics = "path_length_mean"

    values = extract_metric(metric_jsons, conditions, metrics)

    x_names = conditions["pedestrian_dynamic_num"]
    label_names = ["linear", "gaussian", "bezier"]

    values = np.reshape(values, (len(x_names), len(label_names)))
    colors = [Color.Red, Color.PinkGray, Color.LIGHTSALMON]
    assert len(colors) == len(label_names)
    y_label = "Path Length"

    save_path = os.path.join(plot_folder, "path_length_of_different_deformation_methods_in_corridor.png")

    title = "Path length of different deformation methods in corridor"
    bar_graph(x_names, label_names, values, colors, title, y_label, save_path)


def compare_path_length_of_different_algorithm_in_office():
    # 比较同一个方法在不同的环境配置下，如不同的人数，不同的行人速度下，度量的变化
    project_path = get_project_path()
    plot_folder = "plot"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    metrics_folder = os.path.join(project_path, "output", "evaluation")
    metric_jsons = read_json(metrics_folder)

    conditions = {
        "pedestrian_dynamic_num": [2, 3, 4],

        "folder_name": ["reduce_to_original_and_ideal_target_reward",
                        "non_hierarchical_change_update_deal_target_index",
                        "traditional_a_star",
                        "traditional_elastic_band"
                        ],
        "env_types": ["office"],
        "pedestrian_max_speed_range": [0.35],
    }
    metrics = "path_length_mean"

    values = extract_metric(metric_jsons, conditions, metrics)

    x_names = conditions["pedestrian_dynamic_num"]
    label_names = ["linear", "non_hierarchical", "traditional_a_star", "traditional_elastic_band"]
    values = np.reshape(values, (len(x_names), len(label_names)))
    colors = [Color.Red, Color.PinkGray, Color.LIGHTSALMON, Color.PurpleGray]
    assert len(colors) == len(label_names)
    y_label = "Path Length"

    save_path = os.path.join(plot_folder, "path_length_of_different_algorithm_in_office.png")

    title = "Path length of different algorithm in office"
    bar_graph(x_names, label_names, values, colors, title, y_label, save_path)


def compare_path_length_of_different_algorithm_in_corridor():
    # 比较同一个方法在不同的环境配置下，如不同的人数，不同的行人速度下，度量的变化
    project_path = get_project_path()
    plot_folder = "plot"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    metrics_folder = os.path.join(project_path, "output", "evaluation")
    metric_jsons = read_json(metrics_folder)

    conditions = {
        "pedestrian_dynamic_num": [2, 3, 4],

        "folder_name": ["reduce_to_original_and_ideal_target_reward",
                        "gaussian_single_add_stop_reach_target_2",
                        "bezier",
                        ],
        "env_types": ["corridor"],
        "pedestrian_max_speed_range": [0.35],
        "algorithm": ["hierarchical_rl"]

    }
    metrics = "path_length_mean"

    values = extract_metric(metric_jsons, conditions, metrics)

    x_names = conditions["pedestrian_dynamic_num"]
    label_names = ["linear", "gaussian", "bezier"]

    # label_names = ["linear", "non_hierarchical", "traditional_a_star", "traditional_elastic_band"]
    values = np.reshape(values, (len(x_names), len(label_names)))
    colors = [Color.Red, Color.PinkGray, Color.LIGHTSALMON]
    assert len(colors) == len(label_names)
    y_label = "Path Length"

    save_path = os.path.join(plot_folder, "path_length_of_different_deformation_methods_in_corridor.png")

    title = "Path length of different deformation methods in corridor"
    bar_graph(x_names, label_names, values, colors, title, y_label, save_path)


if __name__ == '__main__':
    # compare_path_length_of_different_algorithm_in_office()
    # compare_path_length_of_different_algorithm_in_corridor()
    # compare_path_length_of_deformation_methods_in_corridor()
    # compare_feel_uncomfortable_ratio_of_different_algorithm_in_different_environment()
    # compare_different_deformation_in_different_envs()
    compare_path_length_of_different_algorithm_in_corridor()
