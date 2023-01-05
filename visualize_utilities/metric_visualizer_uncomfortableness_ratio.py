#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 6/15/22 1:13 PM 
    @Description    :
        
===========================================
"""
import os.path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np

from utils.fo_utility import get_project_path
from visualize_utilities.bar_grah import Color, bar_graph
from visualize_utilities.metric_extractor import read_json, extract_metric


def compare_feel_uncomfortable_ratio_of_different_algorithm_in_different_environment():
    project_path = get_project_path()
    plot_folder = "plot"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    metrics_folder = os.path.join(project_path, "output", "evaluation")
    metric_jsons = read_json(metrics_folder)

    conditions = {"env_types": ["hybrid", "corridor", "cross", "office"],
                  "folder_name": ["reduce_to_original_and_ideal_target_reward",
                                  "non_hierarchical_change_update_deal_target_index",
                                  "traditional_a_star",
                                  "traditional_elastic_band"
                                  ],
                  "pedestrian_dynamic_num": [2],
                  "pedestrian_max_speed_range": [0.35]
                  }
    metrics = "feel_uncomfortable_ratio"

    values = extract_metric(metric_jsons, conditions, metrics)

    x_names = conditions["env_types"]
    label_names = ["linear", "non_hierarchical", "traditional_a_star", "traditional_elastic_band"]
    values = np.reshape(values, (len(x_names), len(label_names)))
    colors = [Color.Red, Color.PinkGray, Color.PurpleGray, Color.LIGHTSALMON]
    assert len(colors) == len(label_names)
    y_label = "Uncomfortableness Ratios"

    save_path = os.path.join(plot_folder, "uncomfortableness_ratios.png")

    title = "Compare Uncomfortableness Ratios"
    bar_graph(x_names, label_names, values, colors, title, y_label, save_path)


def compare_feel_uncomfortable_ratio_of_different_deformation_methods_in_different_environment():
    project_path = get_project_path()
    plot_folder = "plot"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    metrics_folder = os.path.join(project_path, "output", "evaluation")
    metric_jsons = read_json(metrics_folder)

    conditions = {"env_types": ["hybrid", "corridor", "cross", "office"],
                  "folder_name": ["reduce_to_original_and_ideal_target_reward",
                                  "non_hierarchical_change_update_deal_target_index",
                                  "traditional_a_star",
                                  "traditional_elastic_band"
                                  ],
                  "pedestrian_dynamic_num": [2],
                  "pedestrian_max_speed_range": [0.35]
                  }
    metrics = "feel_uncomfortable_ratio"

    values = extract_metric(metric_jsons, conditions, metrics)

    x_names = conditions["env_types"]
    label_names = ["linear", "non_hierarchical", "traditional_a_star", "traditional_elastic_band"]
    values = np.reshape(values, (len(x_names), len(label_names)))
    colors = [Color.Red, Color.PinkGray, Color.PurpleGray, Color.LIGHTSALMON]
    assert len(colors) == len(label_names)
    y_label = "Uncomfortableness Ratios"

    save_path = os.path.join(plot_folder, "uncomfortableness_ratios.png")

    title = "Compare Uncomfortableness Ratios"
    bar_graph(x_names, label_names, values, colors, title, y_label, save_path)


if __name__ == '__main__':
    # compare_feel_uncomfortable_ratio_of_different_algorithm_in_different_environment()
    compare_feel_uncomfortable_ratio_of_different_deformation_methods_in_different_environment()
