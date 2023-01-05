#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 7/5/22 9:39 AM 
    @Description    :
        
===========================================
"""

import os.path
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np

from utils.fo_utility import get_project_path
from visualize_utilities.bar_grah import bar_graph, bar_graph_with_error
from visualize_utilities.metric_extractor import read_json, extract_metric
import seaborn as sns

COLORS = sns.color_palette("Paired")


def test_compare_success_rate_of_different_algorithms_in_different_envs():
    # 比较同一个方法在不同的环境配置下，如不同的人数，不同的行人速度下，度量的变化
    project_path = get_project_path()
    plot_folder = "plot"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    metrics_folder = os.path.join(project_path, "output", "evaluation")
    metric_jsons = read_json(metrics_folder)
    conditions = {
        "env_types": ["hybrid", "corridor", "cross", "office"],

        "folder_name": ["CATAttentionNetTemporalSpacialLSTM4_uncomfortable_distance",
                        "CATAttentionNetSpacial_no_large_office",
                        "traditional_elastic_band",
                        "traditional_a_star",
                        ],

        "pedestrian_dynamic_num": [2],
        "pedestrian_max_speed_range": [0.25]
    }

    metrics = "success_rate"

    values = extract_metric(metric_jsons, conditions, metrics)

    x_names = conditions["env_types"]
    label_names = ["temporal and interaction module", "iteraction module",
                   "traditional_elastic_band", "traditional_a_star"]
    values = np.reshape(values, (len(x_names), len(label_names)))
    color_index = [1, 0, 9, 8]
    colors = [COLORS[i] for i in color_index]
    assert len(colors) == len(label_names)
    y_label = "Success Rate"

    save_path = os.path.join(plot_folder, "success_rate_of_different_algorithms.png")

    title = "Compare different algorithm in different environments"
    bar_graph(x_names, label_names, values, colors, title, y_label, save_path)


def test_compare_path_length_of_different_algorithms_in_different_envs():
    # 比较同一个方法在不同的环境配置下，如不同的人数，不同的行人速度下，度量的变化
    project_path = get_project_path()
    plot_folder = "plot"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    metrics_folder = os.path.join(project_path, "output", "evaluation")
    metric_jsons = read_json(metrics_folder)
    conditions = {
        "env_types": ["hybrid", "corridor", "cross", "office"],

        "folder_name": ["CATAttentionNetTemporalSpacialLSTM4_uncomfortable_distance",
                        "CATAttentionNetSpacial_no_large_office",
                        "traditional_elastic_band",
                        "traditional_a_star",
                        ],

        "pedestrian_dynamic_num": [2],
        "pedestrian_max_speed_range": [0.25]
    }

    metrics = "path_length_mean"

    values = extract_metric(metric_jsons, conditions, metrics)

    x_names = conditions["env_types"]
    label_names = ["temporal and interaction module", "iteraction module",
                   "traditional_elastic_band", "traditional_a_star"]
    values = np.reshape(values, (len(x_names), len(label_names)))
    color_index = [5, 4, 1, 0]
    colors = [COLORS[i] for i in color_index]
    assert len(colors) == len(label_names)
    y_label = "Path length (m/s)"

    save_path = os.path.join(plot_folder, "path_length_mean_of_different_algorithms.png")

    title = "Compare different algorithm in different environments"
    bar_graph(x_names, label_names, values, colors, title, y_label, save_path)


def test_compare_travel_time_of_different_algorithms_in_different_envs():
    # 比较同一个方法在不同的环境配置下，如不同的人数，不同的行人速度下，度量的变化
    project_path = get_project_path()
    plot_folder = "plot"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    metrics_folder = os.path.join(project_path, "output", "evaluation")
    metric_jsons = read_json(metrics_folder)
    conditions = {
        "env_types": ["hybrid", "corridor", "cross", "office"],

        "folder_name": ["CATAttentionNetTemporalSpacialLSTM4_uncomfortable_distance",
                        "CATAttentionNetSpacial_no_large_office",
                        "traditional_elastic_band",
                        "traditional_a_star",
                        ],

        "pedestrian_dynamic_num": [2],
        "pedestrian_max_speed_range": [0.25]
    }

    metrics = "travels_time_mean"

    values = extract_metric(metric_jsons, conditions, metrics)

    x_names = conditions["env_types"]
    label_names = ["temporal and interaction module", "iteraction module",
                   "traditional_elastic_band", "traditional_a_star"]
    values = np.reshape(values, (len(x_names), len(label_names)))
    color_index = [6, 7, 0, 1]
    colors = [COLORS[i] for i in color_index]
    assert len(colors) == len(label_names)
    y_label = "Travel time step"

    save_path = os.path.join(plot_folder, "travel_mean_of_different_algorithms.png")

    title = "Compare different algorithm in different environments"
    bar_graph(x_names, label_names, values, colors, title, y_label, save_path)


def test_compare_closest_distance_of_different_algorithms_in_different_envs():
    # 比较同一个方法在不同的环境配置下，如不同的人数，不同的行人速度下，度量的变化
    project_path = get_project_path()
    plot_folder = "plot"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    metrics_folder = os.path.join(project_path, "output", "evaluation")
    metric_jsons = read_json(metrics_folder)
    conditions = {
        "env_types": ["hybrid", "corridor", "cross", "office"],

        "folder_name": ["CATAttentionNetTemporalSpacialLSTM4_uncomfortable_distance",
                        "CATAttentionNetSpacial_no_large_office",
                        "traditional_elastic_band",
                        "traditional_a_star",
                        ],

        "pedestrian_dynamic_num": [2],
        "pedestrian_max_speed_range": [0.25]
    }

    metrics = "min_distance_mean"

    values = extract_metric(metric_jsons, conditions, metrics)
    errors = extract_metric(metric_jsons, conditions, "min_distance_std")
    x_names = conditions["env_types"]
    label_names = ["temporal and interaction module", "iteraction module",
                   "traditional_elastic_band", "traditional_a_star"]
    values = np.reshape(values, (len(x_names), len(label_names)))
    errors = np.reshape(errors, (len(x_names), len(label_names)))

    color_index = [8, 9, 2, 3]
    colors = [COLORS[i] for i in color_index]
    assert len(colors) == len(label_names)
    y_label = "Closest distance (m)"

    save_path = os.path.join(plot_folder, "min_distance_mean_of_different_algorithms.png")

    title = "Compare different algorithm in different environments"
    bar_graph_with_error(x_names, label_names, values, errors, colors, title, y_label, save_path)
