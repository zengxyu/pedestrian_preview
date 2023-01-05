#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 7/5/22 7:48 AM 
    @Description    :
        
===========================================
"""
import os.path
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np

from utils.fo_utility import get_project_path
from visualize_utilities.bar_grah import bar_graph, bar_line_graph
from visualize_utilities.metric_extractor import read_json, extract_metric
import seaborn as sns

COLORS = sns.color_palette("Paired")


def test_compare_success_rate_of_pedestrian_number():
    # 比较同一个方法在不同的环境配置下，如不同的人数，不同的行人速度下，度量的变化
    project_path = get_project_path()
    plot_folder = "plot"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    metrics_folder = os.path.join(project_path, "output", "evaluation")
    metric_jsons = read_json(metrics_folder)

    conditions = {"env_types": ["hybrid"],
                  "pedestrian_dynamic_num": [1, 2, 3, 4],

                  "folder_name": ["CATAttentionNetSpacial_no_large_office",
                                  "CATAttentionNetTemporalSpacialLSTM4_uncomfortable_distance"],
                  "pedestrian_max_speed_range": [0.45],

                  }
    metrics = "success_rate"

    values = extract_metric(metric_jsons, conditions, metrics)

    x_names = conditions["pedestrian_dynamic_num"]
    label_names = ["interaction module", "temporal and interaction module"]
    values = np.reshape(values, (len(x_names), len(label_names)))
    color_index = [8, 9]
    colors = [COLORS[i] for i in color_index]
    assert len(colors) == len(label_names)
    y_label = "Success Rate"
    x_label = "Pedestrian number"
    save_path = os.path.join(plot_folder, "ablation_study_pedestrian_number.png")

    title = "Success rate of different attention network about pedestrian number"
    bar_line_graph(x_names, label_names, values, colors, title, y_label, save_path, width=8, height=4, x_label=x_label)


def test_compare_success_rate_of_pedestrian_speed():
    # 比较同一个方法在不同的环境配置下，如不同的人数，不同的行人速度下，度量的变化
    project_path = get_project_path()
    plot_folder = "plot"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    metrics_folder = os.path.join(project_path, "output", "evaluation")
    metric_jsons = read_json(metrics_folder)

    conditions = {"env_types": ["hybrid"],
                  "pedestrian_max_speed_range": [0.15, 0.25, 0.35, 0.45],

                  "pedestrian_dynamic_num": [2],

                  "folder_name": ["CATAttentionNetSpacial_no_large_office",
                                  "CATAttentionNetTemporalSpacialLSTM4_uncomfortable_distance"],

                  }
    metrics = "success_rate"

    values = extract_metric(metric_jsons, conditions, metrics)

    x_names = conditions["pedestrian_max_speed_range"]
    label_names = ["interaction module", "temporal and interaction module"]
    values = np.reshape(values, (len(x_names), len(label_names)))
    color_index = [8, 9]
    colors = [COLORS[i] for i in color_index]
    assert len(colors) == len(label_names)
    y_label = "Success Rate"
    x_label = "Pedestrian speed (m/s)"
    save_path = os.path.join(plot_folder, "ablation_study_pedestrian_speed.png")

    title = "Success rate of different attention network about pedestrian speed"
    bar_line_graph(x_names, label_names, values, colors, title, y_label, save_path, width=8, height=4, x_label=x_label)
