#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 6/21/22 11:25 AM 
    @Description    :
        
===========================================
"""
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

import pfrl
from torch import nn
from typing import Dict, List
from torch.nn.functional import softmax

from utils.fo_utility import get_project_path


def build_mlp(
        input_dim: int,
        mlp_dims: List[int],
        activate_last_layer=False,
        activate_func=nn.ReLU(),
        last_layer_activate_func=None,
):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        # init parameter
        torch.nn.init.xavier_uniform_(layers[-1].weight)

        if i != len(mlp_dims) - 2:
            layers.append(activate_func)
        if i == len(mlp_dims) - 2 and activate_last_layer:
            func = last_layer_activate_func if last_layer_activate_func is not None else activate_func
            layers.append(func)
    net = nn.Sequential(*layers)
    return net


def build_cnns(input_dim, cnn_dims, kernel_sizes, strides, activation_class=nn.ReLU):
    layers = []
    cnn_dims = [input_dim] + cnn_dims
    for i in range(len(cnn_dims) - 1):
        layers.append(torch.nn.Conv1d(cnn_dims[i], cnn_dims[i + 1], kernel_sizes[i], stride=strides[i], padding=1))
        torch.nn.init.xavier_uniform_(layers[-1].weight)
        layers.append(activation_class())
    net = nn.Sequential(*layers)
    return net


def compute_spacial_weighted_feature(attention_scores, features):
    # attention_scores: (batch_size, self.seq_len, self.ray_part, 1)
    # features: (batch_size, self.seq_len, self.ray_part, v_dim)
    # 选序列上的最大值
    attention_weights = softmax(attention_scores, dim=1)
    weighted_feature = torch.sum(torch.mul(attention_weights, features), dim=1)
    # features: (batch_size, self.seq_len, v_dim)

    return weighted_feature, attention_weights


def compute_temporal_weighted_feature(attention_scores, features):
    attention_weights = softmax(attention_scores, dim=1)
    weighted_feature = torch.sum(torch.mul(attention_weights, features), dim=1)
    return weighted_feature, attention_weights


def compute_qkv_feature(q, k, v):
    # q: (bs, vector_dim), (bs, k_dim)
    # k: (bs, ray_part , k_dim)
    # v: (bs, ray_part , v_dim)
    # compute q与k中每一个都要点乘， 计算重要性，
    # q: (bs, ray_part, k_dim)
    # q = q.repeat((1, self.ray_part, 1))
    # score : (bs, ray_part , 1)
    score = torch.sum(torch.mul(q, k), dim=2)
    score = torch.softmax(score, dim=1).unsqueeze(2)
    weighted_feature = torch.sum(torch.mul(score, v), dim=1)
    return weighted_feature, score


def visualize_polar(thetas, dists, title="Plot lidar polar positions", c='r'):
    plt.polar(thetas, dists, 'ro', lw=2, c=c)
    plt.title(title)
    plt.ylim(0, 1)
    plt.show()


def visualize_cartesian(obs_coordinates_list, waypoints=None, colors=None, title="Plot lidar cartesian positions",
                        c='r'):
    for obs_coordinates, color in zip(obs_coordinates_list, colors):
        for obs_coordinate in obs_coordinates:
            plt.scatter(obs_coordinate[0], obs_coordinate[1], c=color, edgecolors='#cccccc', s=200)

    # 画路径点
    if waypoints is not None:
        # deformed路径点
        plt.plot(waypoints[:, 0], waypoints[:, 1], 'b', label="Deformed path")

        for waypoint in waypoints:
            plt.scatter(waypoint[0], waypoint[1], marker='x', c='b', edgecolors='#cccccc', s=20)

    plt.scatter(0, 0, marker='^', s=200, c='k')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)

    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()
    folder = os.path.join(get_project_path(), "visualize_utilities", "plot_attention")
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, os.path.join(title + ".png")))
    plt.show()
    plt.clf()


def show_attention(obs_coordinates, waypoints, score, ray_part, ray_num_per_part, title, visualize_attention):
    # obs_coordinates 1 x 10 x 10
    # score 1 x 10 x 1
    if visualize_attention:
        obs_coordinates = obs_coordinates.clone().detach().numpy()
        waypoints = waypoints[0][0].clone().detach().numpy()
        score = score.clone().detach().numpy()

        rgba = [0, 0, 0, 0]
        base_rgba = [0, 1, 1, 0]

        colors = []
        score = score.squeeze()
        score = score / max(score)
        for s in score:
            rgba = s * np.array(base_rgba)
            colors.append(rgba.copy())
        colors = 1 - np.array(colors)

        obs_coordinates = obs_coordinates.reshape((ray_part, ray_num_per_part, -1))
        waypoints = waypoints.reshape((-1, 2))
        # 画坐标点 和 waypoints
        visualize_cartesian(obs_coordinates, waypoints, colors, title)
        # 画机器人所在位置


def show_attention_temporal(obs_coordinates, waypoints, score, ray_part, ray_num_per_part, title, visualize_attention):
    # obs_coordinates 1 x 10 x 10
    # score 1 x 10 x 1
    if visualize_attention:
        obs_coordinates = obs_coordinates.clone().detach().numpy()
        waypoints = waypoints[0][0].clone().detach().numpy()
        score = score.clone().detach().numpy()

        rgba = [0, 0, 0, 0]
        base_rgba = [0, 1, 1, 0]

        colors = []
        score = score.squeeze()
        score = score / max(score)
        for s in score:
            rgba = s * np.array(base_rgba)
            colors.append(rgba.copy())
        colors = 1 - np.array(colors)

        obs_coordinates = obs_coordinates.reshape((ray_part, 2, 2))
        waypoints = waypoints.reshape((-1, 2))
        # 画坐标点 和 waypoints
        visualize_cartesian(obs_coordinates, waypoints, colors, title)
        # 画机器人所在位置


def show_coordinates_part(obs_coordinates, ray_part, ray_num_per_part, visualize_attention):
    if visualize_attention:
        base_rgba = [0, 1, 1, 0]
        ss = np.eye(N=ray_part)
        for s1 in ss:
            colors = []
            for s2 in s1:
                rgba = s2 * np.array(base_rgba)
                colors.append(rgba.copy())
            colors = 1 - np.array(colors)
            obs_coordinates = obs_coordinates.reshape((ray_part, ray_num_per_part, 2))
            visualize_cartesian(obs_coordinates, colors, "coordinates_part_{}".format(np.argmax(s1)))
        visualize_cartesian(obs_coordinates, [1 - np.array(base_rgba) for i in range(ray_part)], "coordinates_part_all")


def show_waypoints(waypoints, visualize_attention):
    if visualize_attention:
        base_rgba = [0, 1, 1, 0]
        waypoints = waypoints.reshape((1, -1, 2))

        visualize_cartesian(waypoints, [1 - np.array(base_rgba)], "waypoints")
