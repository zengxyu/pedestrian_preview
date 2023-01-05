#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : navigation_icra 
    @Author  : Xiangyu Zeng
    @Date    : 8/8/22 1:59 PM 
    @Description    :
        
===========================================
"""

from agents.network.network_base import *

model_params = {
    'cnn': [32, 16],
    'mlp': [256, 128, 64],
    'kernel_sizes': [5, 3],
    'strides': [2, 2],
    'mlp_waypoints': [64, 40, 30],
    'mlp_values': [256, 128, 64],

}


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cnn_dims = model_params["cnn"]
        self.kernel_sizes = model_params["kernel_sizes"]
        self.strides = model_params["strides"]
        self.seq_len = len(kwargs["seq_indexes"])

        self.num_points = kwargs["num_points"]
        self.dim_points = kwargs["dim_points"]
        self.num_waypoints = kwargs["num_waypoints"]
        self.dim_waypoints = kwargs["dim_waypoints"]
        self.mlp_waypoints_dims = model_params["mlp_waypoints"]

        self.mlp_dims = model_params["mlp"]
        # 100 x 100
        self.cnns = build_cnns(self.seq_len, self.cnn_dims, self.kernel_sizes, self.strides)

        self.mlps = build_mlp(1440, self.mlp_dims, activate_last_layer=False)

        self.mlp_waypoints = build_mlp(self.num_waypoints * self.dim_waypoints, self.mlp_waypoints_dims, False)


class CNNActorNet(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)
        self.visualize_attention = kwargs["visualize_attention"]

        mlp_values_dims = model_params["mlp_values"]

        self.td3_end = nn.Sequential(nn.Tanh(), pfrl.policies.DeterministicHead())

        self.mlp_action = build_mlp(
            self.mlp_dims[-1] + self.mlp_waypoints_dims[-1],
            mlp_values_dims + [self.n_actions],
            activate_last_layer=False,
        )

    def forward(self, x):
        x = x.float()

        batch_size = x.shape[0]

        temporal_coordinates = x[:, : self.seq_len * self.num_points * self.dim_points].float()
        waypoints = x[:, self.seq_len * self.num_points * self.dim_points:].float()

        out_temporal_coordinates = temporal_coordinates.reshape(batch_size, self.seq_len, -1)
        out_temporal_coordinates = self.cnns(out_temporal_coordinates)
        out_temporal_coordinates = out_temporal_coordinates.reshape(batch_size, -1)
        out_temporal_coordinates = self.mlps(out_temporal_coordinates)

        out_waypoints = waypoints.reshape(batch_size, -1)
        out_waypoints = self.mlp_waypoints(out_waypoints)
        out = torch.cat([out_temporal_coordinates, out_waypoints], dim=-1)
        actions = self.mlp_action(out)
        out = self.td3_end(actions)
        # show_attention(meas_coordinates.clone().detach().numpy(), attention_scores.clone().detach().numpy(),
        #                self.ray_part, self.ray_num_per_part,
        #                "ddpg_attention", self.visualize_attention)
        return out


class CNNCriticNet(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)
        mlp_values_dims = model_params["mlp_values"]

        self.mlp_value = build_mlp(
            self.mlp_dims[-1] + self.mlp_waypoints_dims[-1] + self.n_actions,
            mlp_values_dims + [1],
            activate_last_layer=False,
        )

    def forward(self, x):
        x, action = x
        x = x.float()
        action = action.float()
        batch_size = x.shape[0]

        temporal_coordinates = x[:, : self.seq_len * self.num_points * self.dim_points].float()
        waypoints = x[:, self.seq_len * self.num_points * self.dim_points:].float()

        out_temporal_coordinates = temporal_coordinates.reshape(batch_size, self.seq_len, -1)
        out_temporal_coordinates = self.cnns(out_temporal_coordinates)
        out_temporal_coordinates = out_temporal_coordinates.reshape(batch_size, -1)
        out_temporal_coordinates = self.mlps(out_temporal_coordinates)

        out_waypoints = waypoints.reshape(batch_size, -1)
        out_waypoints = self.mlp_waypoints(out_waypoints)
        out = torch.cat([out_temporal_coordinates, out_waypoints, action], dim=-1)
        return self.mlp_value(out)
