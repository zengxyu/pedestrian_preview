#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 6/17/22 11:55 AM
    @Description    :

===========================================
"""

from agents.network.network_base import *

model_params = {
    'mlp_spacial_k': [256, 128, 64],
    'mlp_spacial_v': [80, 50, 30],
    'mlp_spacial_attention': [60, 50, 1],
    'mlp_waypoints': [64, 40, 30],

    'mlp_values': [128, 64, 64],
    'ray_parts': 15,
}


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.num_points = kwargs["num_points"]
        self.dim_points = kwargs["dim_points"]

        self.dim_robot_states = kwargs["robot_state"]
        self.num_waypoints = kwargs["num_waypoints"]
        self.dim_waypoints = kwargs["dim_waypoints"]
        self.seq_len = len(kwargs["seq_indexes"])

        self.num_parts = model_params["ray_parts"]
        self.points_num_per_part = int(self.num_points / self.num_parts)
        assert self.points_num_per_part * self.num_parts == self.num_points, "Ray number must be divisible."

        self.mlp_spacial_k_dims = model_params["mlp_spacial_k"]
        self.mlp_spacial_v_dims = model_params["mlp_spacial_v"]
        self.mlp_spacial_attention_dims = model_params["mlp_spacial_attention"]
        self.mlp_waypoints_dims = model_params["mlp_waypoints"]

        self.mlp_spacial_k = build_mlp(
            self.points_num_per_part * self.dim_points + self.num_waypoints * self.dim_waypoints + self.dim_robot_states,
            self.mlp_spacial_k_dims, True)
        self.mlp_spacial_v = build_mlp(self.mlp_spacial_k_dims[-1], self.mlp_spacial_v_dims, False)
        self.mlp_spacial_attention = build_mlp(self.mlp_spacial_k_dims[-1], self.mlp_spacial_attention_dims, False)

        self.mlp_waypoints = build_mlp(self.num_waypoints, self.mlp_waypoints_dims, False)


class AttentionSpacialActor(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)

        mlp_values_dims = model_params["mlp_values"]

        self.td3_end = nn.Sequential(nn.Tanh(), pfrl.policies.DeterministicHead())

        self.mlp_action = build_mlp(
            self.mlp_spacial_v_dims[-1] * self.seq_len,
            mlp_values_dims + [self.n_actions],
            activate_last_layer=False,
        )
        self.count = 0
        self.visualize_attention = kwargs["visualize_attention"]

    def forward(self, x):
        if self.visualize_attention:
            self.count += 1
        x = x.float()
        batch_size = x.shape[0]
        obs_coordinates = x[:, :self.num_points * self.dim_points]
        waypoints = x[:, self.num_points * self.dim_points:]
        # show_waypoints(waypoints.clone().detach().numpy(), self.visualize_attention)

        obs_coordinates = obs_coordinates.reshape(batch_size, self.num_parts,
                                                  self.points_num_per_part * self.dim_points)
        waypoints = waypoints.reshape((batch_size, 1, -1))

        waypoints = waypoints.repeat((1, self.num_parts, 1))
        coordinates_waypoints = torch.cat([obs_coordinates, waypoints], dim=2)
        spacial_k_out = self.mlp_spacial_k(coordinates_waypoints)
        spacial_v_out = self.mlp_spacial_v(spacial_k_out)
        spacial_attention_scores = self.mlp_spacial_attention(spacial_k_out)
        spacial_v_out = spacial_v_out.view(batch_size, self.num_parts, -1)
        spacial_attention_scores = spacial_attention_scores.view(batch_size, self.num_parts, 1)
        # batch_size*seq_len,  v_dim
        spacial_weighted_features, scores = compute_spacial_weighted_feature(spacial_attention_scores, spacial_v_out)
        spacial_weighted_features = spacial_weighted_features.view(batch_size,
                                                                   self.mlp_spacial_v_dims[-1])
        # show_attention(obs_coordinates.clone().detach().numpy(),
        #                x[:, self.num_points * self.dim_points:].clone().detach().numpy(),
        #                scores.clone().detach().numpy(),
        #                self.num_parts, self.points_num_per_part, "spacial_count_{}".format(self.count),
        #                self.visualize_attention)
        # show_coordinates_part(obs_coordinates.clone().detach().numpy(),
        #                       self.ray_part, self.ray_num_per_part,
        #                       self.visualize_attention)
        # contact the embedding of ray measurements and waypoints, predict action values
        actions = self.mlp_action(spacial_weighted_features)
        out = self.td3_end(actions)
        return out


class AttentionSpacialCritic(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)
        mlp_values_dims = model_params["mlp_values"]

        self.mlp_value = build_mlp(
            self.mlp_spacial_v_dims[-1] * self.seq_len + self.n_actions,
            mlp_values_dims + [1],
            activate_last_layer=False,
        )

    def forward(self, x):
        x, action = x
        x = x.float()
        action = action.float()

        batch_size = x.shape[0]
        obs_coordinates = x[:, :self.num_points * self.dim_points]
        waypoints = x[:, self.num_points * self.dim_points:]
        obs_coordinates = obs_coordinates.reshape(batch_size, self.num_parts,
                                                  self.points_num_per_part * self.dim_points)
        waypoints = waypoints.reshape((batch_size, 1, -1))

        waypoints = waypoints.repeat((1, self.num_parts, 1))
        coordinates_waypoints = torch.cat([obs_coordinates, waypoints], dim=2)

        spacial_k_out = self.mlp_spacial_k(coordinates_waypoints)
        spacial_v_out = self.mlp_spacial_v(spacial_k_out)
        spacial_attention_scores = self.mlp_spacial_attention(spacial_k_out)
        spacial_v_out = spacial_v_out.view(batch_size, self.num_parts, -1)
        spacial_attention_scores = spacial_attention_scores.view(batch_size, self.num_parts, 1)
        # batch_size*seq_len,  v_dim
        spacial_weighted_features, scores = compute_spacial_weighted_feature(spacial_attention_scores, spacial_v_out)
        spacial_weighted_features = spacial_weighted_features.view(batch_size, -1)

        # mlp_waypoints_out = self.mlp_waypoints(waypoints)
        out = torch.cat([spacial_weighted_features, action], dim=1)
        return self.mlp_value(out)
