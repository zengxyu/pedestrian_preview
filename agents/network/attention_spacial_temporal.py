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
}


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.num_rays = kwargs["num_rays"]
        self.num_points = kwargs["num_points"]
        self.dim_points = kwargs["dim_points"]

        self.num_waypoints = kwargs["num_waypoints"]
        self.dim_waypoints = kwargs["dim_waypoints"]

        self.mlp_spacial_k_dims = model_params["mlp_spacial_k"]
        self.mlp_spacial_v_dims = model_params["mlp_spacial_v"]
        self.mlp_spacial_attention_dims = model_params["mlp_spacial_attention"]
        self.mlp_waypoints_dims = model_params["mlp_waypoints"]

        self.num_points_per_parts = int(round(self.num_points / self.num_rays))
        self.mlp_spacial_k = build_mlp((2 + self.num_waypoints) * self.dim_points,
                                       self.mlp_spacial_k_dims, True)
        self.mlp_spacial_v = build_mlp(self.mlp_spacial_k_dims[-1], self.mlp_spacial_v_dims, False)
        self.mlp_spacial_attention = build_mlp(self.mlp_spacial_k_dims[-1], self.mlp_spacial_attention_dims, False)

        self.mlp_spacial_k1 = build_mlp((self.num_points_per_parts + self.num_waypoints) * self.dim_points,
                                        self.mlp_spacial_k_dims, True)
        self.mlp_spacial_v1 = build_mlp(self.mlp_spacial_k_dims[-1], self.mlp_spacial_v_dims, False)
        self.mlp_spacial_attention1 = build_mlp(self.mlp_spacial_k_dims[-1], self.mlp_spacial_attention_dims, False)


class STAttentionActor(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)

        mlp_values_dims = model_params["mlp_values"]

        self.td3_end = nn.Sequential(nn.Tanh(), pfrl.policies.DeterministicHead())

        self.mlp_action = build_mlp(
            self.mlp_spacial_v_dims[-1] * 2,
            mlp_values_dims + [self.n_actions],
            activate_last_layer=False,
        )
        self.scores_spacial = None
        self.scores_temporal = None
        self.coordinates = None

    def forward(self, x):
        temporal_coordinates = x[0].float()
        coordinates = x[1].float()
        waypoints = x[2].float()
        batch_size = temporal_coordinates.shape[0]

        # 64 x 30, 2 x -
        out_temporal_coordinates = temporal_coordinates.reshape((batch_size, self.num_rays, -1))

        # 64 x 30, 2
        out_coordinates = coordinates.reshape((batch_size, self.num_rays, -1))
        # 64 x 30, 10
        waypoints = waypoints.reshape((batch_size, 1, -1)).repeat((1, self.num_rays, 1))

        inputs = torch.cat([out_temporal_coordinates, waypoints], dim=2)
        spacial_k_out = self.mlp_spacial_k(inputs)
        spacial_v_out = self.mlp_spacial_v(spacial_k_out)
        spacial_attention_scores = self.mlp_spacial_attention(spacial_k_out)
        spacial_v_out = spacial_v_out.view(batch_size, self.num_rays, -1)
        spacial_attention_scores = spacial_attention_scores.view(batch_size, self.num_rays, 1)
        # batch_size*seq_len,  v_dim
        spacial_weighted_features, scores = compute_spacial_weighted_feature(spacial_attention_scores, spacial_v_out)
        self.scores_temporal = scores
        self.coordinates = coordinates

        spacial_weighted_features = spacial_weighted_features.view(batch_size,
                                                                   self.mlp_spacial_v_dims[-1])
        inputs1 = torch.cat([out_coordinates, waypoints], dim=2)
        spacial_k_out1 = self.mlp_spacial_k1(inputs1)
        spacial_v_out1 = self.mlp_spacial_v1(spacial_k_out1)
        spacial_attention_scores1 = self.mlp_spacial_attention1(spacial_k_out1)
        spacial_v_out1 = spacial_v_out1.view(batch_size, self.num_rays, -1)
        spacial_attention_scores1 = spacial_attention_scores1.view(batch_size, self.num_rays, 1)
        # batch_size*seq_len,  v_dim
        spacial_weighted_features1, scores1 = compute_spacial_weighted_feature(spacial_attention_scores1,
                                                                               spacial_v_out1)
        self.scores_spacial = scores1
        spacial_weighted_features1 = spacial_weighted_features1.view(batch_size,
                                                                     self.mlp_spacial_v_dims[-1])

        # contact the embedding of ray measurements and waypoints, predict action values
        features = torch.cat([spacial_weighted_features, spacial_weighted_features1], dim=1)
        actions = self.mlp_action(features)
        out = self.td3_end(actions)
        return out


class STAttentionCritic(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)
        mlp_values_dims = model_params["mlp_values"]

        self.mlp_value = build_mlp(
            self.mlp_spacial_v_dims[-1] * 2 + self.n_actions,
            mlp_values_dims + [1],
            activate_last_layer=False,
        )

    def forward(self, x):
        x, action = x
        action = action.float()

        temporal_coordinates = x[0].float()
        coordinates = x[1].float()
        waypoints = x[2].float()
        batch_size = temporal_coordinates.shape[0]

        # 64 x 30, 2 x -
        out_temporal_coordinates = temporal_coordinates.reshape((batch_size, self.num_rays, -1))

        # 64 x 30, 2
        out_coordinates = coordinates.reshape((batch_size, self.num_rays, -1))
        # 64 x 30, 10
        waypoints = waypoints.reshape((batch_size, 1, -1)).repeat((1, self.num_rays, 1))

        inputs = torch.cat([out_temporal_coordinates, waypoints], dim=2)
        spacial_k_out = self.mlp_spacial_k(inputs)
        spacial_v_out = self.mlp_spacial_v(spacial_k_out)
        spacial_attention_scores = self.mlp_spacial_attention(spacial_k_out)
        spacial_v_out = spacial_v_out.view(batch_size, self.num_rays, -1)
        spacial_attention_scores = spacial_attention_scores.view(batch_size, self.num_rays, 1)
        # batch_size*seq_len,  v_dim
        spacial_weighted_features, scores = compute_spacial_weighted_feature(spacial_attention_scores, spacial_v_out)
        spacial_weighted_features = spacial_weighted_features.view(batch_size,
                                                                   self.mlp_spacial_v_dims[-1])
        inputs1 = torch.cat([out_coordinates, waypoints], dim=2)
        spacial_k_out1 = self.mlp_spacial_k1(inputs1)
        spacial_v_out1 = self.mlp_spacial_v1(spacial_k_out1)
        spacial_attention_scores1 = self.mlp_spacial_attention1(spacial_k_out1)
        spacial_v_out1 = spacial_v_out1.view(batch_size, self.num_rays, -1)
        spacial_attention_scores1 = spacial_attention_scores1.view(batch_size, self.num_rays, 1)
        # batch_size*seq_len,  v_dim
        spacial_weighted_features1, scores1 = compute_spacial_weighted_feature(spacial_attention_scores1,
                                                                               spacial_v_out1)
        spacial_weighted_features1 = spacial_weighted_features1.view(batch_size,
                                                                     self.mlp_spacial_v_dims[-1])

        # contact the embedding of ray measurements and waypoints, predict action values

        out = torch.cat([spacial_weighted_features, spacial_weighted_features1, action], dim=1)
        return self.mlp_value(out)
