from agents.network.network_base import *

model_params = {
    'cnn': [32, 16, 8],
    'mlp': [8, 16],
    'kernel_sizes': [3, 3, 3],
    'strides': [2, 2, 2],
    'mlp_waypoints': [64, 40, 30],
    'mlp_values': [256, 128, 64],

}


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.cnn_dims = model_params["cnn"]
        self.kernel_sizes = model_params["kernel_sizes"]
        self.strides = model_params["strides"]
        self.mlp_dims = model_params["mlp"]

        self.cnn = build_cnns_2d(1, self.cnn_dims, self.kernel_sizes, self.strides)
        self.mlp_relative_position = build_mlp(3, self.mlp_dims, activate_last_layer=False)


class SimpleMlpActor(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)

        mlp_values_dims = model_params["mlp_values"]

        self.td3_end = nn.Sequential(nn.Tanh(), pfrl.policies.DeterministicHead())

        self.mlp_action = build_mlp(16,
                                    mlp_values_dims + [self.n_actions],
                                    activate_last_layer=False,
                                    )

    def forward(self, x):
        # depth_image = x[0].float()
        relative_position = x[1].float()

        # batch_size = depth_image.size(0)
        # out1 = self.cnn(depth_image)
        # out1 = out1.reshape((batch_size, -1))
        out2 = self.mlp_relative_position(relative_position)
        # # out =
        # out = torch.cat((out1, out2), dim=1)

        out = self.mlp_action(out2)
        out = self.td3_end(out)
        return out


class SimpleMlpCritic(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)
        mlp_values_dims = model_params["mlp_values"]

        self.mlp_value = build_mlp(16 + self.n_actions,
                                   mlp_values_dims + [1],
                                   activate_last_layer=False,
                                   )

    def forward(self, x):
        x, action = x
        # depth_image = x[0].float()
        relative_position = x[1].float()

        # batch_size = depth_image.size(0)
        # out1 = self.cnn(depth_image)
        # out1 = out1.reshape((batch_size, -1))
        out2 = self.mlp_relative_position(relative_position)
        # out =
        out = torch.cat((out2, action), dim=1)

        return self.mlp_value(out)
