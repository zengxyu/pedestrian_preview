from agents.network.network_base import *
from torch import distributions, nn

from agents.network.network_head import build_head

model_params = {
    'cnn': [32, 16, 8],
    'kernel_sizes': [3, 3, 3],
    'strides': [2, 2, 2],
    'mlp': [8, 16, 32],
    'mlp_depth': [32, 128, 64, 32],
    'mlp_values': [256, 128, 64],

}


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.cnn_dims = model_params["cnn"]
        self.kernel_sizes = model_params["kernel_sizes"]
        self.strides = model_params["strides"]
        self.mlp_dims = model_params["mlp"]
        self.mlp_depth_dims = model_params["mlp_depth"]
        self.cnn = build_cnns_2d(1, self.cnn_dims, self.kernel_sizes, self.strides)
        self.mlp_relative_position = build_mlp(3, self.mlp_dims, activate_last_layer=False)
        self.mlp_depth = build_mlp(10, self.mlp_depth_dims, activate_last_layer=False)


class SimpleMlpActor(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, agent_type, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)

        mlp_values_dims = model_params["mlp_values"]

        self.head = build_head(agent_type, action_space)

        self.mlp_action = build_mlp(self.mlp_depth_dims[-1] + self.mlp_dims[-1],
                                    mlp_values_dims + [self.n_actions * 2],
                                    activate_last_layer=False,
                                    )

    def forward(self, x):
        depth_image = x[0].float()
        relative_position = x[1].float()

        out1 = self.mlp_depth(depth_image)
        out2 = self.mlp_relative_position(relative_position)
        out = torch.cat((out1, out2), dim=1)
        out = self.mlp_action(out)
        out = self.head(out)
        return out


class SimpleMlpCritic(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, agent_type, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)
        mlp_values_dims = model_params["mlp_values"]

        self.mlp_value = build_mlp(self.mlp_depth_dims[-1] + self.mlp_dims[-1] + self.n_actions,
                                   mlp_values_dims + [1],
                                   activate_last_layer=False,
                                   )

    def forward(self, x):
        x, action = x
        depth_image = x[0].float()
        relative_position = x[1].float()
        out1 = self.mlp_depth(depth_image)
        out2 = self.mlp_relative_position(relative_position)
        out = torch.cat((out1, out2, action), dim=1)

        return self.mlp_value(out)
