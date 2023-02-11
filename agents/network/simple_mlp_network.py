from agents.network.network_base import *
from torch import distributions, nn

from agents.network.network_head import build_head

model_params = {
    'dim_relative_positions': [32],
    'dim_rows': [512, 256, 64, 32],
    'mlp_values': [64, 32, 16],

}


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.image_seq_len = kwargs["image_seq_len"]
        self.pose_seq_len = kwargs["pose_seq_len"]
        self.dim_relative_positions = model_params["dim_relative_positions"]
        self.dim_rows = model_params["dim_rows"]
        self.mlp_relative_position = build_mlp(3 * self.pose_seq_len, self.dim_relative_positions, activate_last_layer=False)
        self.mlp_row = build_mlp(120 * self.image_seq_len, self.dim_rows, activate_last_layer=False)


class SimpleMlpActor(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, agent_type, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)

        mlp_values_dims = model_params["mlp_values"]

        self.head = build_head(agent_type, action_space)

        self.mlp_action = build_mlp(self.dim_rows[-1] + self.dim_relative_positions[-1],
                                    mlp_values_dims + [self.n_actions * 2],
                                    activate_last_layer=False,
                                    )

    def forward(self, x):
        depth_image = x[0].float().squeeze(1)
        relative_position = x[1].float()

        out1 = self.mlp_row(depth_image)
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

        self.mlp_value = build_mlp(self.dim_rows[-1] + self.dim_relative_positions[-1] + self.n_actions,
                                   mlp_values_dims + [1],
                                   activate_last_layer=False,
                                   )

    def forward(self, x):
        x, action = x
        depth_image = x[0].float().squeeze(1)
        relative_position = x[1].float()
        out1 = self.mlp_row(depth_image)
        out2 = self.mlp_relative_position(relative_position)
        out = torch.cat((out1, out2, action), dim=1)

        return self.mlp_value(out)
