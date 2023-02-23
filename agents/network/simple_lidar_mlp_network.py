from agents.network.network_base import *
from torch import distributions, nn

from agents.network.network_head import build_head

model_params = {
    'dim_mlp': [80, 128, 256],
    'mlp_values': [64, 32, 16],

}


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.seq_len = kwargs["seq_len"]
        self.dim_mlp = model_params["dim_mlp"]
        self.mlp = build_mlp((20 + 2) * self.seq_len, self.dim_mlp, activate_last_layer=False)


class SimpleLidarMlpActor(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, agent_type, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)

        mlp_values_dims = model_params["mlp_values"]

        self.head = build_head(agent_type, action_space)

        self.mlp_action = build_mlp(self.dim_mlp[-1],
                                    mlp_values_dims + [self.n_actions * 2],
                                    activate_last_layer=False,
                                    )

    def forward(self, x):
        x = x.float()
        out = self.mlp(x)
        out = self.mlp_action(out)
        out = self.head(out)
        return out


class SimpleLidarMlpCritic(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, agent_type, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)
        mlp_values_dims = model_params["mlp_values"]

        self.mlp_value = build_mlp(self.dim_mlp[-1] + self.n_actions,
                                   mlp_values_dims + [1],
                                   activate_last_layer=False,
                                   )

    def forward(self, x):
        x, action = x
        x = x.float()
        out = self.mlp(x)
        out = torch.cat((out, action), dim=1)

        return self.mlp_value(out)