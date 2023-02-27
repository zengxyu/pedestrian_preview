from agents.network.network_base import *
from torch import distributions, nn

from agents.network.network_head import build_head

model_params = {
    'dim_mlp': [512, 256, 128],
    'mlp_values': [128, 64, 32],

}


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.seq_len = kwargs["seq_len"]
        self.dim_mlp = model_params["dim_mlp"]
        self.mlp = build_mlp(102 * self.seq_len, self.dim_mlp, activate_last_layer=False)


class SimpleLidarMlpActor(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, agent_type, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)

        mlp_values_dims = model_params["mlp_values"]

        self.head = build_head(agent_type, action_space)

        if agent_type == "sac":
            self.mlp_action = build_mlp(self.dim_mlp[-1] ,
                                        mlp_values_dims + [self.n_actions * 2],
                                        activate_last_layer=False,
                                        )
        else:
            self.mlp_action = build_mlp(self.dim_mlp[-1],
                                        mlp_values_dims + [self.n_actions],
                                        activate_last_layer=False,
                                        )

    def forward(self, x):
        x = x.float()
        # lidar = x[:, :100]
        # target = x[:, 100:]
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

        self.mlp_value = build_mlp(self.dim_mlp[-1]  + self.n_actions,
                                   mlp_values_dims + [1],
                                   activate_last_layer=False,
                                   )

    def forward(self, x):
        x, action = x
        x = x.float()
        # lidar = x[:, :100]
        # target = x[:, 100:]
        out = self.mlp(x)
        out = torch.cat((out,  action), dim=1)

        return self.mlp_value(out)
