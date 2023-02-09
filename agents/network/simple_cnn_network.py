import torch

from agents.network.network_base import *
from agents.network.network_head import *
from ncps.wirings import AutoNCP
model_params = {
    'cnn': [32, 64, 32],
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
        self.seq_len = kwargs["seq_len"]
        self.position_len = 2
        self.cnn = build_cnns_2d(self.seq_len, self.cnn_dims, self.kernel_sizes, self.strides)
        # self.mlp_relative_position = build_mlp(3, self.mlp_dims, activate_last_layer=False)

class SimpleCnnNcpActor(BaseModel):
    def __init__(self, agent_type, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)

        self.head = build_head(agent_type, action_space)
        self.cnn = build_cnns_2d(1, self.cnn_dims, self.kernel_sizes, self.strides)
        self.wirings = AutoNCP(48, 4)
        self.rnn = build_ncpltc(1283, wirings=self.wirings)

    def forward(self, x, hx=None):
        image = x[0].float()
        relative_position = x[1].float()

        batch_size = image.size(0)
        seq_len = image.size(1)

        relative_position = relative_position.view(batch_size, seq_len, -1)

        image = image.view(batch_size * seq_len, 1, *image.shape[2:])
        out1 = self.cnn(image)
        out1 = out1.view(batch_size, seq_len, *out1.shape[1:])
        out1 = out1.reshape(batch_size, seq_len, -1)
        out = torch.cat((out1, relative_position), dim=2)

        out, hx = self.rnn(out, hx)
        out = out.mean(dim=1)
        out = self.head(out)
        return out

class SimpleCnnActor(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, agent_type, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)

        mlp_values_dims = model_params["mlp_values"]

        self.head = build_head(agent_type, action_space)

        self.mlp_action = build_mlp(1280 + 3 * self.seq_len,
                                    mlp_values_dims + [self.n_actions * 2],
                                    activate_last_layer=False,
                                    )

    def forward(self, x):
        depth_image = x[0].float()
        relative_position = x[1].float()

        batch_size = depth_image.size(0)
        out1 = self.cnn(depth_image)
        out1 = out1.reshape((batch_size, -1))
        # out2 = self.mlp_relative_position(relative_position)
        out = torch.cat((out1, relative_position), dim=1)

        out = self.mlp_action(out)
        out = self.head(out)
        return out


class SimpleCnnCritic(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, agent_type, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)
        mlp_values_dims = model_params["mlp_values"]

        self.mlp_value = build_mlp(1280 + 3 * self.seq_len + self.n_actions,
                                   mlp_values_dims + [1],
                                   activate_last_layer=False,
                                   )

    def forward(self, x):
        x, action = x
        depth_image = x[0].float()
        relative_position = x[1].float()

        batch_size = depth_image.size(0)
        out1 = self.cnn(depth_image)
        out1 = out1.reshape((batch_size, -1))
        # out =
        out = torch.cat((out1, relative_position, action), dim=1)

        return self.mlp_value(out)
