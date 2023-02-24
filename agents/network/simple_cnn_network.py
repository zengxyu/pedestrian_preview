import torch

from agents.network.network_base import *
from agents.network.network_head import *
from environment.sensors.vision_sensor import ImageMode

from ncps.wirings import AutoNCP

from utils.convolution_width_utility import compute_conv_out_width

model_params = {
    'cnn': [128, 64, 64, 32],
    'kernel_sizes': [3, 3, 3, 3],
    'strides': [2, 2, 2, 2],
    'mlp_values': [512, 128, 64],
    'mlp_relative_position': [36, 64]
}


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.cnn_dims = model_params["cnn"]
        self.kernel_sizes = model_params["kernel_sizes"]
        self.strides = model_params["strides"]
        self.image_seq_len = kwargs["image_seq_len"]
        self.pose_seq_len = kwargs["pose_seq_len"]
        self.image_mode = kwargs["image_mode"]
        self.dim_relative_position = model_params["mlp_relative_position"]
        h_out = compute_conv_out_width(kwargs["image_h"], k=3, s=2, p=1, iter=len(self.cnn_dims))
        w_out = compute_conv_out_width(kwargs["image_w"], k=3, s=2, p=1, iter=len(self.cnn_dims))
        self.dim_cnn_out_flatten = h_out * w_out * self.cnn_dims[-1]
        if self.image_mode == ImageMode.MULTI_ROW_MULTI_SENSOR:
            input_channel = self.image_seq_len * 4
        elif self.image_mode == ImageMode.MULTI_ROW:
            input_channel = self.image_seq_len
        elif self.image_mode == ImageMode.DEPTH:
            input_channel = self.image_seq_len
        elif self.image_mode == ImageMode.RGBD:
            input_channel = 4 * self.image_seq_len
        elif self.image_mode == ImageMode.RGB:
            input_channel = 3 * self.image_seq_len
        elif self.image_mode == ImageMode.GD:
            input_channel = 2 * self.image_seq_len
        else:
            raise NotImplementedError
        self.cnn = build_cnns_2d(input_channel, self.cnn_dims, self.kernel_sizes, self.strides)
        # self.mlp_relative_position = build_mlp(3 * self.pose_seq_len, self.dim_relative_position,
        #                                        activate_last_layer=False)


class SimpleCnnNcpActor(BaseModel):
    def __init__(self, agent_type, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)

        mlp_values_dims = model_params["mlp_values"]

        self.head = build_head(agent_type, action_space)
        self.cnn = build_cnns_2d(1, self.cnn_dims, self.kernel_sizes, self.strides)
        self.wirings = AutoNCP(48, 4)
        self.rnn = build_ncpltc(1283, wirings=self.wirings)

    def forward(self, x, hx=None):
        depth_image = x[0].float()
        relative_position = x[1].float()

        batch_size = depth_image.size(0)
        seq_len = depth_image.size(1)

        relative_position = relative_position.view(batch_size, seq_len, -1)

        depth_image = depth_image.view(batch_size * seq_len, 1, *depth_image.shape[2:])
        out1 = self.cnn(depth_image)
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

        self.mlp_action = build_mlp(self.dim_cnn_out_flatten + 2 * self.pose_seq_len,
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
        try:
            out = self.head(out)
        except Exception as e:
            logging.error("Exception:{}".format(e))
            logging.error("depth_image is nan:{}".format(torch.isnan(depth_image)))
            logging.error("-----depth_image has nan:{}-----".format(torch.isnan(depth_image).any()))
            logging.error("relative_position is nan:{}".format(torch.isnan(relative_position)))
            logging.error("-----relative_position has nan:{}-----".format(torch.isnan(relative_position).any()))
            logging.error("out1 is nan:{}".format(torch.isnan(out1)))
            logging.error("-----out1 has nan:{}-----".format(torch.isnan(out1).any()))
            logging.error("out is nan:{}".format(torch.isnan(out)))
            logging.error("-----out has nan:{}-----".format(torch.isnan(out).any()))
        return out


class SimpleCnnCritic(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, agent_type, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)
        mlp_values_dims = model_params["mlp_values"]

        self.mlp_value = build_mlp(self.dim_cnn_out_flatten + 2 * self.pose_seq_len + self.n_actions,
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
        # out2 = self.mlp_relative_position(relative_position)
        out = torch.cat((out1, relative_position, action), dim=1)

        return self.mlp_value(out)
