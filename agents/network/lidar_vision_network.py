import torch

from agents.network.network_base import *
from agents.network.network_head import *
from environment.sensors.vision_sensor import ImageMode

from ncps.wirings import AutoNCP

from utils.convolution_width_utility import compute_conv_out_width

model_params = {
    'cnn': [32, 64, 32],
    'mlp_after_cnn': [512],
    'kernel_sizes': [3, 3, 3],
    'strides': [2, 2, 2],
    'mlp_lidar': [512, 128],
    'mlp_values': [256, 128, 64],
    'mlp_relative_position': [36, 64],
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
        self.dim_mlp_lidar = model_params["mlp_lidar"]
        self.dim_mlp_after_cnn = model_params["mlp_after_cnn"]
        h_out = compute_conv_out_width(kwargs["image_h"], k=3, s=2, p=1, iter=len(self.cnn_dims))
        w_out = compute_conv_out_width(kwargs["image_w"], k=3, s=2, p=1, iter=len(self.cnn_dims))
        self.dim_cnn_out_flatten = h_out * w_out * self.cnn_dims[-1]
        input_channel = kwargs["in_channel"]
        input_lidar_channel = kwargs["in_lidar_channel"]
        self.cnn = build_cnns_2d(input_channel, self.cnn_dims, self.kernel_sizes, self.strides)
        self.mlp_after_cnn = build_mlp(self.dim_cnn_out_flatten, self.dim_mlp_after_cnn, activate_last_layer=True)
        self.mlp_lidar = build_mlp(input_lidar_channel, self.dim_mlp_lidar, activate_last_layer=True)


class LidarVisionActor(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, agent_type, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)

        mlp_values_dims = model_params["mlp_values"]

        self.head = build_head(agent_type, action_space)

        self.mlp_action = build_mlp(self.dim_mlp_after_cnn[-1] + self.dim_mlp_lidar[-1] + 2 * self.pose_seq_len,
                                    mlp_values_dims + [self.n_actions * 2],
                                    activate_last_layer=False,
                                    )

    def forward(self, x):
        depth_image = x[0].float()
        lidar = x[1].float()
        relative_position = x[2].float()

        batch_size = depth_image.size(0)
        out_depth = self.cnn(depth_image)
        out_depth = out_depth.reshape((batch_size, -1))
        out_depth = self.mlp_after_cnn(out_depth)

        out_lidar = self.mlp_lidar(lidar)
        out = torch.cat((out_depth, out_lidar, relative_position), dim=1)

        out = self.mlp_action(out)
        try:
            out = self.head(out)
        except Exception as e:
            logging.error("Exception:{}".format(e))
            logging.error("depth_image is nan:{}".format(torch.isnan(depth_image)))
            logging.error("-----depth_image has nan:{}-----".format(torch.isnan(depth_image).any()))
            logging.error("relative_position is nan:{}".format(torch.isnan(relative_position)))
            logging.error("-----relative_position has nan:{}-----".format(torch.isnan(relative_position).any()))
            logging.error("out is nan:{}".format(torch.isnan(out)))
            logging.error("-----out has nan:{}-----".format(torch.isnan(out).any()))
        return out


class LidarVisionCritic(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, agent_type, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)
        mlp_values_dims = model_params["mlp_values"]

        self.mlp_value = build_mlp(self.dim_mlp_after_cnn[-1] + self.dim_mlp_lidar[-1] + 2 * self.pose_seq_len + self.n_actions,
                                   mlp_values_dims + [1],
                                   activate_last_layer=False,
                                   )

    def forward(self, x):
        x, action = x

        depth_image = x[0].float()
        lidar = x[1].float()
        relative_position = x[2].float()

        batch_size = depth_image.size(0)

        out_depth = self.cnn(depth_image)
        out_depth = out_depth.reshape((batch_size, -1))
        out_depth = self.mlp_after_cnn(out_depth)

        out_lidar = self.mlp_lidar(lidar)
        out = torch.cat((out_depth, out_lidar, relative_position, action), dim=1)

        return self.mlp_value(out)
