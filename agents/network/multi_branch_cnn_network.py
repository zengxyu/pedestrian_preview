import torch

from agents.network.network_base import *
from agents.network.network_head import *
from environment.sensors.vision_sensor import ImageMode

from ncps.wirings import AutoNCP

from utils.convolution_width_utility import compute_conv_out_width

model_params = {
    'cnn': [32, 64, 32],
    'kernel_sizes': [3, 3, 3],
    'strides': [2, 2, 2],
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
        input_channel = self.image_seq_len
        self.cnn1 = build_cnns_2d(input_channel, self.cnn_dims, self.kernel_sizes, self.strides)
        self.cnn2 = build_cnns_2d(input_channel, self.cnn_dims, self.kernel_sizes, self.strides)
        self.cnn3 = build_cnns_2d(input_channel, self.cnn_dims, self.kernel_sizes, self.strides)
        self.cnn4 = build_cnns_2d(input_channel, self.cnn_dims, self.kernel_sizes, self.strides)

        # self.mlp_relative_position = build_mlp(3 * self.pose_seq_len, self.dim_relative_position,
        #                                        activate_last_layer=False)


class MultiBranchCnnActor(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, agent_type, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)

        mlp_values_dims = model_params["mlp_values"]

        self.head = build_head(agent_type, action_space)

        self.mlp_action = build_mlp(self.dim_cnn_out_flatten * 4 + 3 * self.pose_seq_len,
                                    mlp_values_dims + [self.n_actions * 2],
                                    activate_last_layer=False,
                                    )

    def forward(self, x):
        relative_position = x[1].float()
        depth_image = x[0].float()
        batch_size = depth_image.size(0)
        depth_image = depth_image.view((batch_size, self.image_seq_len, 4, depth_image.shape[-2], depth_image.shape[-1]))
        depth_image_forward = depth_image[:, :, 0, :, :]
        depth_image_right = depth_image[:, :, 1, :, :]
        depth_image_backward = depth_image[:, :, 2, :, :]
        depth_image_left = depth_image[:, :, 3, :, :]

        out_forward = self.cnn1(depth_image_forward)
        out_right = self.cnn2(depth_image_right)
        out_backward = self.cnn3(depth_image_backward)
        out_left = self.cnn4(depth_image_left)

        out_forward = out_forward.reshape((batch_size, -1))
        out_right = out_right.reshape((batch_size, -1))
        out_backward = out_backward.reshape((batch_size, -1))
        out_left = out_left.reshape((batch_size, -1))

        out = torch.cat((out_forward, out_right, out_backward, out_left, relative_position), dim=1)

        out = self.mlp_action(out)
        out = self.head(out)
        return out


class MultiBranchCnnCritic(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, agent_type, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)
        mlp_values_dims = model_params["mlp_values"]

        self.mlp_value = build_mlp(self.dim_cnn_out_flatten * 4 + 3 * self.pose_seq_len + self.n_actions,
                                   mlp_values_dims + [1],
                                   activate_last_layer=False,
                                   )

    def forward(self, x):
        x, action = x
        depth_image = x[0].float()
        relative_position = x[1].float()
        batch_size = depth_image.size(0)

        depth_image = depth_image.view((batch_size, self.image_seq_len, 4, depth_image.shape[-2], depth_image.shape[-1]))

        depth_image_forward = depth_image[:, :, 0, :, :]
        depth_image_right = depth_image[:, :, 1, :, :]
        depth_image_backward = depth_image[:, :, 2, :, :]
        depth_image_left = depth_image[:, :, 3, :, :]

        out_forward = self.cnn1(depth_image_forward)
        out_right = self.cnn2(depth_image_right)
        out_backward = self.cnn3(depth_image_backward)
        out_left = self.cnn4(depth_image_left)

        out_forward = out_forward.reshape((batch_size, -1))
        out_right = out_right.reshape((batch_size, -1))
        out_backward = out_backward.reshape((batch_size, -1))
        out_left = out_left.reshape((batch_size, -1))

        out = torch.cat((out_forward, out_right, out_backward, out_left, relative_position, action), dim=1)

        return self.mlp_value(out)
