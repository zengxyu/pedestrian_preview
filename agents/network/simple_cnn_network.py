from agents.network.network_base import *
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_params = {
    'cnn': [32, 16, 8],
    'mlp': [32, 128],
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
        self.mlp_relative_position = build_mlp(2, self.mlp_dims, activate_last_layer=False)


class SimpleCnnActor(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)

        mlp_values_dims = model_params["mlp_values"]

        self.td3_end = nn.Sequential(nn.Tanh(), pfrl.policies.DeterministicHead())

        self.mlp_action = build_mlp(1202,
                                    mlp_values_dims + [self.n_actions],
                                    activate_last_layer=False,
                                    )

    def forward(self, x):
        depth_image = x[0].float()
        relative_position = x[1].float().to(device)

        batch_size = depth_image.size(0)
        out1 = self.cnn(depth_image)
        out1 = out1.reshape((batch_size, -1))
        # out2 = self.mlp_relative_position(relative_position)
        out2 = relative_position
        # out =
        out = torch.cat((out1, out2), dim=1)

        out = self.mlp_action(out)
        out = self.td3_end(out)
        return out
class StochasticSampleCnnActor(BaseModel):
    def __init__(self, actiondim, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(4802, 512)
        self.fc2 = nn.Linear(512, 256)
        self.action = nn.Linear(256, 4)
        self.flatten = nn.Flatten()
        self.min_log_std = -20
        self.max_log_std = 2
    def forward(self, x):
        depth_image = x[0].float()
        relative_position = x[1].float().to(device)

        c1 = F.relu(self.conv1(depth_image))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        flatten = F.relu(self.flatten(c3))
        f0 = torch.cat((flatten, relative_position),dim=1)
        f1 = F.relu(self.fc1(f0))
        f2 = F.relu(self.fc2(f1))
        action = self.action(f2)

        mean, log_scale = torch.chunk(action, 2, dim=1)
        log_scale = torch.clamp(log_scale, -20.0, 2.0)
        var = torch.exp(log_scale * 2)
        base_distribution = torch.distributions.Independent(
            torch.distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
        )
        return torch.distributions.transformed_distribution.TransformedDistribution(
            base_distribution, [torch.distributions.transforms.TanhTransform(cache_size=1)]
        )

class StochasticSampleCnnCritic(BaseModel):
    def __init__(self,actiondim, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(4804, 512)
        self.fc2 = nn.Linear(512, 256)
        self.Q = nn.Linear(256, 1)

        self.flatten = nn.Flatten()

    def forward(self, x):
        x,action = x
        depth_image = x[0].float()
        relative_position = x[1].float().to(device)

        c1 = F.relu(self.conv1(depth_image))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        flatten = F.relu(self.flatten(c3))
        f0 = torch.cat((flatten, relative_position, action),dim=1)
        f1 = F.relu(self.fc1(f0))
        f2 = F.relu(self.fc2(f1))
        q = self.Q(f2)

        return q
class SimpleCnnCritic(BaseModel):
    """
    Apply attention mechanism on lidar measurements
    """

    def __init__(self, action_space, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = len(action_space.low)
        mlp_values_dims = model_params["mlp_values"]

        self.mlp_value = build_mlp(1202 + self.n_actions,
                                   mlp_values_dims + [1],
                                   activate_last_layer=False,
                                   )

    def forward(self, x):
        x, action = x
        depth_image = x[0].float()
        relative_position = x[1].float().to(device)

        batch_size = depth_image.size(0)
        out1 = self.cnn(depth_image)
        out1 = out1.reshape((batch_size, -1))
        # out2 = self.mlp_relative_position(relative_position)
        out2 = relative_position
        out = torch.cat((out1, out2, action), dim=1)

        return self.mlp_value(out)
