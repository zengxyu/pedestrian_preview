import torch
import pfrl
from torch import distributions, nn
from pfrl.nn.lmbda import Lambda


def squashed_diagonal_gaussian_head(x):
    mean, log_scale = torch.chunk(x, 2, dim=1)
    log_scale = torch.clamp(log_scale, -20.0, 2.0)
    var = torch.exp(log_scale * 2)
    base_distribution = distributions.Independent(
        distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
    )
    # cache_size=1 is required for numerical stability
    return distributions.transformed_distribution.TransformedDistribution(
        base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
    )


def build_sac_head():
    return Lambda(squashed_diagonal_gaussian_head)


def build_td3_head():
    return nn.Sequential(nn.Tanh(), pfrl.policies.DeterministicHead())


def build_head(agent_type, action_space):
    if agent_type == "ddpg" or agent_type == "td3":
        return build_td3_head()
    elif agent_type == "sac":
        action_dim = len(action_space.low)
        # linear
        return build_sac_head()
    else:
        raise NotImplementedError
