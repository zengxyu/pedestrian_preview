import gym
import numpy as np
import pfrl.action_value
from pfrl import explorers, replay_buffers, q_functions, experiments,agents
from torch import optim
from torch import nn
import torch.nn.functional as F

import logging

logging.basicConfig(level=logging.INFO)
start_epsilon = 0.3
end_epsilon = 0.05
final_exploration_steps = 10 ** 4

env = gym.make("CartPole-v0")

obs_space = env.observation_space
obs_size = obs_space.low.size
action_space = env.action_space
n_actions = action_space.n


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return pfrl.action_value.DiscreteActionValue(x)


q_func = Net(input_size=4, hidden_size=256, output_size=action_space.n)

opt = optim.Adam(q_func.parameters())

rbuf = replay_buffers.ReplayBuffer(capacity=5 * 10 ** 5)
explorer = explorers.LinearDecayEpsilonGreedy(
    start_epsilon,
    end_epsilon,
    final_exploration_steps,
    action_space.sample,
)
gpu = 0
gamma = 0.99
replay_start_size = 200
target_update_interval = 100
update_interval = 1
target_update_method = "hard"
soft_update_tau = 0.01

agent = agents.DoubleDQN(
    q_func,
    opt,
    rbuf,
    gpu=gpu,
    gamma=gamma,
    explorer=explorer,
    replay_start_size=replay_start_size,
    target_update_interval=target_update_interval,
    update_interval=update_interval,
    soft_update_tau=soft_update_tau,
)
steps = 10 ** 5
for i_episode in range(1000):
    done = False
    obs = env.reset()
    rewards = []
    while not done:
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            reward = -1
        agent.observe(obs, reward, done, reset=False)
        rewards.append(reward)
    print("i_episode:{}; Sum reward:{}".format(i_episode, np.sum(rewards)))
