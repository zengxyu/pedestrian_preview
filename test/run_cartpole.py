import gym
import numpy as np

from configs.config import Config
from agents.action_space.d_action_space import DiscreteActionSpace
from agents.agents.dqn_agents.agent_ddqn import AgentDDQN
from agents.agents.dqn_agents.agent_dqn import AgentDQN
from agents.network.network_dqn import LinearDQNNetwork, DQNNetwork

action_space = DiscreteActionSpace(2)
config = Config().load_configs(yaml_path="../configs/default_dqn.yaml")

# discrete
network_cls = LinearDQNNetwork
# continuous
# network_cls = {"actor_network": None, "critic_network": None}

network = DQNNetwork(action_size=2)

agent = AgentDQN(config, network_cls, action_space)

env = gym.make('CartPole-v0')

for i in range(1000):
    done = False
    rewards = []
    obs = env.reset()
    while not done:
        action = agent.pick_action([obs])
        next_obs, reward, done, info = env.step(action)
        if done:
            reward = -1
        agent.step(state=[obs], action=action, reward=reward, next_state=[next_obs], done=done)
        rewards.append(reward)
        obs = next_obs.copy()
        agent.learn()

        # env.render()
    print("Episode:{}; Sum reward:{}".format(i, np.sum(rewards)))
env.close()
