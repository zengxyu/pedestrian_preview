import os

from configs.config import Config
from load_args import load_dqn_args, load_ac_args
from agents.agents.actor_critic_agents.ddpg import AgentDDPG
from agents.network.network_ddpg import PolicyNet, QNetwork

""" Deactivate all GPUs, explicitly activate below """
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gym

if __name__ == "__main__":
    env = gym.make("HalfCheetah-v2")
    network_cls = {"actor_network": PolicyNet, "critic_network": QNetwork}

    obs_space = env.observation_space
    action_space = env.action_space

    parser_config, config = load_ac_args()

    agent = AgentDDPG(config, network_cls, action_space)

    print("Observation space:", obs_space)
    print("Action space:", action_space)


