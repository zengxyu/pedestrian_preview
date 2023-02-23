#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning
    @Author  : Xiangyu Zeng
    @Date    : 2/6/22 6:06 PM 
    @Description    :
        
===========================================
"""
from agents.mapping import get_agent_name, get_network_config
from agents.network.multi_branch_cnn_network import MultiBranchCnnActor, MultiBranchCnnCritic
from agents.network.simple_cnn_network import SimpleCnnCritic, SimpleCnnActor
from agents.network.simple_lidar_mlp_network import SimpleLidarMlpActor, SimpleLidarMlpCritic
from agents.network.simple_ncp_network import SimpleCnnNcpActor, SimpleCnnNcpCritic
from agents.network.simple_mlp_network import SimpleMlpActor, SimpleMlpCritic

actor_critic_model_mapping = {
    'MLP': (SimpleMlpActor, SimpleMlpCritic),
    'LidarMLP': (SimpleLidarMlpActor, SimpleLidarMlpCritic),
    'CNN': (SimpleCnnActor, SimpleCnnCritic),
    'NCP': (SimpleCnnNcpActor, SimpleCnnNcpCritic),
    "MultiBranchCnn": (MultiBranchCnnActor, MultiBranchCnnCritic),
}


def get_actor_critic_class(network_name):
    return actor_critic_model_mapping[network_name]


def build_network(parser_args, action_space, input_kwargs):
    agent_type = get_agent_name(parser_args)
    network_name = get_network_config(parser_args)
    actor_class, critic_class = get_actor_critic_class(network_name)
    if agent_type == "ddpg" or agent_type == "ddpg_recurrent":
        actor_network = actor_class(agent_type, action_space, **input_kwargs)
        critic_network = critic_class(agent_type, action_space, **input_kwargs)
        return actor_network, critic_network

    elif agent_type == "td3" or agent_type == "sac":
        actor_network = actor_class(agent_type, action_space, **input_kwargs)
        critic_network1 = critic_class(agent_type, action_space, **input_kwargs)
        critic_network2 = critic_class(agent_type, action_space, **input_kwargs)
        return actor_network, critic_network1, critic_network2

    else:
        raise NotImplementedError("agent : {} no such an agent".format(agent_type))
