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
from agents.network.attention_spacial_temporal import STAttentionCritic, STAttentionActor
from agents.network.attention_temporal import AttentionTemporalCritic, AttentionTemporalActor
from agents.network.attention_spacial import AttentionSpacialActor, AttentionSpacialCritic
from agents.network.convolution_1d_network import CNNCriticNet, CNNActorNet
from agents.network.simple_cnn_network import SimpleCnnCritic, SimpleCnnActor, StochasticSampleCnnActor, StochasticSampleCnnCritic

actor_critic_model_mapping = {
    'CNN': (SimpleCnnActor, SimpleCnnCritic),
}
sac_actor_critic_model_mapping = {
    'CNN': (StochasticSampleCnnActor, StochasticSampleCnnCritic),
}

def get_actor_critic_class(agent, network_name):
    if agent == "sac":
        return sac_actor_critic_model_mapping[network_name]
    elif agent == "ddpg":
        return actor_critic_model_mapping[network_name]


def build_network(parser_args, action_space, input_kwargs):
    agent = get_agent_name(parser_args)
    network_name = get_network_config(parser_args)
    actor_class, critic_class = get_actor_critic_class(agent, network_name)

    if agent == "ddpg" or agent == "ddpg_recurrent":
        actor_network = actor_class(action_space, **input_kwargs)
        critic_network = critic_class(action_space, **input_kwargs)
        return actor_network, critic_network

    elif agent == "td3" or agent == "sac":
        actor_network = actor_class(action_space, **input_kwargs)
        critic_network1 = critic_class(action_space, **input_kwargs)
        critic_network2 = critic_class(action_space, **input_kwargs)
        return actor_network, critic_network1, critic_network2

    else:
        raise NotImplementedError("agent : {} no such an agent".format(agent))
