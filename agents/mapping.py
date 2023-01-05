#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 5/25/22 4:41 PM 
    @Description    :
        
===========================================
"""
from utils.config_utility import read_yaml


class Env:
    Real = "real"
    Gazebo = "gazebo"
    Pybullet = "pybullet"
    IGibson = "igibson"


class InputsProcess:
    BaselineProcessing = "baseline_processing"
    TagdProcessing = "tagd_processing"


def get_agent_name(parser_args):
    agent_name = parser_args.running_config["agent"]
    return agent_name


def get_reward_config(args):
    candidate_rewards = read_reward_configs(args)
    reward_config_name = args.running_config["reward_config_name"]
    reward_config = candidate_rewards[reward_config_name]
    return reward_config


def get_input_config(args):
    candidate_inputs = read_input_configs(args)
    input_config_name = args.running_config["input_config_name"]
    input_config = candidate_inputs[input_config_name]
    return input_config


def get_network_config(args):
    input_config = get_input_config(args)
    return input_config["network"]


def read_reward_configs(args):
    """
    reach reward_configs
    :param args:
    :return:
    """
    reward_configs = read_yaml(config_dir=args.configs_folder, config_name="rewards.yaml")
    return reward_configs


def read_input_configs(args):
    """
    reach reward_configs
    :param args:
    :return:
    """
    input_configs = read_yaml(config_dir=args.configs_folder, config_name="inputs.yaml")
    return input_configs
