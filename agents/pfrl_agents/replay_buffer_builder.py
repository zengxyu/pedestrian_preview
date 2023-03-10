#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 1/22/22 4:30 PM 
    @Description    :
        
===========================================
"""
import os.path

from pfrl import replay_buffers

from config import read_yaml


def get_replay_buffer_by_name(parser_args, name):
    config = read_yaml(config_dir=parser_args.agents_config_folder, config_name="replay_buffers.yaml")
    if name == "ReplayBuffer":
        return replay_buffers.ReplayBuffer(
            capacity=config[name]["capacity"],
            num_steps=config[name]["num_steps"],
        )
    elif name == "PrioritizedReplayBuffer":
        return replay_buffers.PrioritizedReplayBuffer(
            capacity=config[name]["capacity"],
            num_steps=config[name]["num_steps"],
            alpha=config[name]["alpha"],
            beta0=config[name]["beta0"],
            betasteps=config[name]["betasteps"],
            normalize_by_max=config[name]["normalize_by_max"],
        )
    elif name == "EpisodicReplayBuffer":
        return replay_buffers.EpisodicReplayBuffer(config[name]["capacity"])

    elif name == "PrioritizedEpisodicReplayBuffer":
        return replay_buffers.PrioritizedEpisodicReplayBuffer(capacity=config[name]["capacity"],
                                                              alpha=config[name]["alpha"],
                                                              beta0=config[name]["beta0"],
                                                              normalize_by_max=config[name]["normalize_by_max"],
                                                              return_sample_weights=config[name][
                                                                  "return_sample_weights"],
                                                              wait_priority_after_sampling=config[name][
                                                                  "wait_priority_after_sampling"]

                                                              )

    else:
        raise NotImplementedError("Cannot find replay buffer by name : {}".format(name))
