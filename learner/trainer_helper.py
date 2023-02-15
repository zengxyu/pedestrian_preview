#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/11/22 5:51 PM 
    @Description    :
        
===========================================
"""
import os
import pickle
from typing import List, Dict
import numpy as np
import torch


def add_graph(pf_learner, states):
    if not pf_learner.add_graph_to_writer:
        states = [torch.from_numpy(state[np.newaxis, ...]).to(pf_learner.device) for state in states]
        pf_learner.writer.add_graph(pf_learner.args.networks[0], (states,))
        pf_learner.add_graph_to_writer = True


def add_scalar(writer, phase, episode_info, i_episode):
    for key, item in episode_info.items():
        writer.add_scalar(str(phase) + "/" + str(key), item, i_episode)


def save_episodes_info(phase, episode_info_collector, i_episode, parser_args):
    save_path = os.path.join(parser_args.out_folder, phase + "_log.pkl")
    save_n = parser_args.running_config["save_result_n"]
    if i_episode % save_n == 0:
        file = open(save_path, 'wb')
        pickle.dump(episode_info_collector.episode_infos, file)


def get_items(infos_episode: List[Dict]):
    new_info_episodes = {}
    for info_step in infos_episode:
        for key in info_step.keys():
            if key not in new_info_episodes.keys():
                new_info_episodes[key] = [info_step[key]]
            else:
                new_info_episodes[key].append(info_step[key])
    return new_info_episodes
