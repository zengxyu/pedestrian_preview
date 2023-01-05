#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 6/8/22 5:20 PM 
    @Description    :
        
===========================================
"""
import json
import logging
import os.path
import pickle

import numpy as np

from visualize_utilities.bar_grah import bar_graph


class EvaluationHelper:
    def __init__(self, args):
        self._args = args
        self.env_config = args.env_config
        self.running_config = args.running_config
        self._physical_step_duration = args.env_config["step_duration"]

        self._env_types = self.get_env_types()
        self._pedestrian_dynamic_num = self.env_config["pedestrian_dynamic_num"]
        self._pedestrian_static_num = self.env_config["pedestrian_static_num"]
        self._pedestrian_speed_range = self.env_config["pedestrian_speed_range"]

        self._folder = os.path.join(self._args.out_folder,
                                    "..",
                                    "evaluation",
                                    self._in_folder_name(),
                                    self._env_types,
                                    self._pedestrian_num_folder_name(),
                                    self._pedestrian_speed_folder_name())

        self.history_robot_positions = []
        self.history_robot_yaws = []
        self.history_robot_velocities = []
        # [episode, step, pedestrian, positions]
        self.history_pedestrians_positions = []
        self.history_success = []
        self.history_collision = []
        self.history_over_max_step = []
        self.history_step_simulation_counts = []
        self.history_actions = []

        self.motion_history_waypoint_spacings = []
        self.motion_history_waypoint_distances = []
        self.motion_history_planned_speeds = []
        self.motion_history_real_speeds = []
        self.history_occupancy_maps = []

    def refresh_evaluation_episodes(self):
        self.history_robot_positions.append([])
        self.history_robot_yaws.append([])
        self.history_robot_velocities.append([])
        self.history_step_simulation_counts.append(0)
        self.history_pedestrians_positions.append([])
        self.history_actions.append([])

        self.motion_history_waypoint_spacings.append([])
        self.motion_history_waypoint_distances.append([])
        self.motion_history_planned_speeds.append([])
        self.motion_history_real_speeds.append([])

    def get_env_types(self):
        env_probs = self.env_config["env_probs"] / np.sum(self.env_config["env_probs"])
        env_types = self.env_config["env_types"]
        env_types = np.array(env_types)[np.argwhere(env_probs != 0).squeeze()]
        env_probs = np.array(env_probs)[np.argwhere(env_probs != 0).squeeze()]
        if isinstance(env_types, str):
            return env_types
        else:
            return "hybrid"

    def _in_folder_name(self):
        "deform名字"
        if self._args.in_folder:
            in_folder_name = os.path.basename(self._args.in_folder)
        else:
            in_folder_name = self._args.policy
        return in_folder_name

    def _pedestrian_num_folder_name(self):
        pedestrian_num_str = "dynamic_" + str(self._pedestrian_dynamic_num) + "_static_" + str(
            self._pedestrian_static_num)
        return pedestrian_num_str

    def _pedestrian_speed_folder_name(self):
        pedestrian_speed_range_str = "speed_" + str(self._pedestrian_speed_range[1])
        return pedestrian_speed_range_str

    def _compute_successful_trajectory_distances(self):
        """
        计算距离
        :return:
        """
        total_distances = []
        for success, episode_positions in zip(self.history_success, self.history_robot_positions):
            if success:
                start_positions = np.array(episode_positions[:-1])
                end_positions = np.array(episode_positions[1:])
                distances = np.linalg.norm(end_positions - start_positions, axis=1)
                total_distance = np.sum(distances)
                total_distances.append(total_distance)
        return total_distances

    def _compute_travels_time(self):
        """
        compute travel time
        :return:
        """
        travel_time = np.array(self.history_step_simulation_counts) * self._physical_step_duration
        travel_time = travel_time.tolist()
        return travel_time

    def _compute_smoothness(self):
        return

    def _compute_min_distance(self):
        """
        compute the comfortness from the successful episodes
        :return:
        """
        min_distance_of_episodes = []
        for episode_robot_positions, episode_pedestrians_positions in zip(self.history_robot_positions,
                                                                          self.history_pedestrians_positions):
            episode_pedestrians_positions = np.array(episode_pedestrians_positions)
            episode_robot_positions = np.array(episode_robot_positions)

            min_distances_of_pedestrians = []
            # 行人的数量
            for i in range(episode_pedestrians_positions.shape[1]):
                episode_pedestrian_positions = episode_pedestrians_positions[:, i]
                distances = np.linalg.norm(episode_pedestrian_positions - episode_robot_positions, axis=1)
                min_distance = min(distances)
                min_distances_of_pedestrians.append(min_distance)
            # 几个行人中，谁离机器人最近
            min_distance_per_episode = min(min_distances_of_pedestrians)
            min_distance_of_episodes.append(min_distance_per_episode)

        return min_distance_of_episodes

    def save_metrics(self):
        """
        save metrics
        # 是两层还是一层 { 0, 1}
        :return:
        """
        result_config = {}
        # running config配置，和模型种类有关，evaluation的时候不改变
        result_config["algorithm"] = self.running_config["algorithm"]
        result_config["folder_name"] = self._in_folder_name()
        result_config["network"] = self.running_config["network"]

        # env config配置，evaluation的时候会调节这些参数来看改变
        result_config["pedestrian_dynamic_num"] = self._pedestrian_dynamic_num
        result_config["pedestrian_static_num"] = self._pedestrian_static_num
        result_config["env_types"] = self._env_types
        result_config["pedestrian_max_speed_range"] = self._pedestrian_speed_range[1]

        # 保存所有所有Episodes的travel time
        travels_time = self._compute_travels_time()
        result_config["travels_time_mean"] = np.mean(travels_time)
        result_config["travels_time_std"] = np.std(travels_time)

        # 保存所有episodes的路径长度
        distances = self._compute_successful_trajectory_distances()
        result_config["path_length_mean"] = np.mean(distances)
        result_config["path_length_std"] = np.std(distances)

        result_config["success_rate"] = np.mean(self.history_success)
        result_config["collision_rate"] = np.mean(self.history_collision)
        result_config["over_max_step_rate"] = np.mean(self.history_over_max_step)

        result_config["min_distance_mean"] = np.mean(self._compute_min_distance())
        result_config["min_distance_std"] = np.std(self._compute_min_distance())

        history_config = {}
        history_config["history_robot_positions"] = self.history_robot_positions
        history_config["history_robot_yaws"] = self.history_robot_yaws
        history_config["history_robot_velocities"] = self.history_robot_velocities
        history_config["history_pedestrians_positions"] = self.history_pedestrians_positions
        history_config["history_success"] = self.history_success
        history_config["history_collision"] = self.history_collision
        history_config["history_over_max_step"] = self.history_over_max_step
        history_config["history_step_simulation_counts"] = self.history_step_simulation_counts
        history_config["history_actions"] = self.history_actions
        history_config["history_occupancy_maps"] = self.history_occupancy_maps
        self._save_json(result_config, "metrics.json")
        self._save_pickle(history_config, "history.json")

        return result_config

    def _save_json(self, result, file_name):
        "write to local"
        if not os.path.exists(self._folder):
            os.makedirs(self._folder)
        file_path = os.path.join(self._folder, file_name)
        with open(file_path, "w") as f:
            json.dump(result, f)
        logging.warning("Write {} to local : {}".format(file_name, file_path))
        f.close()

    def _save_pickle(self, result, file_name):
        "write to local"
        if not os.path.exists(self._folder):
            os.makedirs(self._folder)
        file_path = os.path.join(self._folder, file_name)
        with open(file_path, "wb") as f:
            pickle.dump(result, f)
        logging.warning("Write {} to local : {}".format(file_name, file_path))
        f.close()
