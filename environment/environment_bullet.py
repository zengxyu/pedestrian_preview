#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 3/29/22 11:25 PM 
    @Description    :
        
===========================================
"""

import logging
import os
import sys

import cv2

from environment.robots.human import Man

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "traditional_planner", "a_star"))
print(sys.path)
import numpy as np
from agents.action_space.high_level_action_space import AbstractHighLevelActionSpace
from environment.base_pybullet_env import PybulletBaseEnv
from environment.nav_utilities.check_helper import check_collision
from environment.nav_utilities.counter import Counter
from environment.robots.obstacle_collections import ObstacleCollections
from environment.robots.turtlebot import TurtleBot

from utils.config_utility import read_yaml
from utils.math_helper import compute_yaw, compute_distance
from environment.gen_scene.scene_generator import load_environment_scene
from environment.nav_utilities.coordinates_converter import cvt_to_bu, cvt_to_om
from environment.path_manager import PathManager
from traditional_planner.a_star.astar import AStar
from torch.utils.tensorboard import SummaryWriter


class EnvironmentBullet(PybulletBaseEnv):
    def __init__(self, args, action_space):
        PybulletBaseEnv.__init__(self, args)
        self.args = args
        self.running_config = args.running_config
        self.robot_config = args.robot_config
        self.env_config = args.env_config
        self.world_config = args.world_config
        self.render = args.render

        self.grid_res = self.env_config["grid_res"]

        self.max_step = self.running_config["max_steps"]
        self.inference_every_duration = self.running_config["inference_per_duration"]
        self.action_space: AbstractHighLevelActionSpace = action_space

        self.path_manager = PathManager(self.args)

        self.robot: TurtleBot = None
        self.occ_map = None
        self.dilated_occ_map = None
        self.door_occ_map = None

        self.s_bu_pose, self.g_bu_pose = [None] * 2

        self.obstacle_collections: ObstacleCollections = ObstacleCollections(args)
        self.obstacle_ids = []
        # n_radius, n_from_start, n_to_end [unit: pixel]
        # generate human agent, and other human npc
        self.n_radius = int(self.env_config["unobstructed_radius"] / self.grid_res)
        self.n_from_start = int(self.env_config["distance_from_start"] / self.grid_res)
        self.n_to_end = int(self.env_config["distance_to_end"] / self.grid_res)
        self.n_dynamic_obstacle_num = self.env_config["pedestrian_dynamic_num"]
        self.n_static_obstacle_num = self.env_config["pedestrian_static_num"]

        self.n_kept_distance = int(self.env_config["kept_distance"] / self.grid_res)
        self.n_kept_distance_to_start = int(self.env_config["kept_distance_to_start"] / self.grid_res)
        self.dilation_size = self.env_config["dilation_size"]
        self.start_goal_sampler, self.static_obs_sampler, self.dynamic_obs_sampler = None, None, None
        self.action_space_keys = None
        self.physical_steps = Counter()

        self.get_action_space_keys()

        self.last_distance = None
        self.writer = SummaryWriter("runs/logs_reward")
        self.step_nums = 0
        """
        initialize environment
        initialize dynamic human npc
        initialize the human agent
        compute the relative position to the target
        
        
        """

    def render(self, mode="human"):
        width, height, rgb_image, depth_image, seg_image = self.robot.sensor.get_obs()
        return width, height, rgb_image, depth_image, seg_image

    def reset(self):
        self.episode_count += 1
        self.physical_steps = Counter()
        logging.info("\n---------------------------------------reset--------------------------------------------")
        logging.info("Episode : {}".format(self.episode_count))
        self.reset_simulation()
        self.clear_variables()

        # randomize environment
        self.randomize_env()

        state = self.get_state()
        # self.visualize_ground_destination()
        return state

    def visualize_ground_destination(self):
        thetas = np.linspace(0, np.pi * 2, 50)
        radius = 0.5
        points_x = np.cos(thetas) * radius + self.g_bu_pose[0]
        points_y = np.sin(thetas) * radius + self.g_bu_pose[1]
        z = np.zeros_like(thetas)
        points = np.array([points_x, points_y, z]).T
        points_next = np.roll(points, -1, axis=0)
        # froms = [[1, 1, 0], [-1, 1, 0], [-1, 1, 3], [1, 1, 3]]
        # tos = [[-1, 1, 0], [-1, 1, 3], [1, 1, 3], [1, 1, 0]]
        for f, t in zip(points, points_next):
            self.p.addUserDebugLine(
                lineFromXYZ=f,
                lineToXYZ=t,
                lineColorRGB=[0, 1, 0],
                lineWidth=2
            )

    def get_action_space_keys(self):
        action_spaces_configs = read_yaml(self.args.action_space_config_folder, "action_space.yaml")
        action_class = self.action_space.__class__.__name__
        self.action_space_keys = list(action_spaces_configs[action_class].keys())

    def step(self, action):
        self.step_count += 1
        self.step_nums += 1
        action = self.action_space.to_force(action=action)

        reach_goal, collision = self.iterate_steps(*action)

        state = self.get_state()
        reward, reward_info = self.get_reward(reach_goal=reach_goal, collision=collision,
                                              step_count=self.step_count.value)
        # print("reward=",reward)
        over_max_step = self.step_count >= self.max_step

        # whether done
        done = collision or reach_goal or over_max_step
        # done = reach_goal or over_max_step
        step_info = reward_info
        # store information
        episode_info = {"collision": collision, "a_success": reach_goal,
                        "over_max_step": over_max_step, "step_count": self.step_count.value}

        if done:
            print("success:{}; collision:{}; over_max_step:{}".format(reach_goal, collision, over_max_step))

        # plot stored information
        return state, reward, done, step_info, episode_info

    def get_reward(self, reach_goal, collision, step_count):
        if self.last_distance is None:
            self.last_distance = compute_distance(self.g_bu_pose, self.s_bu_pose)
        reward = 0
        collision_reward = 0
        reach_goal_reward = 0
        """================collision reward=================="""
        if collision:
            collision_reward = -100
            reward += collision_reward

        """================delta distance reward=================="""
        # compute distance from current to goal
        distance = compute_distance(self.g_bu_pose, self.robot.get_position())
        delta_distance_reward = (self.last_distance - distance) * 100
        self.last_distance = distance
        reward += delta_distance_reward

        """================step reward=================="""
        step_count_reward = - float(np.log(step_count) * 0.1) * 0
        reward += step_count_reward

        """================reach goal reward=================="""

        if reach_goal:
            reach_goal_reward = 100
            reward += reach_goal_reward

        reward_info = {"reward/reward_collision": collision_reward,
                       "reward/reward_delta_distance": delta_distance_reward,
                       "reward/reward_step_count": step_count_reward,
                       "reward/distance": distance,
                       "reward/reward_reach_goal": reach_goal_reward,
                       "reward/reward": reward
                       }
        return reward, reward_info

    def get_state(self):
        return self.get_state2()

    def get_state2(self):
        # compute depth image
        width, height, rgb_image, depth_image, seg_image = self.robot.sensor.get_obs()

        # compute relative position to goal
        relative_position = self.g_bu_pose - self.robot.get_position()

        relative_yaw = compute_yaw(self.g_bu_pose, self.robot.get_position()) - self.robot.get_yaw()

        relative_pose = np.array([relative_position[0], relative_position[1], relative_yaw])

        return depth_image[np.newaxis, :, :], relative_pose.flatten()

    def p_step_simulation(self):
        self.p.stepSimulation()
        self.physical_steps += 1

    def _check_collision(self):
        return check_collision(self.p, [self.robot.robot_id], self.obstacle_ids)

    def iterate_steps(self, planned_v, planned_w):
        iterate_count = 0
        reach_goal, collision = False, False
        # 0.4/0.05 = 8
        n_step = np.round(self.inference_every_duration / self.physical_step_duration)

        while iterate_count < n_step and not reach_goal and not collision:
            self.robot.small_step(planned_v, planned_w)
            self.obstacle_collections.step()
            self.p_step_simulation()

            collision = self._check_collision()
            reach_goal = compute_distance(self.robot.get_position(), self.g_bu_pose) < 0.5
            iterate_count += 1
        return reach_goal, collision

    # def randomize_dynamic_obstacles(self):
    #     if self.n_dynamic_obstacle_num == 0:
    #         return
    #
    #     logging.debug(
    #         "randomize dynamic obstacles... length of original path:{}".format(len(self.path_manager.original_path)))
    #
    #     # randomize the start and end position for occupancy map
    #     obs_bu_starts = []
    #     obs_bu_ends = []
    #     obs_bu_paths = []
    #     start_index = self.n_from_start
    #     end_index = len(self.om_path) - self.n_to_end
    #     equal_n = (end_index - start_index) / 4
    #
    #     seg_start_indexes = np.arange(start_index, end_index - equal_n, equal_n)
    #     seg_end_indexes = np.arange(start_index + equal_n, end_index, equal_n)
    #
    #     assert len(seg_start_indexes) == len(seg_end_indexes)
    #     used_indexes = []
    #     count = 0
    #     while count < 30 and len(obs_bu_starts) < self.n_dynamic_obstacle_num - 1:
    #         count += 1
    #         index = self.sample_segment_index(used_indexes, seg_start_indexes, seg_end_indexes)
    #         seg_start, seg_end = seg_start_indexes[index], seg_end_indexes[index]
    #         [obs_om_start, obs_om_end], sample_success = self.dynamic_obs_sampler(
    #             occupancy_map=dilate_image(self.occ_map, 2),
    #             door_map=self.door_occ_map,
    #             robot_om_path=self.om_path,
    #             margin=self.dilation_size + 1,
    #             radius=self.n_radius,
    #             start_index=seg_start,
    #             end_index=seg_end,
    #             kept_distance=self.n_kept_distance,
    #             kept_distance_to_start=self.n_kept_distance_to_start)
    #
    #         # if sample obstacle start position and end position failed, continue to next loop and resample
    #         if not sample_success:
    #             continue
    #
    #         # plan a global path for this (start, end) pair
    #         obs_om_path = AStar(dilate_image(self.occ_map, 2)).search_path(tuple(self.s_om_pose), tuple(self.g_om_pose))
    #
    #         if len(obs_om_path) == 0:
    #             continue
    #         logging.debug("There are now {} sampled obstacles".format(len(obs_bu_starts)))
    #         obs_bu_path = cvt_to_bu(obs_om_path, self.grid_res)
    #         obs_bu_start = cvt_to_bu(obs_om_start, self.grid_res)
    #         obs_bu_end = cvt_to_bu(obs_om_end, self.grid_res)
    #
    #         obs_bu_paths.append(obs_bu_path)
    #         obs_bu_starts.append(obs_bu_start)
    #         obs_bu_ends.append(obs_bu_end)
    #
    #         if len(obs_bu_starts) >= self.n_dynamic_obstacle_num - 1:
    #             break
    #
    #     # 添加障碍物， 该障碍物从机器人路径倒着走回
    #     self.add_one_backwards(obs_bu_starts, obs_bu_ends, obs_bu_paths)
    #
    #     if len(obs_bu_starts) == 0 and self.n_dynamic_obstacle_num > 0:
    #         return self.restart_environment()
    #
    #     # create n dynamic obstacles, put them into the environment
    #     dynamic_obstacle_group = DynamicObstacleGroup(self.p, self.args, self.physical_step_duration).create(
    #         obs_bu_starts,
    #         obs_bu_ends,
    #         obs_bu_paths)
    #
    #     self.obstacle_collections.add(dynamic_obstacle_group, dynamic=True)
    #     self.obstacle_ids.extend(self.obstacle_collections.get_obstacle_ids())

    def add_one_backwards(self, obs_bu_starts, obs_bu_ends, obs_bu_paths):
        len_original_path = len(self.path_manager.original_path)
        random_start_index = np.random.randint(int(0.5 * len_original_path), len_original_path - 2)
        # random_start_index = min(int(0.5 * len_original_path), len_original_path - 1)
        random_start = self.path_manager.original_path[random_start_index]
        path = self.path_manager.original_path[:random_start_index + 1, :]
        path_inverse = path[::-1]
        if len(path) > 15:
            obs_bu_starts.append(random_start)
            obs_bu_ends.append(self.s_bu_pose)
            obs_bu_paths.append(path_inverse)

    def randomize_env(self):
        """
        create office
        dilate occupancy map
        sample start point and end point
        plan global path using a* algorithm

        if the planned path is None, loop --> recreate the office
        :return: initialize global variables :
                self.obstacle_ids, self.occ_map, self.grid_res, self.dilated_occ_map,
                self.s_om_pose, self.s_bu_pose, self.g_om_pose, self.g_bu_pose,
                self.om_path, self.bu_path
        """
        # randomize building
        maps, samplers, obstacle_ids, start, end = load_environment_scene(p=self.p,
                                                                          env_config=self.env_config,
                                                                          world_config=self.world_config)

        # sample start pose and goal pose
        self.s_bu_pose = cvt_to_bu(start, self.grid_res)
        self.g_bu_pose = cvt_to_bu(end, self.grid_res)
        self.obstacle_ids = obstacle_ids
        self.occ_map = maps["occ_map"]
        self.dilated_occ_map = maps["dilated_occ_map"]
        self.door_occ_map = maps["door_map"]
        # extract samplers
        self.start_goal_sampler, self.static_obs_sampler, self.dynamic_obs_sampler = samplers

        # initialize robot
        logging.debug("Create the environment, Done...")

        # self.robot = self.initialize_human_agent(self.s_bu_pose)
        self.robot = self.initialize_turtlebot()

    def initialize_turtlebot(self):
        turtlebot = TurtleBot(self.p, self.client_id, self.physical_step_duration, self.robot_config, self.args,
                              self.s_bu_pose, compute_yaw(self.s_bu_pose, self.g_bu_pose))
        return turtlebot

    def initialize_human_agent(self, start):

        human = Man(self.client_id, partitioned=True, timestep=self.physical_step_duration,
                    translation_scaling=0.95 / 5)
        human.reset()
        human.resetGlobalTransformation(
            xyz=np.array([start[0], start[1], 0.94 * human.scaling]),
            rpy=np.array([0, 0, 0]),
            gait_phase_value=0
        )
        return human

    def clear_variables(self):
        self.step_count = Counter()
        self.robot = None
        self.occ_map = None
        self.dilated_occ_map = None
        self.door_occ_map = None
        self.s_bu_pose, self.g_bu_pose = [None] * 2

        self.obstacle_collections.clear()
        self.last_distance = None
        self.obstacle_ids = []

    def logging_action(self, action):
        logging_str = ""
        for key, item in zip(self.action_space_keys, action):
            logging_str += "{}: {}; ".format(key[:-1], item)
        logging.warning(logging_str)
