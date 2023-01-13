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

from environment.human_npc_generator import generate_human_npc
from environment.robots.dynamic_obstacle import DynamicObstacleGroup
from environment.robots.human import Man
from utils.image_utility import dilate_image

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
from environment.scene_generator import load_environment_scene
from environment.nav_utilities.coordinates_converter import cvt_to_bu
from environment.path_manager import PathManager


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

        self.s_om_pose, self.s_bu_pose, self.g_om_pose, self.g_bu_pose = [None] * 4

        self.obstacle_collections: ObstacleCollections = ObstacleCollections(args)
        self.obstacle_ids = []
        # generate human agent, and other human npc

        self.start_goal_sampler, self.static_obs_sampler, self.dynamic_obs_sampler = None, None, None
        self.action_space_keys = None
        self.physical_steps = Counter()

        self.get_action_space_keys()

        self.last_distance = None
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
        self.initialize_turtlebot()
        self.randomize_human_npc()

        state = self.get_state()

        return state

    def get_action_space_keys(self):
        action_spaces_configs = read_yaml(self.args.action_space_config_folder, "action_space.yaml")
        action_class = self.action_space.__class__.__name__
        self.action_space_keys = list(action_spaces_configs[action_class].keys())

    def step(self, action):
        self.step_count += 1

        action = self.action_space.to_force(action=action)

        reach_goal, collision = self.iterate_steps(*action)

        state = self.get_state()
        reward = self.get_reward(reach_goal=reach_goal)

        over_max_step = self.step_count >= self.max_step

        # whether done
        done = collision or reach_goal or over_max_step

        # store information
        info_for_last = {"collision": collision, "a_success": reach_goal,
                         "over_max_step": over_max_step, "step_count": self.step_count.value}

        if done:
            print("success:{}; collision:{}; over_max_step:{}".format(reach_goal, collision, over_max_step))

        # plot stored information
        return state, reward, done, {}, info_for_last

    def get_reward(self, reach_goal):
        if self.last_distance is None:
            self.last_distance = compute_distance(self.g_bu_pose, self.s_bu_pose)

        reward = 0
        # compute distance from current to goal
        distance = compute_distance(self.g_bu_pose, self.robot.get_position())
        delta_distance = self.last_distance - distance
        self.last_distance = distance

        reward += delta_distance

        if reach_goal:
            reward += 100

        return reward

    def get_state(self):
        # compute depth image
        width, height, rgb_image, depth_image, seg_image = self.robot.sensor.get_obs()
        # compute relative position to goal
        relative_position = self.g_bu_pose - self.robot.get_position()
        # visit map
        return depth_image[np.newaxis, :, :], relative_position

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

    def randomize_human_npc(self):
        """
        add human npc
        """
        # randomize the start and end position for occupancy map
        obs_bu_starts, obs_bu_ends, obs_bu_paths = generate_human_npc(dynamic_obs_sampler=self.dynamic_obs_sampler,
                                                                      env_config=self.env_config,
                                                                      occ_map=self.occ_map,
                                                                      robot_start=self.s_bu_pose,
                                                                      robot_end=self.g_bu_pose)

        # create n dynamic obstacles, put them into the environment
        dynamic_obstacle_group = DynamicObstacleGroup(p=self.p,
                                                      args=self.args,
                                                      step_duration=self.physical_step_duration).create(
            start_positions=obs_bu_starts,
            end_positions=obs_bu_ends,
            paths=obs_bu_paths)

        self.obstacle_collections.add(dynamic_obstacle_group, dynamic=True)
        self.obstacle_ids.extend(self.obstacle_collections.get_obstacle_ids())

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

        self.obstacle_ids = obstacle_ids
        self.occ_map = maps["occ_map"]
        self.dilated_occ_map = maps["dilated_occ_map"]
        self.door_occ_map = maps["door_map"]
        # extract samplers
        self.start_goal_sampler, self.static_obs_sampler, self.dynamic_obs_sampler = samplers
        self.s_bu_pose = start
        self.g_bu_pose = end

        # initialize robot
        logging.debug("Create the environment, Done...")

    def initialize_turtlebot(self):
        self.robot = TurtleBot(self.p, self.client_id, self.physical_step_duration, self.robot_config, self.args,
                               self.s_bu_pose, compute_yaw(self.s_bu_pose, self.g_bu_pose))

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
        self.s_om_pose, self.s_bu_pose, self.g_om_pose, self.g_bu_pose = [None] * 4

        self.obstacle_collections.clear()
        self.obstacle_ids = []

    def logging_action(self, action):
        logging_str = ""
        for key, item in zip(self.action_space_keys, action):
            logging_str += "{}: {}; ".format(key[:-1], item)
        logging.warning(logging_str)
