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
import time
from collections import deque

import cv2

from environment.human_npc_generator import generate_human_npc
from environment.robots.dynamic_obstacle import DynamicObstacleGroup
from environment.robots.human import Man

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "traditional_planner", "a_star"))
print(sys.path)
import numpy as np
from agents.action_space.high_level_action_space import AbstractHighLevelActionSpace
from environment.base_pybullet_env import PybulletBaseEnv
from environment.nav_utilities.check_helper import check_collision, CollisionType
from environment.nav_utilities.counter import Counter
from environment.robots.obstacle_collections import ObstacleCollections
from environment.robots.turtlebot import TurtleBot

from utils.config_utility import read_yaml
from utils.math_helper import compute_yaw, compute_distance, compute_ManhattanDistance
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

        self.wall_obstacle_ids = []
        self.pedestrian_obstacle_ids = []
        # n_radius, n_from_start, n_to_end [unit: pixel]
        # generate human agent, and other human npc
        self.start_goal_sampler, self.static_obs_sampler, self.dynamic_obs_sampler = None, None, None
        self.action_space_keys = None
        self.physical_steps = Counter()

        self.get_action_space_keys()

        self.last_distance = None
        self.writer = SummaryWriter("runs/logs_reward")
        self.step_nums = 0

        self.depth_images = None
        self.relative_poses = None
        self.max_len = self.args.input_config[self.running_config['input_config_name']]["seq_len"]

        self.evaluate_crowd = True
        self.robots = None
        self.init_coordinates = [[5, 1]]
        self.ma_depth_images = None
        self.ma_relative_poses = []
        self.robots_list_id = []
        self.ma_depth_images_deque = None

        """
        initialize environment
        initialize dynamic human npc
        initialize the human agent
        compute the relative position to the target
        
        
        """

    def render(self, mode="human"):
        # width, height, rgb_image, depth_image, seg_image = self.robot.sensor.get_obs()
        # return width, height, rgb_image, depth_image, seg_image
        return

    def reset(self):
        self.episode_count += 1
        self.physical_steps = Counter()
        logging.info("\n---------------------------------------reset--------------------------------------------")
        logging.info("Episode : {}".format(self.episode_count))
        self.reset_simulation()
        self.clear_variables()

        # randomize environment
        self.randomize_env()
        self.randomize_human_npc()
        state = self.get_state()
        self.visualize_ground_destination()
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
        done = (collision == CollisionType.CollisionWithWall) or reach_goal or over_max_step
        # done = reach_goal or over_max_step
        step_info = reward_info
        # store information
        episode_info = {"collision": collision, "a_success": reach_goal,
                        "over_max_step": over_max_step, "step_count": self.step_count.value}

        if done:
            print("success:{}; collision:{}; over_max_step:{}".format(reach_goal, collision, over_max_step))

        if done and not self.args.train:
            self.add_episode_end_prompt(episode_info)

        # plot stored information
        return state, reward, done, step_info, episode_info

    def evaluate_step(self, actions):
        reach_goals = []
        for i, rt in enumerate(self.robots):
            planned_v, planned_w = actions[i]
            rt.small_step(planned_v, planned_w)
            reach_goals.append(compute_distance(rt.get_position(), self.g_bu_pose) < 0.5)
        self.obstacle_collections.step()
        self.p_step_simulation()
        stats = self.get_state()

        return stats, reach_goals

    def get_reward(self, reach_goal, collision, step_count):
        if self.last_distance is None:
            self.last_distance = compute_ManhattanDistance(self.g_bu_pose, self.s_bu_pose)
        reward = 0
        collision_reward = 0
        reach_goal_reward = 0
        """================collision reward=================="""
        if collision == CollisionType.CollisionWithWall:
            collision_reward = -200
            reward += collision_reward

        """================delta distance reward=================="""
        # compute distance from current to goal
        distance = compute_ManhattanDistance(self.g_bu_pose, self.robot.get_position())
        delta_distance_reward = (self.last_distance - distance) * 50
        self.last_distance = distance
        reward += delta_distance_reward

        """================step reward=================="""
        step_count_reward = - float(np.log(step_count) * 0.1)
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
        return self.get_state3()

    def get_state2(self):
        # compute depth image
        width, height, rgb_image, depth_image, seg_image = self.robot.sensor.get_obs()

        # compute relative position to goal
        relative_position = self.g_bu_pose - self.robot.get_position()

        relative_yaw = compute_yaw(self.g_bu_pose, self.robot.get_position()) - self.robot.get_yaw()

        relative_pose = np.array([relative_position[0], relative_position[1], relative_yaw])

        depth_image = cv2.resize(depth_image, (int(depth_image.shape[0] / 2), int(depth_image.shape[1] / 2)))

        if len(self.depth_images) == 0:
            for i in range(self.max_len - 1):
                temp = np.zeros_like(depth_image)
                self.depth_images.append(temp)
                temp2 = np.zeros_like(relative_pose)
                self.relative_poses.append(temp2)

        self.depth_images.append(depth_image)
        self.relative_poses.append(relative_pose)

        # return depth_image[np.newaxis, :, :], relative_pose.flatten()
        return np.array(self.depth_images), np.array(self.relative_poses).flatten()

    def get_state3(self):
        # compute depth image
        if self.evaluate_crowd:
            ma_depth_images = []
            res = []
            for i, rt in enumerate(self.robots):
                width, height, rgb_image, depth_image, seg_image = rt.sensor.get_obs()
                relative_position = self.g_bu_pose - rt.get_position()
                depth_image = cv2.resize(depth_image, (int(depth_image.shape[1] / 2), int(depth_image.shape[0] / 2)))

                if len(self.ma_depth_images_deque[i]) == 0:
                    for j in range(self.max_len - 1):
                        temp = np.zeros_like(depth_image)
                        self.ma_depth_images_deque[i].append(temp)

                self.ma_depth_images_deque[i].append(depth_image)
                ls = np.array(self.ma_depth_images_deque[i])
                ma_depth_images.append(ls)
                self.ma_relative_poses.append(relative_position)

            for i in range(len(self.robots)):
                res.append([ma_depth_images[i], self.ma_relative_poses[i]])
            return res
        else:
            width, height, rgb_image, depth_image, seg_image = self.robot.sensor.get_obs()
            # compute relative position to goal
            relative_position = self.g_bu_pose - self.robot.get_position()
            # relative_yaw = compute_yaw(self.g_bu_pose, self.robot.get_position()) - self.robot.get_yaw()
            # relative_pose = np.array([relative_position[0], relative_position[1]])

            depth_image = cv2.resize(depth_image, (int(depth_image.shape[1] / 2), int(depth_image.shape[0] / 2)))

            if len(self.depth_images) == 0:
                for i in range(self.max_len - 1):
                    temp = np.zeros_like(depth_image)
                    self.depth_images.append(temp)
            self.depth_images.append(depth_image)
            # return depth_image[np.newaxis, :, :], relative_pose.flatten()
            return np.array(self.depth_images), relative_position

    def p_step_simulation(self):
        self.p.stepSimulation()
        self.physical_steps += 1

    def _check_collision(self):
        if check_collision(self.p, [self.robot.robot_id], self.wall_obstacle_ids):
            return CollisionType.CollisionWithWall
        elif check_collision(self.p, [self.robot.robot_id], self.pedestrian_obstacle_ids):
            return CollisionType.CollisionWithPedestrian
        elif check_collision(self.p, self.robots_list_id, self.robots_list_id):
            return CollisionType.CollisionWithAgent
        else:
            return False

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

    def add_episode_end_prompt(self, info):
        display_duration = 0.5
        if info["a_success"]:
            self.p.addUserDebugText("Success!", textPosition=[1.5, 1.5, 1.], textColorRGB=[0, 1, 0], textSize=5)
            time.sleep(display_duration)
        elif info["collision"]:
            self.p.addUserDebugText("collision!", textPosition=[1.5, 1.5, 1.], textColorRGB=[1, 0, 0], textSize=5)
            time.sleep(display_duration)
        else:
            self.p.addUserDebugText("Over the max step!", textPosition=[1.5, 1.5, 1.], textColorRGB=[0, 0, 1],
                                    textSize=5)
            time.sleep(display_duration)

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
        self.pedestrian_obstacle_ids = self.obstacle_collections.get_obstacle_ids()

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
        self.wall_obstacle_ids = obstacle_ids
        self.occ_map = maps["occ_map"]
        self.dilated_occ_map = maps["dilated_occ_map"]
        self.door_occ_map = maps["door_map"]
        # extract samplers
        self.start_goal_sampler, self.static_obs_sampler, self.dynamic_obs_sampler = samplers

        # initialize robot
        logging.debug("Create the environment, Done...")

        # self.robot = self.initialize_human_agent(self.s_bu_pose)
        if self.evaluate_crowd:
            self.robots = self.init_robots(self.init_coordinates)
        else:
            self.robot = self.initialize_turtlebot()

    def init_robots(self, init_coordinates):
        agents = list()
        for init_coordinate in init_coordinates:
            turtlebot = TurtleBot(self.p, self.client_id, self.physical_step_duration, self.robot_config, self.args,
                                  init_coordinate, compute_yaw(self.s_bu_pose, self.g_bu_pose))
            self.robots_list_id.append(turtlebot.robot_id)
            agents.append(turtlebot)
        return agents

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
        self.wall_obstacle_ids = []
        self.depth_images = deque(maxlen=self.max_len)
        self.relative_poses = deque(maxlen=self.max_len)

        self.ma_depth_images_deque = [deque(maxlen=self.max_len) for i in range(len(self.init_coordinates))]
        self.ma_depth_images = []
        self.ma_relative_poses = []
        self.robots = None
        self.robots_list_id = []

    def logging_action(self, action):
        logging_str = ""
        for key, item in zip(self.action_space_keys, action):
            logging_str += "{}: {}; ".format(key[:-1], item)
        logging.warning(logging_str)
