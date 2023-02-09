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

from environment.gen_scene.scene_loader import load_scene
from environment.human_npc_generator import generate_human_npc
from environment.robots.npc import DynamicObstacleGroup
from environment.robots.robot_roles import RobotRoles
from environment.robots.robot_types import RobotTypes, init_robot
from environment.sensors.vision_sensor import ImageMode

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "traditional_planner", "a_star"))
print(sys.path)
import numpy as np
from agents.action_space.action_space import AbstractActionSpace
from environment.base_pybullet_env import PybulletBaseEnv
from environment.nav_utilities.check_helper import check_collision, CollisionType
from environment.nav_utilities.counter import Counter
from environment.robots.obstacle_collections import ObstacleCollections
from utils.math_helper import compute_yaw, compute_distance, compute_manhattan_distance
from environment.gen_scene.scene_generator import load_environment_scene
from environment.path_manager import PathManager
from traditional_planner.a_star.astar import AStar


class EnvironmentBullet(PybulletBaseEnv):
    def __init__(self, args, action_space):
        PybulletBaseEnv.__init__(self, args)
        self.args = args
        self.running_config = args.running_config
        self.env_config = args.env_config
        self.worlds_config = args.worlds_config
        self.robot_config = args.robots_config[self.running_config["robot_name"]]
        self.sensor_config = args.sensors_config[self.running_config["sensor_name"]]
        self.input_config = args.inputs_config[args.running_config["input_config_name"]]
        self.robot_name = self.running_config["robot_name"]

        self.render = args.render

        self.grid_res = self.env_config["grid_res"]

        self.max_step = self.running_config["max_steps"]
        self.inference_every_duration = self.running_config["inference_per_duration"]
        self.action_space: AbstractActionSpace = action_space

        self.path_manager = PathManager(self.args)

        self.occ_map = None
        self.dilated_occ_map = None
        self.door_occ_map = None

        self.bu_starts, self.bu_goals = [None] * 2
        self.npc_goals = []
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

        self.image_seq_len = self.args.inputs_config[self.running_config['input_config_name']]["image_seq_len"]
        self.pose_seq_len = self.args.inputs_config[self.running_config['input_config_name']]["pose_seq_len"]

        self.robots = None
        self.num_agents = self.args.env_config["num_agents"]
        self.ma_relative_poses_deque = []
        self.ma_images_deque = None
        self.robot_ids = []
        self.state = args.state
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

        if self.args.load_map_from is not None and self.args.load_map_from != "":
            self.load_env()
        else:
            # randomize environment
            self.randomize_env()
            self.randomize_human_npc()

        # # randomize environment
        # self.randomize_env()
        # self.randomize_human_npc()
        state = self.get_state()
        # self.visualize_goals(self.bu_goals, self.robots)

        if not self.args.train:
            self.visualize_goals(self.bu_goals, self.robots)
            # self.visualize_goals(self.npc_goals, self.obstacle_collections.get_obstacle_ids())
        return state

    def visualize_goals(self, bu_goals, robots):
        thetas = np.linspace(0, np.pi * 2, 50)
        radius = 0.2
        for i in range(len(bu_goals)):
            robot = robots[i]
            points_x = np.cos(thetas) * radius + bu_goals[i][0]
            points_y = np.sin(thetas) * radius + bu_goals[i][1]
            z = np.zeros_like(thetas)
            points = np.array([points_x, points_y, z]).T
            points_next = np.roll(points, -1, axis=0)
            # froms = [[1, 1, 0], [-1, 1, 0], [-1, 1, 3], [1, 1, 3]]
            # tos = [[-1, 1, 0], [-1, 1, 3], [1, 1, 3], [1, 1, 0]]
            for f, t in zip(points, points_next):
                self.p.addUserDebugLine(
                    lineFromXYZ=f,
                    lineToXYZ=t,
                    lineColorRGB=robot.color,
                    lineWidth=2
                )

    def get_action_space_keys(self):
        action_class = self.action_space.__class__.__name__
        self.action_space_keys = list(self.args.action_spaces_config[action_class].keys())

    def step(self, actions):
        self.step_count += 1
        # print("actions:{}".format(actions))
        reach_goal, collision = self.iterate_steps(actions)
        # print("v:{};w:{}".format(*self.robots[0].get_v_w()))
        state = self.get_state()
        reward, reward_info = self.get_reward(reach_goal=reach_goal, collision=collision)
        over_max_step = self.step_count >= self.max_step

        # whether done
        if self.args.train:
            done = (collision == CollisionType.CollisionWithWall) or reach_goal or over_max_step
        else:
            done = reach_goal or over_max_step
        # done = reach_goal or over_max_step
        step_info = reward_info
        # store information
        episode_info = {"collision": collision == CollisionType.CollisionWithWall, "a_success": reach_goal,
                        "over_max_step": over_max_step, "step_count": self.step_count.value}

        if done:
            print("success:{}; collision:{}; over_max_step:{}".format(reach_goal, collision, over_max_step))

        if done and not self.args.train:
            self.add_episode_end_prompt(episode_info)

        # plot stored information
        return state, reward, done, step_info, episode_info

    def get_reward(self, reach_goal, collision):
        if self.last_distance is None:
            self.last_distance = compute_distance(self.bu_goals[0], self.bu_starts)
        reward = 0
        collision_reward = 0
        reach_goal_reward = 0
        """================collision reward=================="""
        if collision == CollisionType.CollisionWithWall:
            collision_reward = -200
            reward += collision_reward

        """================delta distance reward=================="""
        # compute distance from current to goal
        distance = compute_distance(self.bu_goals[0], self.robots[0].get_position())
        delta_distance_reward = (self.last_distance - distance) * 30
        self.last_distance = distance
        reward += delta_distance_reward

        """================reach goal reward=================="""

        if reach_goal:
            reach_goal_reward = 200
            reward += reach_goal_reward

        reward_info = {"reward/reward_collision": collision_reward,
                       "reward/reward_delta_distance": delta_distance_reward,
                       "reward/distance": distance,
                       "reward/reward_reach_goal": reach_goal_reward,
                       "reward/reward": reward
                       }
        return reward, reward_info

    def get_state(self):
        return self.get_state3()

    def get_state3(self):
        # compute depth image
        ma_images = []
        ma_relative_poses = []
        res = []
        w = 0
        h = 0
        for i, rt in enumerate(self.robots):
            width, height, rgbd_image, depth_image, seg_image = rt.sensor.get_obs()
            relative_position = self.bu_goals[i] - rt.get_position()
            relative_yaw = compute_yaw(self.bu_goals[i], rt.get_position()) - rt.get_yaw()
            relative_pose = np.array([relative_position[0], relative_position[1], relative_yaw])

            w = int(depth_image.shape[1] / 4)
            h = int(depth_image.shape[0] / 4)
            if self.input_config["image_mode"] == ImageMode.DEPTH:
                image = cv2.resize(depth_image, (w, h))
            elif self.input_config["image_mode"] == ImageMode.RGBD:
                image = cv2.resize(rgbd_image, (w, h))
                image = np.transpose(image, (2, 0, 1))
            else:
                raise NotImplementedError
            # depth_image = (depth_image - 0.8) / 0.2
            if len(self.ma_images_deque[i]) == 0:
                for j in range(self.image_seq_len - 1):
                    temp = np.zeros_like(image)
                    self.ma_images_deque[i].append(temp)
                    temp2 = np.zeros_like(relative_pose)
                    self.ma_relative_poses_deque[i].append(temp2)

            self.ma_images_deque[i].append(image)
            ma_images.append(np.array(self.ma_images_deque[i]))

            self.ma_relative_poses_deque[i].append(relative_pose)
            ma_relative_poses.append(np.array([self.ma_relative_poses_deque[i]]).flatten())

        for i in range(len(self.robots)):
            res.append([ma_images[i].reshape((-1, h, w)), ma_relative_poses[i]])
        return res

    def p_step_simulation(self):
        self.p.stepSimulation()
        self.physical_steps += 1

    def _check_collision(self):
        if check_collision(self.p, self.robot_ids, self.wall_obstacle_ids):
            return CollisionType.CollisionWithWall
        elif check_collision(self.p, self.robot_ids, self.pedestrian_obstacle_ids):
            return CollisionType.CollisionWithPedestrian
        elif check_collision(self.p, self.robot_ids, self.robot_ids):
            return CollisionType.CollisionWithAgent
        else:
            return False

    def iterate_steps(self, actions):
        iterate_count = 0
        reach_goals = []
        all_reach_goal = False
        # 0.4/0.05 = 8
        n_step = np.round(self.inference_every_duration / self.physical_step_duration)

        while iterate_count < n_step and not all_reach_goal:
            for i, robot in enumerate(self.robots):
                planned_v, planned_w = self.action_space.to_force(action=actions[i])
                reach_goal = compute_distance(robot.get_position(), self.bu_goals[i]) < self.env_config[
                    "goal_reached_thresh"]
                if reach_goal:
                    robot.small_step(0, 0)
                else:
                    robot.small_step(planned_v, planned_w)
                    if not self.args.train:
                        print("robot {} not reached goal".format(i))

            self.obstacle_collections.step()
            self.p_step_simulation()

            for i, robot in enumerate(self.robots):
                reach_goal = compute_distance(robot.get_position(), self.bu_goals[i]) < self.env_config[
                    "goal_reached_thresh"]
                reach_goals.append(reach_goal)

            iterate_count += 1
            all_reach_goal = all(reach_goals)

        collision = self._check_collision()
        return all_reach_goal, collision

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
                                                                      robot_start=self.bu_starts[0],
                                                                      robot_end=self.bu_goals[0])
        self.npc_goals = obs_bu_ends
        # create n dynamic obstacles, put them into the environment
        dynamic_obstacle_group = DynamicObstacleGroup(p=self.p,
                                                      client_id=self.client_id,
                                                      args=self.args,
                                                      step_duration=self.physical_step_duration,
                                                      paths=obs_bu_paths)

        self.obstacle_collections.add(dynamic_obstacle_group, dynamic=True)
        self.pedestrian_obstacle_ids = self.obstacle_collections.get_obstacle_ids()

    def load_env(self):
        """
        read env from path
        Returns:
        """
        map_path = self.args.load_map_from
        coordinates_from = self.args.load_coordinates_from

        maps, samplers, obstacle_ids, bu_starts, bu_goals = load_scene(self.p, self.env_config, self.worlds_config,
                                                                       map_path, coordinates_from)

        self.wall_obstacle_ids = obstacle_ids
        self.occ_map = maps["occ_map"]
        self.dilated_occ_map = maps["dilated_occ_map"]
        self.door_occ_map = maps["door_map"]
        self.start_goal_sampler, self.static_obs_sampler, self.dynamic_obs_sampler = samplers

        self.bu_starts = bu_starts
        self.bu_goals = [bu_goals[0] for i in range(self.num_agents)]

        logging.debug("Create the environment, Done...")
        self.robots = self.init_robots()

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
        maps, samplers, obstacle_ids, bu_starts, bu_goals = load_environment_scene(p=self.p,
                                                                                   env_config=self.env_config,
                                                                                   worlds_config=self.worlds_config)

        # sample start pose and goal pose
        self.wall_obstacle_ids = obstacle_ids
        self.occ_map = maps["occ_map"]
        self.dilated_occ_map = maps["dilated_occ_map"]
        self.door_occ_map = maps["door_map"]
        # extract samplers
        self.start_goal_sampler, self.static_obs_sampler, self.dynamic_obs_sampler = samplers

        self.bu_starts = bu_starts
        self.bu_goals = [bu_goals[0] for i in range(self.num_agents)]
        # initialize robot
        logging.debug("Create the environment, Done...")
        self.robots = self.init_robots()

    def init_robots(self):
        agents = []
        for i in range(self.num_agents):
            robot = init_robot(self.p, self.client_id, self.robot_name, RobotRoles.AGENT, self.physical_step_duration,
                               self.robot_config, self.sensor_config, self.bu_starts[i],
                               compute_yaw(self.bu_starts[i], self.bu_goals[i]))
            self.robot_ids.append(robot.robot_id)
            agents.append(robot)
        return agents

    def clear_variables(self):
        self.step_count = Counter()
        self.occ_map = None
        self.dilated_occ_map = None
        self.door_occ_map = None
        self.bu_starts, self.bu_goals = [None] * 2

        self.obstacle_collections.clear()
        self.last_distance = None
        self.wall_obstacle_ids = []

        self.ma_images_deque = [deque(maxlen=self.image_seq_len) for i in range(self.num_agents)]
        self.ma_relative_poses_deque = [deque(maxlen=self.pose_seq_len) for i in range(self.num_agents)]
        self.robots = None
        self.robot_ids = []

    def logging_action(self, action):
        logging_str = ""
        for key, item in zip(self.action_space_keys, action):
            logging_str += "{}: {}; ".format(key[:-1], item)
        logging.warning(logging_str)
