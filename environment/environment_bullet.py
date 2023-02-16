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
from matplotlib import pyplot as plt

from environment.gen_scene.world_loader import load_scene
from environment.human_npc_generator import generate_human_npc
from environment.nav_utilities.coordinates_converter import cvt_to_om, cvt_to_bu
from environment.robots.npc import NpcGroup
from environment.robots.robot_roles import RobotRoles
from environment.robots.robot_types import RobotTypes, init_robot
from environment.sensors.sensor_types import SensorTypes
from environment.sensors.vision_sensor import ImageMode
from global_planning.prm.prm_path_planning import prm_path_planning
from utils.image_utility import dilate_image

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "traditional_planner", "a_star"))
print(sys.path)
import numpy as np
from agents.action_space.action_space import AbstractActionSpace
from environment.base_pybullet_env import PybulletBaseEnv
from environment.nav_utilities.check_helper import check_collision, CollisionType
from environment.nav_utilities.counter import Counter
from utils.math_helper import compute_yaw, compute_distance, compute_manhattan_distance
from environment.gen_scene.world_generator import load_environment_scene
from global_planning.a_star.astar import AStar


class EnvironmentBullet(PybulletBaseEnv):
    def __init__(self, args, action_space):
        PybulletBaseEnv.__init__(self, args)
        self.args = args
        self.running_config = args.running_config
        self.worlds_config = args.worlds_config
        self.agent_robot_config = args.robots_config[self.running_config["agent_robot_name"]]
        self.sensor_config = args.sensors_config[self.running_config["sensor_name"]]
        self.input_config = args.inputs_config[args.running_config["input_config_name"]]
        self.agent_sg_sampler_config = args.samplers_config[args.running_config["agent_sg_sampler"]]
        self.npc_sg_sampler_config = args.samplers_config[args.running_config["npc_sg_sampler"]]
        self.reward_config = args.rewards_config[self.running_config["reward_config_name"]]
        self.agent_robot_name = self.running_config["agent_robot_name"]
        self.sensor_name = self.running_config["sensor_name"]
        self.render = args.render

        self.grid_res = self.running_config["grid_res"]

        self.max_step = self.running_config["max_steps"]
        self.inference_duration = self.running_config["inference_duration"]
        self.action_space: AbstractActionSpace = action_space

        self.occ_map = None

        self.agent_starts, self.agent_goals = [None] * 2
        self.agent_robots = None
        self.agent_robot_ids = []

        self.num_agents = self.args.running_config["num_agents"]

        self.npc_goals = []
        self.wall_ids = []
        self.npc_ids = []
        self.npc_group: NpcGroup = None

        # generate human agent, and other human npc
        self.start_goal_sampler, self.static_obs_sampler, self.dynamic_obs_sampler = None, None, None
        self.action_space_keys = self.get_action_space_keys()
        self.physical_steps = Counter()

        self.last_distance = None

        self.image_seq_len = self.args.inputs_config[self.running_config['input_config_name']]["image_seq_len"]
        self.pose_seq_len = self.args.inputs_config[self.running_config['input_config_name']]["pose_seq_len"]

        self.ma_relative_poses_deque = []
        self.ma_images_deque = None
        """
        initialize environment
        initialize dynamic human npc
        initialize the human agent
        compute the relative position to the target
        
        
        """
        self.agent_sub_goals = None
        self.agent_sub_goals_indexes = None
        self.paths = None
        self.temp_ids = []

    def render(self, mode="human"):
        width, height, rgb_image, depth_image, seg_image = self.agent_robots[0].sensor.get_obs()
        image = cv2.resize(depth_image, (40, 60))
        image = image[:16, :]
        h = 16
        return width, height, rgb_image, image, seg_image

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
        if self.args.prm:
            self.paths = self.get_all_path()
            self.update_agent_sub_goals()
        state = self.get_state()
        if self.args.render:
            self.visualize_goals(self.agent_goals, [[1, 0, 0, 1] for rb in self.agent_robots])
            # self.visualize_goals(self.npc_goals, self.obstacle_collections.get_obstacle_ids())
        return state

    def visualize_goals(self, bu_goals, colors):
        thetas = np.linspace(0, np.pi * 2, 10)
        radius = 0.2 * np.random.random()
        ids = []
        for i in range(len(bu_goals)):
            color = colors[i]
            points_x = np.cos(thetas) * radius + bu_goals[i][0]
            points_y = np.sin(thetas) * radius + bu_goals[i][1]
            z = np.zeros_like(thetas)
            points = np.array([points_x, points_y, z]).T
            points_next = np.roll(points, -1, axis=0)
            # froms = [[1, 1, 0], [-1, 1, 0], [-1, 1, 3], [1, 1, 3]]
            # tos = [[-1, 1, 0], [-1, 1, 3], [1, 1, 3], [1, 1, 0]]
            for f, t in zip(points, points_next):
                id = self.p.addUserDebugLine(
                    lineFromXYZ=f,
                    lineToXYZ=t,
                    lineColorRGB=color[:3],
                    lineWidth=2
                )
                ids.append(id)
        return ids

    def remove_debug_items(self, ids):
        for id in ids:
            self.p.removeUserDebugItem(id)

    def add_debug_line(self, robot, points):
        points_x = points[:, 0]
        points_y = points[:, 1]
        z = np.zeros_like(points_x)

        points = np.array([points_x, points_y, z]).T
        points_next = np.roll(points, -1, axis=0)
        for f, t in zip(points[:-1], points_next[:-1]):
            self.p.addUserDebugLine(
                lineFromXYZ=f,
                lineToXYZ=t,
                lineColorRGB=robot.color,
                lineWidth=2
            )

    def get_action_space_keys(self):
        action_class = self.action_space.__class__.__name__
        action_space_keys = list(self.args.action_spaces_config[action_class].keys())
        return action_space_keys

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
            done = reach_goal or over_max_step
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
            self.last_distance = compute_distance(self.agent_goals[0], self.agent_starts)
        reward = 0
        collision_reward = 0
        reach_goal_reward = 0
        delta_distance_reward = 0
        distance = 0
        """================collision reward=================="""
        if collision == CollisionType.CollisionWithWall:
            collision_reward = self.reward_config["collision"]
            reward += collision_reward

        """================delta distance reward=================="""
        # compute distance from current to goal
        if self.step_count.value % 6 == 0:
            distance = compute_distance(self.agent_goals[0], self.agent_robots[0].get_position())
            delta_distance_reward = (self.last_distance - distance) * self.reward_config["delta_distance"]
            self.last_distance = distance
            reward += delta_distance_reward

        """================reach goal reward=================="""

        if reach_goal:
            reach_goal_reward = self.reward_config["reach_goal"]
            reward += reach_goal_reward

        reward_info = {"reward/reward_collision": collision_reward,
                       "reward/reward_delta_distance": delta_distance_reward,
                       "reward/distance": distance,
                       "reward/reward_reach_goal": reach_goal_reward,
                       "reward/reward": reward
                       }
        return reward, reward_info

    def get_state(self):
        return self.get_state1()

    def get_all_path(self):
        paths = []
        prm = prm_path_planning(dilate_image(self.occ_map.copy(), 2), num_samples=100, end=self.agent_goals[0],
                                grid_res=self.grid_res)
        for i in range(len(self.agent_robots)):
            start = self.agent_starts[i]
            goal = self.agent_goals[i]
            robot = self.agent_robots[i]
            path = prm.get_shortest_path_from_start(cvt_to_om(start, self.grid_res))
            prm.display_path(path)
            path = cvt_to_bu(path, self.grid_res)
            paths.append(path)
            self.add_debug_line(robot, path)
        plt.show()
        return paths

    def update_agent_sub_goals(self):
        if self.agent_sub_goals is None:
            self.agent_sub_goals = []
            self.agent_sub_goals_indexes = []
            for i, rt in enumerate(self.agent_robots):
                self.agent_sub_goals_indexes.append(0)
                self.agent_sub_goals.append(self.paths[i][0])

        else:
            for i, rt in enumerate(self.agent_robots):
                reach_sub_goal = compute_distance(rt.get_position(), self.agent_sub_goals[i]) < 1.2
                if reach_sub_goal:
                    self.temp_ids = self.visualize_goals([self.agent_sub_goals[i]], [self.agent_robots[i].color])
                    self.agent_sub_goals_indexes[i] = min(self.agent_sub_goals_indexes[i] + 1, len(self.paths[i]) - 1)
        print()
        for i, index in enumerate(self.agent_sub_goals_indexes):
            self.agent_sub_goals[i] = self.paths[i][index]
        # self.remove_debug_items(self.temp_ids)
        # self.temp_ids = self.visualize_goals(self.agent_sub_goals, self.agent_robots)

    def get_state1(self):
        # compute depth image
        ma_images = []
        ma_relative_poses = []
        res = []
        w = 0
        h = 0
        for i, rt in enumerate(self.agent_robots):
            width, height, rgba_image, depth_image, seg_image = rt.sensor.get_obs()
            rgba_image = rgba_image / 255
            depth_image = (depth_image - 0.75) / 0.25
            if self.args.prm:
                relative_position = self.agent_sub_goals[i] - rt.get_position()
                relative_yaw = compute_yaw(self.agent_sub_goals[i], rt.get_position()) - rt.get_yaw()
            else:
                relative_position = self.agent_goals[i] - rt.get_position()
                relative_yaw = compute_yaw(self.agent_goals[i], rt.get_position()) - rt.get_yaw()
            relative_pose = np.array([relative_position[0], relative_position[1], relative_yaw])

            w = self.input_config["image_w"]
            h = self.input_config["image_h"]
            if self.input_config["image_mode"] == ImageMode.MULTI_ROW_MULTI_SENSOR:
                _w = int(depth_image.shape[2] / 2)
                _h = int(depth_image.shape[1] / 2)
                image = np.transpose(depth_image, (1, 2, 0))
                image = cv2.resize(image, (_w, _h))
                image = image[:h, :]
                image = np.transpose(image, (2, 0, 1))
            elif self.input_config["image_mode"] == ImageMode.ROW:
                image = depth_image[25]
                h = 1
                w = int(depth_image.shape[1])
            elif self.input_config["image_mode"] == ImageMode.MULTI_ROW:
                image = cv2.resize(depth_image, (int(depth_image.shape[1] / 2), int(depth_image.shape[0] / 2)))
                image = image[:h, :]
                # plt.imsave("")
                # plt.imshow(image)
                # plt.show()
            elif self.input_config["image_mode"] == ImageMode.DEPTH:
                image = cv2.resize(depth_image, (w, h))
            elif self.input_config["image_mode"] == ImageMode.RGB:
                image = cv2.resize(rgba_image[:, :, :3], (w, h))
                image = np.transpose(image, (2, 0, 1))
            elif self.input_config["image_mode"] == ImageMode.RGBD:
                image = np.append(rgba_image[:, :, :3], depth_image[:, :, np.newaxis], axis=-1)
                image = cv2.resize(image, (w, h))
                image = np.transpose(image, (2, 0, 1))
            else:
                raise NotImplementedError
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

        for i in range(len(self.agent_robots)):
            res.append([ma_images[i].reshape((-1, h, w)), ma_relative_poses[i]])
        return res

    def p_step_simulation(self):
        self.p.stepSimulation()
        self.physical_steps += 1

    def _check_collision(self):
        if check_collision(self.p, self.agent_robot_ids, self.wall_ids):
            return CollisionType.CollisionWithWall
        elif check_collision(self.p, self.agent_robot_ids, self.npc_ids):
            return CollisionType.CollisionWithPedestrian
        elif check_collision(self.p, self.agent_robot_ids, self.agent_robot_ids):
            return CollisionType.CollisionWithAgent
        else:
            return False

    def iterate_steps(self, actions):
        iterate_count = 0
        reach_goals = []
        all_reach_goal = False
        # 0.4/0.05 = 8
        n_step = np.round(self.inference_duration / self.physical_step_duration)

        while iterate_count < n_step:
            for i, robot in enumerate(self.agent_robots):
                planned_v, planned_w = self.action_space.to_force(action=actions[i])
                reach_goal = compute_distance(robot.get_position(), self.agent_goals[i]) < self.running_config[
                    "goal_reached_thresh"]
                if reach_goal:
                    robot.small_step(0, 0)
                else:
                    robot.small_step(planned_v, planned_w)
                    if not self.args.train:
                        print("robot {} not reached goal".format(i))
            if self.npc_group is not None:
                self.npc_group.step()
            self.p_step_simulation()

            iterate_count += 1

        collision = self._check_collision()

        # check if all reach goal
        for i, robot in enumerate(self.agent_robots):
            reach_goal = compute_distance(robot.get_position(), self.agent_goals[i]) < self.running_config[
                "goal_reached_thresh"]
            reach_goals.append(reach_goal)
        all_reach_goal = all(reach_goals)

        if self.args.prm:
            self.update_agent_sub_goals()
        # self.print_v()
        return all_reach_goal, collision

    def print_v(self):
        for i, robot in enumerate(self.agent_robots):
            print("robot id : {}; v:{}".format(robot.robot_id, robot.get_v()))
        for i, npc in enumerate(self.npc_group.npc_robots):
            print("npc id : {}; v:{}".format(npc.robot.robot_id, npc.robot.get_v()))

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
        npc_starts, npc_goals, npc_paths = generate_human_npc(running_config=self.running_config,
                                                              occ_map=self.occ_map,
                                                              npc_sg_sampler_config=self.npc_sg_sampler_config)
        self.npc_goals = npc_goals
        # create n dynamic obstacles, put them into the environment
        self.npc_group = NpcGroup(p=self.p,
                                  client_id=self.client_id,
                                  args=self.args,
                                  step_duration=self.physical_step_duration,
                                  paths=npc_paths)
        self.npc_ids = self.npc_group.npc_robot_ids

    def load_env(self):
        """
        read env from path
        Returns:
        """
        map_path = self.args.load_map_from
        coordinates_from = self.args.load_coordinates_from

        occ_map, wall_ids, agent_starts, agent_goals = load_scene(self.p, self.running_config, self.worlds_config,
                                                                  map_path, coordinates_from)

        self.wall_ids = wall_ids
        self.occ_map = occ_map

        self.agent_starts = agent_starts
        self.agent_goals = [agent_goals[0] for i in range(self.num_agents)]

        logging.debug("Create the environment, Done...")
        self.agent_robots = self.init_robots()

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
        occ_map, wall_ids, agent_starts, agent_goals = load_environment_scene(p=self.p,
                                                                              running_config=self.running_config,
                                                                              worlds_config=self.worlds_config,
                                                                              agent_sg_sampler_config=self.agent_sg_sampler_config)

        # sample start pose and goal pose
        self.wall_ids = wall_ids
        self.occ_map = occ_map
        self.agent_starts = agent_starts
        # 如果有多个agent，去往同一个目标
        self.agent_goals = [agent_goals[0] for i in range(self.num_agents)]
        # initialize robot
        logging.debug("Create the environment, Done...")
        self.agent_robots = self.init_robots()

    def init_robots(self):
        agents = []
        for i in range(self.num_agents):
            robot = init_robot(self.p, self.client_id, self.agent_robot_name, RobotRoles.AGENT,
                               self.physical_step_duration,
                               self.agent_robot_config, self.sensor_name, self.sensor_config, self.agent_starts[i],
                               compute_yaw(self.agent_starts[i], self.agent_goals[i]))
            self.agent_robot_ids.append(robot.robot_id)
            agents.append(robot)
        return agents

    def clear_variables(self):
        self.step_count = Counter()
        self.occ_map = None
        if self.npc_group is not None:
            self.npc_group.clear()
            self.npc_group = None
        self.last_distance = None
        self.wall_ids = []

        self.ma_images_deque = [deque(maxlen=self.image_seq_len) for i in range(self.num_agents)]
        self.ma_relative_poses_deque = [deque(maxlen=self.pose_seq_len) for i in range(self.num_agents)]
        self.agent_robots = None
        self.agent_robot_ids = []
        self.agent_starts, self.agent_goals = [None] * 2
        self.agent_sub_goals = None
        self.paths = None

    def logging_action(self, action):
        logging_str = ""
        for key, item in zip(self.action_space_keys, action):
            logging_str += "{}: {}; ".format(key[:-1], item)
        logging.warning(logging_str)
