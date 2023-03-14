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
from typing import Dict, List

import cv2

from environment.env_types import EnvTypes
from environment.gen_scene.office1000_loader import load_office1000_scene, check_office1000_folder_structure
from environment.gen_scene.world_loader import load_p2v_scene
from environment.human_npc_generator import generate_human_npc
from environment.nav_utilities.coordinates_converter import cvt_positions_to_reference, \
    transform_local_to_world
from environment.nav_utilities.pybullet_helper import plot_robot_direction_line
from environment.robots.npc import NpcGroup
from environment.robots.object_robot import ObjectRobot
from environment.robots.robot_environment_bridge import RobotEnvBridge
from environment.robots.robot_roles import RobotRoles
from environment.robots.robot_types import RobotTypes, init_robot
from environment.sensors.sensor_types import SensorTypes
from utils.fo_utility import get_project_path, get_sg_walls_path, get_goal_at_door_path, get_sg_no_walls_path, \
    get_p2v_sg_walls_path, get_p2v_goal_at_door_path

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "traditional_planner", "a_star"))
print(sys.path)
import numpy as np
from agents.action_space.action_space import AbstractActionSpace, ContinuousXYYAWActionSpace, ContinuousXYActionSpace
from environment.base_pybullet_env import PybulletBaseEnv
from environment.nav_utilities.check_helper import check_collision, CollisionType
from environment.nav_utilities.counter import Counter
from utils.math_helper import compute_cosine_similarity
from environment.gen_scene.world_generator import load_environment_scene
from global_planning.a_star.astar import AStar


class EnvironmentBullet(PybulletBaseEnv):
    def __init__(self, args, action_space):
        PybulletBaseEnv.__init__(self, args)
        self.args = args
        self.running_config = args.running_config
        self.worlds_config = args.worlds_config
        self.agent_robot_config = args.robots_config[self.running_config["agent_robot_name"]]
        self.input_config = args.inputs_config[args.running_config["input_config_name"]]
        self.agent_sg_sampler_config = args.samplers_config[args.running_config["agent_sg_sampler"]]
        self.npc_sg_sampler_config = args.samplers_config[args.running_config["npc_sg_sampler"]]
        self.reward_config = args.rewards_config[self.running_config["reward_config_name"]]
        self.agent_robot_name = self.running_config["agent_robot_name"]
        self.sensors_name = self.running_config["sensors_name"]
        self.render = args.render

        self.grid_res = self.running_config["grid_res"]

        self.max_step = self.running_config["max_steps"]
        self.inference_duration = self.running_config["inference_duration"]
        self.action_space: AbstractActionSpace = action_space

        self.occ_map = None
        self.obstacle_distance_map = None
        self.force_u1_x, self.force_u1_y, self.force_u1 = None, None, None
        self.force_vxs, self.force_vys, self.force_vs = [], [], []
        self.agent_robots_copy: List[ObjectRobot] = []
        self.agent_robots: List[ObjectRobot] = []
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

        self.image_seq_len = self.input_config["image_seq_len"] if "image_seq_len" in self.input_config.keys() else 0
        self.pose_seq_len = self.input_config["pose_seq_len"] if "pose_seq_len" in self.input_config.keys() else 0

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
        self.geodesic_distance_dict_list: List[Dict] = None

        self.collision_count = 0
        self.max_collision_count = 5
        self.phase = Phase.TRAIN

    def render(self, mode="human"):
        width, height, rgb_image, depth_image, seg_image = self.agent_robots[0].sensors.get_obs()
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

        if not self.args.env == EnvTypes.RANDOM:
            self.load_office_evacuation()
            self.randomize_human_npc()
        else:
            # randomize environment
            self.randomize_env()
            self.randomize_human_npc()

        # # randomize environment
        state = self.get_state()
        if self.args.render:
            self.visualize_goals([robot.goal for robot in self.agent_robots],
                                 [[1, 0, 0, 1] for rb in self.agent_robots])
            for i, robot in enumerate(self.agent_robots):
                robot.bridge.plot_robot_direction_line(robot)
        return state

    def load_office_evacuation(self):
        check_office1000_folder_structure()
        if self.args.env == EnvTypes.OFFICE1500:
            parent_folders = [get_sg_walls_path(), get_goal_at_door_path(), get_sg_no_walls_path()]
            parent_folder = np.random.choice(parent_folders, size=(1,), p=np.array([0.8, 0.1, 0.1]))[0]
            parent_folder = get_sg_walls_path()
            print("scene folder:{}".format(parent_folder))
        elif self.args.env == EnvTypes.P2V:
            parent_folders = [get_p2v_sg_walls_path(), get_p2v_goal_at_door_path()]
            parent_folder = np.random.choice(parent_folders, size=(1,), p=np.array([0, 1]))[0]
            print("scene folder:{}".format(parent_folder))
        else:
            raise NotImplementedError

        occ_map, geodesic_distance_dict_list, obstacle_distance_map, force_ux, force_uy, force_u, force_vxs, force_vys, force_vs, wall_ids, agent_starts, agent_goals = load_office1000_scene(
            p=self.p,
            running_config=self.running_config,
            worlds_config=self.worlds_config,
            phase=self.phase,
            parent_folder=parent_folder
        )

        # sample start pose and goal pose
        self.wall_ids = wall_ids
        self.occ_map = occ_map
        self.obstacle_distance_map = obstacle_distance_map
        self.force_u1_x, self.force_u1_y, self.force_u1 = force_ux, force_uy, force_u
        self.force_vxs, self.force_vys, self.force_vs = force_vxs, force_vys, force_vs
        self.geodesic_distance_dict_list = geodesic_distance_dict_list

        # 如果有多个agent，去往同一个目标
        agent_goals = [agent_goals[0] for i in range(self.num_agents)]

        # display_v_images(self.force_vxs[0], self.force_vys[0], self.force_vs[0])
        # initialize robot
        logging.debug("Create the environment, Done...")
        self.agent_robots = self.init_robots(agent_starts, agent_goals)
        self.agent_robots_copy = [robot for robot in self.agent_robots]

    def visualize_goals(self, bu_goals, colors):
        thetas = np.linspace(0, np.pi * 2, 10)
        radius = 0.2 * np.random.random() + 0.1
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

    def get_action_space_keys(self):
        action_class = self.action_space.__class__.__name__
        action_space_keys = list(self.args.action_spaces_config[action_class].keys())
        return action_space_keys

    def step(self, actions):
        self.step_count += 1

        self.iterate_steps_xy_yaw_control(actions)

        state = self.get_state()

        reward, reward_info = self.get_reward()
        over_max_step = self.step_count >= self.max_step

        episode_info = {}

        if self.args.train:
            reach_goal = self.agent_robots[0].bridge.check_reach_goal()
            collision = self.agent_robots[0].bridge.check_collision()
            # 训练过程中，检查碰撞次数是否到达上限
            if collision == CollisionType.CollisionWithWall:
                self.collision_count += 1
            else:
                self.collision_count = 0

            # 判断是否完成此episode
            done = self.collision_count >= self.max_collision_count or reach_goal or over_max_step

        else:
            collision = self.agent_robots[0].bridge.check_collision()
            # 会有多个智能体
            reach_goals = [False for i in self.agent_robots]
            remove_ids = []
            # 检查是否到达目标
            for i, robot in enumerate(self.agent_robots):
                reach_goal = robot.bridge.check_reach_goal()
                if reach_goal:
                    remove_ids.append(i)
                    reach_goals[i] = reach_goal

            # check if all reach goal
            # 判断是否多个智能体都到达终点
            reach_goal = all(reach_goals)

            # 并将到达目标的robot删掉
            for i in remove_ids[::-1]:
                robot = self.agent_robots[i]
                self.agent_robots.remove(robot)
                robot.bridge.clear_itself()

            # 判断是否完成此episode
            done = reach_goal or over_max_step

        # done = reach_goal or over_max_step
        step_info = reward_info

        if done:
            # store information
            episode_info = {"collision": collision == CollisionType.CollisionWithWall, "a_success": reach_goal,
                            "over_max_step": over_max_step, "step_count": self.step_count.value,
                            "success_step_count": self.step_count.value}
            print("success:{}; collision:{}; over_max_step:{}".format(reach_goal, collision, over_max_step))

            if not self.args.train:
                self.add_episode_end_prompt(episode_info)

        # plot stored information
        return state, reward, done, step_info, episode_info

    def get_reward(self):
        for i, robot in enumerate(self.agent_robots):
            reward = 0

            """================collision reward=================="""
            collision_reward, collision = self.compute_collision_reward(robot)
            reward += collision_reward

            """================reach goal reward=================="""
            reach_goal_reward, reach_goal = self.compute_reach_goal_reward(robot)
            reward += reach_goal_reward

            """================delta euclidean distance reward=================="""
            # compute distance from current to goal
            obj_euclidean_distance_reward = self.compute_goal_euclidean_reward(robot)
            reward += obj_euclidean_distance_reward

            """================delta distance reward=================="""
            geo_obs_reward, reward_info_geo_obs = self.compute_geo_obs_reward(robot)
            reward += geo_obs_reward

            uv_reward, reward_info_uv = self.compute_uv_reward(robot)
            reward += uv_reward
            """=================obstacle distance reward==============="""
            reward_info = {"reward/reward_collision": np.around(collision_reward, 2),
                           "reward/reward_obj_euclidean_distance": np.around(obj_euclidean_distance_reward, 2),
                           "reward/reward_reach_goal": np.around(reach_goal_reward, 2),
                           }
            reward_info.update(reward_info_geo_obs)
            reward_info.update(reward_info_uv)
            reward_info.update({"reward/reward": np.around(reward, 2)})

            # 如果在训练，i == 0就返回
            if self.args.train:
                return reward, reward_info
            else:

                # print("robot:{}; reward info:{}".format(i, reward_info))
                pass
        return 0, {}

    def compute_geo_obs_reward(self, robot: ObjectRobot):
        reward = 0
        """================obstacle distance reward"""
        obstacle_distance_reward = self.compute_obstacle_distance_reward(robot)
        reward += obstacle_distance_reward
        """================delta geodesic distance reward=================="""
        geo_distance_reward = self.compute_goal_geo_reward(robot)
        reward += geo_distance_reward

        reward_info = {"reward/reward_obs": np.around(obstacle_distance_reward, 2),
                       "reward/reward_goal_geo": np.around(geo_distance_reward, 2)}

        return reward, reward_info

    def compute_goal_euclidean_reward(self, robot: ObjectRobot):
        delta_euclidean_distance = robot.bridge.compute_delta_euclidean_distance()
        delta_distance_reward = delta_euclidean_distance * self.reward_config["goal_euclidean"]
        return delta_distance_reward

    def compute_goal_geo_reward(self, robot: ObjectRobot):
        delta_geodesic_distance = robot.bridge.compute_delta_geodesic_distance()
        geo_distance_reward = delta_geodesic_distance * self.reward_config["goal_geo"]
        return geo_distance_reward

    def compute_collision_reward(self, robot: ObjectRobot):
        collision_reward = 0
        # 检测碰撞
        collision = robot.bridge.check_collision()

        if collision == CollisionType.CollisionWithWall:
            collision_reward = self.reward_config["collision"]
        return collision_reward, collision

    def compute_obstacle_distance_reward(self, robot: ObjectRobot):
        obstacle_distance = robot.bridge.compute_obstacle_distance()
        distance_thresh = 0.4
        min_distance = min(obstacle_distance, distance_thresh)
        obstacle_distance_reward = (distance_thresh - min_distance) * self.reward_config["obs_dist"]
        return obstacle_distance_reward

    def compute_uv_reward(self, robot: ObjectRobot):
        if "force_u1" not in self.reward_config.keys():
            return 0, {}
        s, s_direction = robot.bridge.compute_move_s()
        f_u1 = robot.bridge.get_force_u1()
        f_v = robot.bridge.get_force_v()

        cosine_similarity_u1 = compute_cosine_similarity(s_direction, f_u1)
        cosine_similarity_v = compute_cosine_similarity(s_direction, f_v)

        f_u1_scalar = np.linalg.norm(f_u1)
        f_v_scalar = np.linalg.norm(f_v)

        w_u1 = self.reward_config["force_u1"] * f_u1_scalar * s * cosine_similarity_u1
        w_v = self.reward_config["force_v"] * f_v_scalar * s * cosine_similarity_v

        w = w_u1 + w_v
        # print("cosine_similarity:{}".format(cosine_similarity))
        return w, {"reward/reward_u1": np.around(w_u1, 2), "reward/reward_v": np.around(w_v, 2)}

    def compute_reach_goal_reward(self, robot: ObjectRobot):
        reach_goal_reward = 0
        reach_goal = robot.bridge.check_reach_goal()
        if reach_goal:
            reach_goal_reward = self.reward_config["reach_goal"]
        return reach_goal_reward, reach_goal

    def get_state(self):
        if self.running_config["input_config_name"] == "input_lidar_vision_geo":
            return self.get_state6()
        elif len(self.sensors_name) >= 2:
            return self.get_state5()
        elif self.sensors_name[0] == SensorTypes.LidarSensor:
            return self.get_state2()

    def get_state6(self):
        # compute depth image
        ma_images = []
        ma_relative_poses = []
        ma_hit_fractions = []
        res = []
        w = 0
        h = 0
        for i, rt in enumerate(self.agent_robots):
            thetas, hit_fractions = rt.sensors[0].get_obs()

            width, height, rgba_image, depth_image, seg_image = rt.sensors[1].get_obs()
            depth_image = depth_image / rt.sensors[1].farVal

            relative_pose = cvt_positions_to_reference([rt.goal], rt.get_position(), rt.get_yaw())

            geodesic_distance = rt.bridge.compute_geodesic_distance()

            # if not self.args.train:
            #     print("input geodesic_distance:{}".format(geodesic_distance))
            rela_pose_geo = np.array([*relative_pose.flatten().tolist(), geodesic_distance]).astype(float)[np.newaxis,
                            :]
            # print(rela_pose_geo.dtype)
            # print(rela_pose_geo)
            w = self.input_config["image_w"]
            h = self.input_config["image_h"]
            image = cv2.resize(depth_image, (w, h))
            image[np.isnan(image)] = 1
            if len(self.ma_images_deque[i]) == 0:
                for j in range(self.image_seq_len - 1):
                    temp = np.zeros_like(image)
                    self.ma_images_deque[i].append(temp)
                for j in range(self.pose_seq_len - 1):
                    temp2 = np.zeros_like(rela_pose_geo)
                    self.ma_relative_poses_deque[i].append(temp2)
            # plt.imshow(depth_image)
            # plt.show()
            self.ma_images_deque[i].append(image)
            ma_images.append(np.array(self.ma_images_deque[i]))

            self.ma_relative_poses_deque[i].append(rela_pose_geo)
            ma_relative_poses.append(np.array([self.ma_relative_poses_deque[i]]).flatten())

            ma_hit_fractions.append(hit_fractions.flatten().astype(float))

        for i in range(len(self.agent_robots)):
            res.append([ma_images[i].reshape((-1, h, w)), ma_hit_fractions[i], ma_relative_poses[i]])
        return res

    def get_state5(self):
        # compute depth image
        ma_images = []
        ma_relative_poses = []
        ma_hit_fractions = []
        res = []
        w = 0
        h = 0
        for i, rt in enumerate(self.agent_robots):
            thetas, hit_fractions = rt.sensors[0].get_obs()
            # plt.clf()
            # plt.polar(thetas, hit_fractions)
            # # plt.show()
            # plt.tight_layout()
            # plt.savefig(os.path.join(get_project_path(), "output", "lidar_step_{}.png".format(self.step_count.value)))

            width, height, rgba_image, depth_image, seg_image = rt.sensors[1].get_obs()
            depth_image = depth_image / rt.sensors[1].farVal
            # plt.clf()
            # plt.imshow(depth_image)
            # plt.axis("off")
            # plt.tight_layout()
            # # plt.show()
            # plt.savefig(os.path.join(get_project_path(), "output", "depth_step_{}.png".format(self.step_count.value)))

            relative_pose = cvt_positions_to_reference([rt.goal], rt.get_position(), rt.get_yaw())
            w = self.input_config["image_w"]
            h = self.input_config["image_h"]
            image = cv2.resize(depth_image, (w, h))
            image[np.isnan(image)] = 1
            hit_fractions = hit_fractions.astype(float)
            hit_fractions[np.isnan(hit_fractions)] = 1
            relative_pose = relative_pose / rt.sensors[1].farVal
            if len(self.ma_images_deque[i]) == 0:
                for j in range(self.image_seq_len - 1):
                    temp = np.zeros_like(image)
                    self.ma_images_deque[i].append(temp)
                for j in range(self.pose_seq_len - 1):
                    temp2 = np.zeros_like(relative_pose)
                    self.ma_relative_poses_deque[i].append(temp2)
            # plt.imshow(depth_image)
            # plt.show()
            self.ma_images_deque[i].append(image)
            ma_images.append(np.array(self.ma_images_deque[i]))

            self.ma_relative_poses_deque[i].append(relative_pose)
            ma_relative_poses.append(np.array([self.ma_relative_poses_deque[i]]).flatten())

            ma_hit_fractions.append(hit_fractions.flatten().astype(float))

        for i in range(len(self.agent_robots)):
            res.append([ma_images[i].reshape((-1, h, w)), ma_hit_fractions[i], ma_relative_poses[i]])
        return res

    def get_state2(self):
        res = []
        for i, rt in enumerate(self.agent_robots):
            thetas, hit_fractions = rt.sensor.get_obs()
            relative_pose = cvt_positions_to_reference([rt.goal], rt.get_position(), rt.get_yaw())
            state = np.concatenate([hit_fractions, relative_pose.flatten()], axis=0).flatten()
            res.append(state.astype(float))

        return res

    def p_step_simulation(self):
        self.p.stepSimulation()
        self.physical_steps += 1

    def iterate_steps_xy_yaw_control(self, actions):
        iterate_count = 0
        n_step = np.round(self.inference_duration / self.physical_step_duration)
        while iterate_count < n_step:
            for i, robot in enumerate(self.agent_robots):
                delta_x, delta_y, delta_yaw = self.action_space.to_force(action=actions[i])
                # 机器人n_step步将delta_x, delta_y, delta_yaw走完
                d_x, d_y, d_yaw = delta_x / n_step, delta_y / n_step, delta_yaw / n_step

                d_x, d_y = transform_local_to_world(np.array([d_x, d_y]), robot.get_position(),
                                                    robot.get_yaw()) - robot.get_position()

                robot.small_step_xy_yaw_control(d_x, d_y, d_yaw)

                # 画机器人朝向线条
                robot.bridge.plot_robot_direction_line(self.render)

            if self.npc_group is not None:
                self.npc_group.step()

            # 物理模拟一步
            self.p_step_simulation()

            iterate_count += 1

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

    def load_p2v_env(self):
        """
        read env from path
        Returns:
        """
        map_path = self.args.load_map_from
        coordinates_from = self.args.load_coordinates_from

        occ_map, wall_ids, agent_starts, agent_goals = load_p2v_scene(self.p, self.running_config, self.worlds_config,
                                                                      map_path, coordinates_from, self.num_agents)

        self.wall_ids = wall_ids
        self.occ_map = occ_map

        agent_goals = [agent_goals[0] for i in range(self.num_agents)]

        logging.debug("Create the environment, Done...")
        self.agent_robots = self.init_robots(agent_starts, agent_goals)

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
        # 如果有多个agent，去往同一个目标
        agent_goals = [agent_goals[0] for i in range(self.num_agents)]
        # initialize robot
        logging.debug("Create the environment, Done...")
        self.agent_robots = self.init_robots(agent_starts, agent_goals)

    def init_robots(self, agent_starts, agent_goals):
        agents = []
        for i in range(self.num_agents):
            robot = init_robot(self.p, self.client_id, self.agent_robot_name, RobotRoles.AGENT,
                               self.physical_step_duration,
                               self.agent_robot_config, self.sensors_name, self.args.sensors_config,
                               agent_starts[i], agent_goals[i],
                               2 * np.pi * np.random.random())
            robot.set_bridge(RobotEnvBridge(robot, self, i))
            self.agent_robot_ids.append(robot.robot_id)
            agents.append(robot)

        return agents

    def clear_variables(self):
        self.step_count = Counter()
        self.occ_map = None
        self.obstacle_distance_map = None
        if self.npc_group is not None:
            self.npc_group.clear()
            self.npc_group = None
        self.wall_ids = []
        self.collision_count = 0

        self.ma_images_deque = [deque(maxlen=self.image_seq_len) for i in range(self.num_agents)]
        self.ma_relative_poses_deque = [deque(maxlen=self.pose_seq_len) for i in range(self.num_agents)]
        self.agent_robots = None
        self.agent_robot_ids = []
        self.agent_sub_goals = None

    def logging_action(self, action):
        logging_str = ""
        for key, item in zip(self.action_space_keys, action):
            logging_str += "{}: {}; ".format(key[:-1], item)
        logging.warning(logging_str)


class Phase:
    TRAIN = "train"
    TEST = "test"
