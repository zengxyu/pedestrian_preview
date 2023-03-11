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
import pickle
import sys
import time
from collections import deque
from typing import Dict, List

import cv2
from matplotlib import pyplot as plt

from environment.env_types import EnvTypes
from environment.gen_scene.build_office_world import create_cylinder
from environment.gen_scene.office1000_door_loader import check_office1000_goal_outdoor_folder_structure, \
    load_office1000_goal_outdoor, load_office1000_goal_scene
from environment.gen_scene.office1000_loader import load_office1000_scene, check_office1000_folder_structure
from environment.gen_scene.world_loader import load_p2v_scene
from environment.human_npc_generator import generate_human_npc
from environment.nav_utilities.coordinates_converter import cvt_to_om, cvt_to_bu, cvt_positions_to_reference, \
    transform_local_to_world
from environment.nav_utilities.pybullet_helper import plot_robot_direction_line
from environment.robots.npc import NpcGroup
from environment.robots.robot_roles import RobotRoles
from environment.robots.robot_types import RobotTypes, init_robot
from environment.sensors.sensor_types import SensorTypes
from environment.sensors.vision_sensor import ImageMode
from utils.compute_v_forces import display_v_images
from utils.fo_utility import get_project_path
from utils.image_utility import dilate_image

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "traditional_planner", "a_star"))
print(sys.path)
import numpy as np
from agents.action_space.action_space import AbstractActionSpace, ContinuousXYYAWActionSpace, ContinuousXYActionSpace
from environment.base_pybullet_env import PybulletBaseEnv
from environment.nav_utilities.check_helper import check_collision, CollisionType
from environment.nav_utilities.counter import Counter
from utils.math_helper import compute_yaw, compute_distance, compute_manhattan_distance, compute_cosine_similarity
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
        self.last_geodesic_distance = None
        self.last_position = None
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
        self.robot_direction_ids = [None] * self.num_agents
        self.geodesic_distance_dict_list: List[Dict] = None

        self.collision_count = 0
        self.max_collision_count = 5
        self.reach_goals = [False for i in range(self.num_agents)]
        self.phase = Phase.TRAIN

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

        if self.args.env == EnvTypes.OFFICE1500:
            self.load_office_evacuation()
            self.randomize_human_npc()
        elif self.args.env == EnvTypes.P2V:
            assert self.args.load_map_from is not None and self.args.load_map_from != "", "args.load_map_from is None and args.load_map_from == ''"
            self.load_p2v_env()
        else:
            # randomize environment
            self.randomize_env()
            self.randomize_human_npc()
        # create_cylinder(self.p, self.agent_goals[0], with_collision=False, height=3, radius=0.1)

        # # randomize environment
        state = self.get_state()
        if self.args.render:
            self.visualize_goals(self.agent_goals, [[1, 0, 0, 1] for rb in self.agent_robots])
            for i, robot in enumerate(self.agent_robots):
                robot_direction_id = plot_robot_direction_line(self.p, self.robot_direction_ids[i], robot.get_x_y_yaw())
                self.robot_direction_ids[i] = robot_direction_id
        # print(self.agent_robots[0].get_position())
        return state

    def load_office_evacuation(self):
        check_office1000_folder_structure()
        occ_map, geodesic_distance_dict_list, obstacle_distance_map, force_ux, force_uy, force_u, force_vxs, force_vys, force_vs, wall_ids, agent_starts, agent_goals = load_office1000_scene(
            p=self.p,
            running_config=self.running_config,
            worlds_config=self.worlds_config,
            phase=self.phase)

        # sample start pose and goal pose
        self.wall_ids = wall_ids
        self.occ_map = occ_map
        self.obstacle_distance_map = obstacle_distance_map
        self.force_u1_x, self.force_u1_y, self.force_u1 = force_ux, force_uy, force_u
        self.force_vxs, self.force_vys, self.force_vs = force_vxs, force_vys, force_vs
        self.agent_starts = agent_starts
        # 如果有多个agent，去往同一个目标
        self.agent_goals = [agent_goals[0] for i in range(self.num_agents)]

        # display_v_images(self.force_vxs[0], self.force_vys[0], self.force_vs[0])
        # initialize robot
        logging.debug("Create the environment, Done...")
        self.agent_robots = self.init_robots()
        self.geodesic_distance_dict_list = geodesic_distance_dict_list

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
        # print("actions:{}".format(actions))
        if isinstance(self.action_space, ContinuousXYYAWActionSpace):
            reach_goal, collision = self.iterate_steps_xy_yaw_control(actions)
        elif isinstance(self.action_space, ContinuousXYActionSpace):
            reach_goal, collision = self.iterate_steps_xy_control(actions)
        else:
            raise NotImplementedError
        # reach_goal, collision = self.iterate_steps(actions)
        # print("v:{};w:{}".format(*self.robots[0].get_v_w()))
        state = self.get_state()
        reward, reward_info = self.get_reward(reach_goal=reach_goal, collision=collision)
        if not self.args.train:
            print("reward={}".format(reward))
        over_max_step = self.step_count >= self.max_step
        if collision == CollisionType.CollisionWithWall:
            self.collision_count += 1
        else:
            self.collision_count = 0

        # whether done
        if self.args.train:
            done = self.collision_count >= self.max_collision_count or reach_goal or over_max_step
        else:
            done = reach_goal or over_max_step

        # done = reach_goal or over_max_step
        step_info = reward_info

        # store information
        episode_info = {"collision": collision == CollisionType.CollisionWithWall, "a_success": reach_goal,
                        "over_max_step": over_max_step, "step_count": self.step_count.value}
        if reach_goal:
            episode_info.update({"success_step_count": self.step_count.value})
        if done:
            print("success:{}; collision:{}; over_max_step:{}".format(reach_goal, collision, over_max_step))

        if done and not self.args.train:
            self.add_episode_end_prompt(episode_info)

        # plot stored information
        return state, reward, done, step_info, episode_info

    def compute_geodesic_distance(self, robot_index, cur_position):
        occ_pos = cvt_to_om(cur_position, self.grid_res)
        occ_pos = tuple(occ_pos)

        if self.geodesic_distance_dict_list is None:
            return 0
        geodesic_distance_map = self.geodesic_distance_dict_list[robot_index]
        if occ_pos in geodesic_distance_map.keys():
            geodesic_distance = geodesic_distance_map[occ_pos]
        else:
            geodesic_distance = 100
        geodesic_distance = geodesic_distance * self.grid_res
        # print("geodesic_distance:{}".format(geodesic_distance))
        return geodesic_distance

    def compute_obstacle_distance(self, cur_position):
        occ_pos = cvt_to_om(cur_position, self.grid_res)
        if self.obstacle_distance_map is None:
            return 0
        x = np.clip(occ_pos[0], 0, self.obstacle_distance_map.shape[0] - 1)
        y = np.clip(occ_pos[1], 0, self.obstacle_distance_map.shape[1] - 1)

        obstacle_distance = self.obstacle_distance_map[x, y]
        # print("obstacle distance:{}".format(obstacle_distance))
        obstacle_distance = obstacle_distance * self.grid_res
        return obstacle_distance

    def get_reward(self, reach_goal, collision):
        reward = 0
        """================collision reward=================="""
        collision_reward = self.compute_collision_reward(collision)
        reward += collision_reward

        """================delta distance reward=================="""
        # compute distance from current to goal
        obj_euclidean_distance_reward = self.compute_goal_euclidean_reward()
        reward += obj_euclidean_distance_reward

        """================reach goal reward=================="""
        reach_goal_reward = self.compute_reach_goal_reward(reach_goal)
        reward += reach_goal_reward

        geo_obs_reward, reward_info_geo_obs = self.compute_geo_obs_reward()
        reward += geo_obs_reward

        uv_reward, reward_info_uv = self.compute_uv_reward()
        reward += uv_reward
        """=================obstacle distance reward==============="""
        reward_info = {"reward/reward_collision": np.around(collision_reward, 2),
                       "reward/reward_obj_euclidean_distance": np.around(obj_euclidean_distance_reward, 2),
                       "reward/reward_reach_goal": np.around(reach_goal_reward, 2),
                       }
        reward_info.update(reward_info_geo_obs)
        reward_info.update(reward_info_uv)
        reward_info.update({"reward/reward": np.around(reward, 2)})
        print("reward info:{}".format(reward_info))

        return reward, reward_info

    def compute_geo_obs_reward(self):
        reward = 0
        """================obstacle distance reward"""
        obstacle_distance_reward = self.compute_obstacle_distance_reward()
        reward += obstacle_distance_reward
        """================delta geodesic distance reward=================="""
        geo_distance_reward = self.compute_goal_geo_reward()
        reward += geo_distance_reward

        reward_info = {"reward/reward_obs": np.around(obstacle_distance_reward, 2),
                       "reward/reward_goal_geo": np.around(geo_distance_reward, 2)}

        return reward, reward_info

    def compute_goal_euclidean_reward(self):
        if self.last_distance is None:
            self.last_distance = compute_distance(self.agent_goals[0], self.agent_starts)

        distance = compute_distance(self.agent_goals[0], self.agent_robots[0].get_position())
        delta_distance_reward = (self.last_distance - distance) * self.reward_config["goal_euclidean"]
        self.last_distance = distance
        return delta_distance_reward

    def compute_goal_geo_reward(self):
        if self.last_geodesic_distance is None:
            self.last_geodesic_distance = self.compute_geodesic_distance(robot_index=0, cur_position=self.agent_robots[
                0].get_position())
        geodesic_distance = self.compute_geodesic_distance(robot_index=0,
                                                           cur_position=self.agent_robots[0].get_position())

        geo_distance_reward = (self.last_geodesic_distance - geodesic_distance) * self.reward_config["goal_geo"]
        self.last_geodesic_distance = geodesic_distance
        return geo_distance_reward

    def compute_collision_reward(self, collision):
        collision_reward = 0
        if collision == CollisionType.CollisionWithWall:
            collision_reward = self.reward_config["collision"]
        return collision_reward

    def compute_obstacle_distance_reward(self):
        obstacle_distance = self.compute_obstacle_distance(cur_position=self.agent_robots[0].get_position())
        distance_thresh = 0.4
        min_distance = min(obstacle_distance, distance_thresh)
        obstacle_distance_reward = (distance_thresh - min_distance) * self.reward_config["obs_dist"]
        return obstacle_distance_reward

    def compute_uv_reward(self):
        if self.force_u1_x is None:
            return 0, {}
        if "force_u1" not in self.reward_config.keys():
            return 0, {}
        cur_position = self.agent_robots[0].get_position()
        if self.last_position is None:
            self.last_position = cur_position

        cur_position_om = cvt_to_om(cur_position, self.grid_res)
        last_position_om = cvt_to_om(self.last_position, self.grid_res)
        self.last_position = cur_position

        s = np.linalg.norm(cur_position_om - last_position_om)
        s_direction = cur_position_om - last_position_om

        f_u1 = self.get_force_u1(cur_position_om)
        f_v = self.get_force_v(robot_index=0, pos=cur_position_om)

        cosine_similarity_u1 = compute_cosine_similarity(s_direction, f_u1)
        cosine_similarity_v = compute_cosine_similarity(s_direction, f_v)

        f_u1_scalar = np.linalg.norm(f_u1)
        f_v_scalar = np.linalg.norm(f_v)

        w_u1 = self.reward_config["force_u1"] * f_u1_scalar * s * cosine_similarity_u1
        w_v = self.reward_config["force_v"] * f_v_scalar * s * cosine_similarity_v

        w = w_u1 + w_v
        # print("cosine_similarity:{}".format(cosine_similarity))
        return w, {"reward/reward_u1": np.around(w_u1, 2), "reward/reward_v": np.around(w_v, 2)}

    def get_force_u1(self, pos):
        """
        地图障碍合力方向
        """
        pos[0] = np.clip(pos[0], 0, self.occ_map.shape[0] - 1)
        pos[1] = np.clip(pos[1], 0, self.occ_map.shape[1] - 1)
        fx = self.force_u1_x[pos[0], pos[1]]
        fy = self.force_u1_y[pos[0], pos[1]]
        f = np.array([fy, fx])

        return f

    def get_force_u2(self):
        """
        动态障碍物合力方向
        """
        return

    def get_force_v(self, robot_index, pos):
        """
        价值合力方向（测地距离）
        """
        pos[0] = np.clip(pos[0], 0, self.occ_map.shape[0] - 1)
        pos[1] = np.clip(pos[1], 0, self.occ_map.shape[1] - 1)
        force_vx = self.force_vxs[robot_index]
        force_vy = self.force_vys[robot_index]
        fx = force_vx[pos[0], pos[1]]
        fy = force_vy[pos[0], pos[1]]
        f = np.array([fy, fx])
        return f

    def compute_reach_goal_reward(self, reach_goal):
        reach_goal_reward = 0
        if reach_goal:
            reach_goal_reward = self.reward_config["reach_goal"]
        return reach_goal_reward

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

            relative_pose = cvt_positions_to_reference([self.agent_goals[i]], rt.get_position(), rt.get_yaw())

            geodesic_distance = self.compute_geodesic_distance(robot_index=0,
                                                               cur_position=self.agent_robots[0].get_position())
            if not self.args.train:
                print("input geodesic_distance:{}".format(geodesic_distance))
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

            relative_pose = cvt_positions_to_reference([self.agent_goals[i]], rt.get_position(), rt.get_yaw())
            w = self.input_config["image_w"]
            h = self.input_config["image_h"]
            image = cv2.resize(depth_image, (w, h))
            image[np.isnan(image)] = 1
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
            relative_pose = cvt_positions_to_reference([self.agent_goals[i]], rt.get_position(), rt.get_yaw())
            state = np.concatenate([hit_fractions, relative_pose.flatten()], axis=0).flatten()
            res.append(state.astype(float))

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

                reach_goal = compute_distance(robot.get_position(), self.agent_goals[i]) < self.running_config[
                    "goal_reached_thresh"]
                robot.small_step_xy_yaw_control(d_x, d_y, d_yaw)

                if reach_goal:
                    self.reach_goals[i] = True

                # 画机器人朝向线条
                if self.render:
                    robot_direction_id = plot_robot_direction_line(self.p, self.robot_direction_ids[i],
                                                                   robot.get_x_y_yaw())
                    self.robot_direction_ids[i] = robot_direction_id

            if self.npc_group is not None:
                self.npc_group.step()

            # 物理模拟一步
            self.p_step_simulation()

            iterate_count += 1
        # 检测碰撞
        collision = self._check_collision()
        # check if all reach goal
        all_reach_goal = all(self.reach_goals)
        return all_reach_goal, collision

    def iterate_steps_xy_control(self, actions):
        iterate_count = 0
        n_step = np.round(self.inference_duration / self.physical_step_duration)
        while iterate_count < n_step:
            for i, robot in enumerate(self.agent_robots):
                delta_x, delta_y = self.action_space.to_force(action=actions[i])
                # 机器人n_step步将delta_x, delta_y, delta_yaw走完
                d_x, d_y = delta_x / n_step, delta_y / n_step

                d_x, d_y = transform_local_to_world(np.array([d_x, d_y]), robot.get_position(),
                                                    robot.get_yaw()) - robot.get_position()

                reach_goal = compute_distance(robot.get_position(), self.agent_goals[i]) < self.running_config[
                    "goal_reached_thresh"]
                robot.small_step_xy_control(d_x, d_y)

                if reach_goal:
                    self.reach_goals[i] = True

                # 画机器人朝向线条
                if self.render:
                    robot_direction_id = plot_robot_direction_line(self.p, self.robot_direction_ids[i],
                                                                   robot.get_x_y_yaw())
                    self.robot_direction_ids[i] = robot_direction_id

            if self.npc_group is not None:
                self.npc_group.step()

            # 物理模拟一步
            self.p_step_simulation()

            iterate_count += 1
        # 检测碰撞
        collision = self._check_collision()
        # check if all reach goal
        all_reach_goal = all(self.reach_goals)
        return all_reach_goal, collision

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
                    # if not self.args.train:
                    #     print("robot {} not reached goal".format(i))

                if self.render:
                    robot_direction_id = plot_robot_direction_line(self.p, self.robot_direction_ids[i],
                                                                   robot.get_x_y_yaw())
                    self.robot_direction_ids[i] = robot_direction_id

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
                               self.agent_robot_config, self.sensors_name, self.args.sensors_config,
                               self.agent_starts[i],
                               2 * np.pi * np.random.random())

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
        self.last_distance = None
        self.last_geodesic_distance = None
        self.wall_ids = []
        self.collision_count = 0

        self.ma_images_deque = [deque(maxlen=self.image_seq_len) for i in range(self.num_agents)]
        self.ma_relative_poses_deque = [deque(maxlen=self.pose_seq_len) for i in range(self.num_agents)]
        self.agent_robots = None
        self.agent_robot_ids = []
        self.agent_starts, self.agent_goals = [None] * 2
        self.agent_sub_goals = None
        self.reach_goals = [False for i in range(self.num_agents)]

    def logging_action(self, action):
        logging_str = ""
        for key, item in zip(self.action_space_keys, action):
            logging_str += "{}: {}; ".format(key[:-1], item)
        logging.warning(logging_str)


class Phase:
    TRAIN = "train"
    TEST = "test"
