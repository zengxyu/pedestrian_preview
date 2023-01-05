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
import uuid
from collections import deque

from matplotlib import pyplot as plt

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "traditional_planner", "a_star"))
print(sys.path)
import time
import numpy as np
from agents.action_space.high_level_action_space import AbstractHighLevelActionSpace
from environment.base_pybullet_env import PybulletBaseEnv
from environment.nav_utilities.check_helper import check_collision
from environment.nav_utilities.counter import Counter
from environment.nav_utilities.evaluation_helper import EvaluationHelper
from environment.nav_utilities.state_helper import StateHelper
from environment.robots.dynamic_obstacle import DynamicObstacleGroup, DynamicObstacle
from environment.robots.obstacle_collections import ObstacleCollections
from environment.robots.static_obstacle import StaticObstacleGroup, StaticObstacle
from environment.robots.turtlebot import TurtleBot
from environment.nav_utilities.pybullet_debug_gui_helper import plot_trajectory, get_random_color, \
    plot_robot_direction_line, plot_mark_spot

from utils.config_utility import read_yaml
from utils.image_utility import dilate_image
from utils.math_helper import compute_yaw
from environment.nav_utilities.scene_loader import load_environment_scene
from environment.nav_utilities.coordinates_converter import cvt_to_bu, cvt_to_om, cvt_polar_to_cartesian
from environment.path_manager import PathManager
from visualize_utilities.plot_action_space import plot_action_distribution
from visualize_utilities.plot_env import plot_a_star_path, plot_occupancy_map, plot_robot_trajectory
from visualize_utilities.plot_speed_density_curve import plot_speed_distance_curve, plot_real_speed_curve
from traditional_planner.a_star.astar import AStar


class EnvironmentBullet(PybulletBaseEnv, EvaluationHelper):
    def __init__(self, args, action_space):
        PybulletBaseEnv.__init__(self, args)
        if not args.train:
            EvaluationHelper.__init__(self, args)
        self.args = args
        self.running_config = args.running_config
        self.robot_config = args.robot_config
        self.env_config = args.env_config
        self.world_config = args.world_config
        self.render = args.render

        self.grid_res = self.env_config["grid_res"]

        self.max_step = self.running_config["max_steps"]
        self.inference_every_duration = self.running_config["inference_per_duration"]
        self.lidar_scan_interval = self.robot_config["lidar_scan_interval"]
        self.action_space: AbstractHighLevelActionSpace = action_space
        self.global_planner = None
        self.path_manager = PathManager(self.args)

        self.turtle_bot: TurtleBot = None
        self.occ_map = None
        self.dilated_occ_map = None
        self.door_occ_map = None
        self.state_helper = None

        self.s_om_pose, self.s_bu_pose, self.g_om_pose, self.g_bu_pose = [None] * 4
        self.om_path, self.bu_path = [], []

        self.obstacle_collections: ObstacleCollections = ObstacleCollections(args)

        # n_radius, n_from_start, n_to_end [unit: pixel]
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
        self.robot_direction_id = None
        self.physical_steps = Counter()
        self.n_lidar_scan_step = np.round(self.lidar_scan_interval / self.physical_step_duration)

        self.get_action_space_keys()

        self.seq_len = 4
        self.hit_vector_list = deque(maxlen=self.seq_len)
        self.polar_positions_list = deque(maxlen=self.seq_len)
        self.cartesian_coordinates_list = deque(maxlen=self.seq_len)
        # self.ray_num = self.args.robot_config["ray_num"]
        self.visible_zone_limit = self.args.robot_config["visible_width"]

    def render(self, mode="human"):
        width, height, rgb_image, depth_image, seg_image = self.turtle_bot.sensor.get_obs()
        return width, height, rgb_image, depth_image, seg_image

    def reset(self):
        self.episode_count += 1
        self.physical_steps = Counter()
        logging.info("\n---------------------------------------reset--------------------------------------------")
        logging.info("Episode : {}".format(self.episode_count))
        #  reset 有三件事情，
        #  1. restart the environment,
        #  2. re plan from current to the goal,
        #  3. re initialize the state helper.
        self.restart_environment()
        # self.init_states()

        self.turtle_bot = self.initialize_turtle_bot()

        # get the next observation and future waypoint
        self.plot_mark_plot(self.path_manager.original_path[self.path_manager.ideal_ind], color=[0, 1, 0])

        if not self.args.train:
            self.refresh_evaluation_episodes()
            self.history_occupancy_maps.append(self.occ_map)

        if self.render:
            # time.sleep(self.physical_step_duration)
            self.robot_direction_id = plot_robot_direction_line(self.p, self.robot_direction_id,
                                                                self.turtle_bot.get_x_y_yaw())
        return {}

    def get_action_space_keys(self):
        action_spaces_configs = read_yaml(self.args.action_space_config_folder, "action_space.yaml")
        action_class = self.action_space.__class__.__name__
        self.action_space_keys = list(action_spaces_configs[action_class].keys())

    def restart_environment(self):
        self.reset_simulation()
        self.clear_variables()

        # randomize environment
        self.randomize_env_and_plan_path()

        self.randomize_static_obstacles()
        self.randomize_dynamic_obstacles()

    def initialize_turtle_bot(self):
        # load turtle bot towards path direction
        start_yaw = compute_yaw(self.path_manager.get_waypoints(self.path_manager.original_path, 0),
                                self.path_manager.get_waypoints(self.path_manager.original_path, 2))
        return TurtleBot(self.p, self.client_id, self.physical_step_duration, self.args.robot_config, self.args,
                         self.s_bu_pose, start_yaw)

    def draw_attention_figure(self):

        def arrow(position, direction, arrow_length, c):
            start = position
            end = position + direction * arrow_length

            plt.plot([start[0], end[0]], [start[1], end[1]])

        def circle(position, radius=0.4, c='r'):
            theta = np.linspace(0, 2 * np.pi, 150)
            a = radius * np.cos(theta) + position[0]
            b = radius * np.sin(theta) + position[1]
            plt.plot(a, b, c=c)

        def draw_pedestrians():
            for obstacle in self.obstacle_collections.obstacles:
                if isinstance(obstacle, DynamicObstacle):
                    position = obstacle.get_cur_position()
                    position_occ = cvt_to_om(position, self.grid_res)
                    direction = obstacle.get_direction()
                    draw_entity(position_occ, direction, radius=2, arrow_length=4, color='g')
                elif isinstance(obstacle, StaticObstacle) and obstacle.type == "irrelevant":
                    position = obstacle.get_cur_position()
                    position_occ = cvt_to_om(position, self.grid_res)
                    draw_entity(position_occ, radius=2, arrow_length=4, color='gray')
                elif isinstance(obstacle, StaticObstacle) and obstacle.type == "static":
                    position = obstacle.get_cur_position()
                    position_occ = cvt_to_om(position, self.grid_res)
                    draw_entity(position_occ, radius=2, arrow_length=4, color='r')

        def draw_entity(position, direction=None, radius=5, arrow_length=10, color='r'):
            circle(position, radius, c=color)
            if direction is not None:
                arrow(position, direction, arrow_length=arrow_length, c=color)

        def draw_robot():
            position = self.turtle_bot.get_position()
            position_occ = cvt_to_om(position, self.grid_res)
            yaw = self.turtle_bot.get_yaw()
            direction = np.array([np.cos(yaw), np.sin(yaw)])
            draw_entity(position_occ, direction, radius=3, arrow_length=6, color='r')

        def draw_path():
            point_occ = cvt_to_om(self.path_manager.original_path, self.grid_res)
            plt.plot(point_occ[:, 0], point_occ[:, 1])

        def draw_lidar(coordinates, c="r"):
            start_coordinate = self.turtle_bot.get_position()
            theta = self.turtle_bot.get_yaw() - np.pi / 2
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
            end_coordinates = (R @ coordinates.T).T * self.visible_zone_limit + start_coordinate
            start_occ = cvt_to_om(start_coordinate, self.grid_res)
            end_occ_coordinates = cvt_to_om(end_coordinates, self.grid_res)

            for i, end_occ_coordinate in enumerate(end_occ_coordinates):
                plt.plot(np.array([start_occ[0], end_occ_coordinate[0]]),
                         np.array([start_occ[1], end_occ_coordinate[1]]), c=c, alpha=0.2)

        def draw_attention(coordinates, attention_scores, color):
            start_coordinate = self.turtle_bot.get_position()
            theta = self.turtle_bot.get_yaw() - np.pi / 2
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
            end_coordinates = (R @ coordinates.T).T * self.visible_zone_limit + start_coordinate
            start_occ = cvt_to_om(start_coordinate, self.grid_res)
            end_occ_coordinates = cvt_to_om(end_coordinates, self.grid_res)

            # attention_scores = attention_scores / (np.max(attention_scores) - np.min(attention_scores))
            attention_scores = attention_scores / max(attention_scores)
            skip = len(end_coordinates) / len(attention_scores)

            for i, end_occ_coordinate in enumerate(end_occ_coordinates):
                index = int(i // skip)
                plt.plot(np.array([start_occ[0], end_occ_coordinate[0]]),
                         np.array([start_occ[1], end_occ_coordinate[1]]), c=color, alpha=attention_scores[index][0])

        def draw_lidar_figure(color, save_folder, save_path):
            plt.figure(figsize=(5, 5))
            ax = plt.gca()
            plt.tight_layout()
            plt.axis('off')
            ax.set_xlim([0, self.occ_map.shape[0]])
            ax.set_ylim([0, self.occ_map.shape[1]])
            plt.imshow(np.rot90(np.flip(1 - self.occ_map, 1)), cmap='pink')
            draw_path()
            draw_pedestrians()
            draw_robot()
            draw_lidar(coordinates, color)
            # plt.show()
            plt.savefig(os.path.join(save_folder, save_path))
            plt.clf()

        def draw_attention_figure(scores, color, save_folder, save_path):
            plt.figure(figsize=(5, 5))
            ax = plt.gca()
            plt.tight_layout()
            plt.axis('off')
            ax.set_xlim([0, self.occ_map.shape[0]])
            ax.set_ylim([0, self.occ_map.shape[1]])
            plt.imshow(np.rot90(np.flip(1 - self.occ_map, 1)), cmap='pink')
            draw_path()
            draw_pedestrians()
            draw_robot()
            draw_attention(coordinates, scores, color=color)
            # plt.show()
            plt.savefig(os.path.join(save_folder, save_path))
            plt.clf()

        scores_spacial = self.args.actor_network.scores_spacial.detach().cpu().numpy()[0]
        scores_temporal = self.args.actor_network.scores_temporal.detach().cpu().numpy()[0]
        coordinates = self.args.actor_network.coordinates.detach().cpu().numpy()[0]

        save_folder = os.path.join("output", "attention_figure")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        random_path = uuid.uuid1()
        draw_lidar_figure(color='g', save_folder=save_folder, save_path="{}_lidar.png".format(random_path))
        draw_attention_figure(scores_spacial, color='b', save_folder=save_folder,
                              save_path="{}_attention_spacial.png".format(random_path))
        draw_attention_figure(scores_temporal, color='r', save_folder=save_folder,
                              save_path="{}_attention_temporal.png".format(random_path))

        print()

    def step(self, action):
        self.step_count += 1
        # self.draw_attention_figure()

        action = self.action_space.to_force(action=action)
        action = np.array([0.1, 0.1])
        success, collision = self.iterate_steps(*action)

        over_max_step = self.step_count >= self.max_step

        # get next state and reward
        width, height, rgb_image, depth_image, seg_image = self.turtle_bot.sensor.get_obs()

        # whether done
        done = collision or success or over_max_step

        # store information
        info_for_last = {"collision": collision, "a_success": success,
                         "over_max_step": over_max_step, "step_count": self.step_count.value}
        # logging.warning("target_speed:{}; robot : {}; speed : {}".format(action[0], self.turtle_bot.robot_id,
        #                                                                  self.turtle_bot.get_velocity()))
        # logging.debug("info_for_sum:{}".format(info_for_sum))
        # logging.debug("info_for_last:{}".format(info_for_last))
        # logging.debug("\n")

        # print()
        if done:
            print("success:{}; collision:{}; over_max_step:{}".format(success, collision, over_max_step))

        if done and self.render:
            self.add_episode_end_prompt(info_for_last)

        if done and not self.args.train and not self.args.resume:
            self.history_success.append(success)
            self.history_collision.append(collision)
            self.history_over_max_step.append(over_max_step)
            self.history_actions[-1].append(action)

        if done and self.args.plot_motion:
            plot_speed_distance_curve(self.motion_history_waypoint_distances, self.motion_history_real_speeds,
                                      y_label="real speed",
                                      parent_dir="plot_motion/speed_distance",
                                      save_name="real_speed_waypoint_distance_{}".format(self.episode_count.value))
            plot_speed_distance_curve(self.motion_history_waypoint_distances, self.motion_history_planned_speeds,
                                      y_label="planned speed",
                                      parent_dir="plot_motion/speed_distance",
                                      save_name="planned_speed_waypoint_distance_{}".format(self.episode_count.value))
            plot_real_speed_curve(self.motion_history_real_speeds[-1],
                                  parent_dir="plot_motion/real_speed_time_step",
                                  save_name="real_speed_time_step_{}".format(self.episode_count.value))
        if done and self.args.plot_action and self.episode_count.value % 10 == 0:
            plot_action_distribution(self.history_actions,
                                     parent_dir="plot_action",
                                     save_name="action_{}".format(self.episode_count.value))

        if done and self.args.plot_trajectory:
            plot_robot_trajectory(self.occ_map, self.om_path[0], self.om_path[-1],
                                  cvt_to_om(self.path_manager.original_path, self.grid_res),
                                  cvt_to_om(self.path_manager.deformed_path, self.grid_res),
                                  cvt_to_om(self.history_robot_positions[-1], self.grid_res),
                                  self.history_robot_velocities[-1],
                                  self.history_pedestrians_positions[-1],
                                  self.grid_res,
                                  parent_dir="plot_robot_trajectory",
                                  save_name="robot_trajectory_{}".format(self.episode_count.value)
                                  )
            # plot stored information
        return {}, {}, done, {}, info_for_last

    def plot_mark_plot(self, sample_position, color):
        if self.render:
            plot_mark_spot(self.p,
                           spot_position=[sample_position[0], sample_position[1]],
                           size=0.05, color=color)

    def p_step_simulation(self):
        self.p.stepSimulation()
        self.physical_steps += 1
        if not self.args.train:
            self.history_robot_positions[-1].append(self.turtle_bot.get_position())
            self.history_robot_yaws[-1].append(self.turtle_bot.get_yaw())
            self.history_robot_velocities[-1].append(self.turtle_bot.get_velocity())
            self.history_step_simulation_counts[-1] += 1
            # 将行人的位置也每一次都保存下来
            self.history_pedestrians_positions[-1].append(self.obstacle_collections.get_positions())

    def _check_collision(self):
        return check_collision(self.p, [self.turtle_bot.robot_id], self.obstacle_ids)

    def visualize_polar(self, thetas, dists, title="Plot lidar polar positions", c='r'):
        plt.polar(thetas, dists, 'ro', lw=2, c=c)
        plt.title(title)
        plt.ylim(0, 1)
        plt.show()

    def visualize_(self, xs, ys, title="Plot lidar cartesian positions", c='r'):
        plt.scatter(xs, ys, lw=2, c=c)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        plt.clf()

    def collect_vision_obs(self, turtle_bot: TurtleBot):
        width, height, rgb_image, depth_image, seg_image = turtle_bot.sensor.get_obs()

        return rgb_image, depth_image, seg_image

    def collect_observation(self, turtle_bot: TurtleBot):
        # lidar scan
        # hit_fractions, the first ray is in the robot direction
        _, hit_fractions = turtle_bot.sensor.get_obs()
        hit_thetas = np.linspace(0, 2 * np.pi, len(hit_fractions), endpoint=False)
        # compute polar positions
        polar_positions = np.array([hit_thetas, hit_fractions]).transpose()

        # hit vectors
        hit_vector = np.where(polar_positions[:, 1] < 1, 1, 0)

        # cartesian positions
        cartesian_positions = cvt_polar_to_cartesian(polar_positions)

        self.polar_positions_list.append(polar_positions)
        self.cartesian_coordinates_list.append(cartesian_positions)
        self.hit_vector_list.append(hit_vector)

    def time_to_scan(self):
        # lidar_scan_interval = 0.2, 这意味着每走四步 small_step, 进行一次扫描
        remainder = np.round(self.physical_steps.value % self.n_lidar_scan_step, 2)
        if abs(remainder) <= 1e-5:
            return True
        return False

    def iterate_steps(self, planned_v, planned_w):
        iterate_count = 0
        reach_goal, collision = False, False
        # 0.4/0.05 = 8
        n_step = np.round(self.inference_every_duration / self.physical_step_duration)

        while iterate_count < n_step and not reach_goal and not collision:
            self.turtle_bot.small_step(planned_v, planned_w)
            self.obstacle_collections.step()
            self.p_step_simulation()

            collision = self._check_collision()

            if self.render:
                # time.sleep(self.physical_step_duration)
                self.robot_direction_id = plot_robot_direction_line(self.p, self.robot_direction_id,
                                                                    self.turtle_bot.get_x_y_yaw())

            # 每隔一定的间隔 进行一次雷达扫描
            if self.time_to_scan():
                # self.collect_observation(self.turtle_bot)
                self.collect_vision_obs(self.turtle_bot)

            position = self.turtle_bot.get_position()
            self.path_manager.update_nearest_waypoint(position)
            reach_goal = self.path_manager.check_reach_goal(position)
            iterate_count += 1
        return reach_goal, collision

    def add_episode_end_prompt(self, info):
        if info["a_success"]:
            self.p.addUserDebugText("reach goal!", textPosition=[1.5, 1.5, 1.], textColorRGB=[0, 1, 0], textSize=5)
            time.sleep(1)
        elif info["collision"]:
            self.p.addUserDebugText("collision!", textPosition=[1.5, 1.5, 1.], textColorRGB=[1, 0, 0], textSize=5)
            time.sleep(1)
        else:
            self.p.addUserDebugText("Over the max step!", textPosition=[1.5, 1.5, 1.], textColorRGB=[0, 0, 1],
                                    textSize=5)
            time.sleep(1)

    def randomize_static_obstacles(self):
        if self.n_static_obstacle_num == 0:
            return

        count = 0
        obs_bu_positions = []
        start_index = self.n_from_start
        end_index = len(self.om_path) - self.n_to_end
        # sample static points from the path
        while len(obs_bu_positions) < self.n_static_obstacle_num and count < 50:
            obs_om_position, sample_success = self.static_obs_sampler(occupancy_map=self.dilated_occ_map,
                                                                      door_map=self.door_occ_map,
                                                                      sample_from_path=True,
                                                                      robot_om_path=self.om_path,
                                                                      margin=self.dilation_size + 1,
                                                                      radius=self.n_radius,
                                                                      start_index=start_index,
                                                                      end_index=end_index)

            count += 1
            if sample_success:
                obs_bu_position = cvt_to_bu(np.array(obs_om_position), self.grid_res)
                obs_bu_positions.append(obs_bu_position)

        # if no static point can be sampled from this environment, restart environment
        if len(obs_bu_positions) > 0:
            # create n dynamic obstacles, put them into the environment
            static_obstacle_group = StaticObstacleGroup(self.p, self.occ_map, self.grid_res).create(
                obs_bu_positions, type="static")

            self.obstacle_collections.add(static_obstacle_group, dynamic=False)

            self.obstacle_ids.extend(self.obstacle_collections.get_obstacle_ids())

        # ====================================================
        for i in range(self.env_config["num_extra_pedestrians"]):
            # sample static points from occupancy map, not limited on the path
            obs_om_position, sample_success = self.static_obs_sampler(occupancy_map=self.dilated_occ_map,
                                                                      door_map=self.door_occ_map,
                                                                      sample_from_path=False,
                                                                      robot_om_path=self.om_path,
                                                                      margin=self.dilation_size + 1,
                                                                      radius=self.n_radius,
                                                                      start_index=start_index,
                                                                      end_index=end_index)
            if sample_success:
                obs_bu_position = cvt_to_bu(np.array(obs_om_position), self.grid_res)
                obs_bu_positions.append(obs_bu_position)

        # if no static point can be sampled from this environment, restart environment
        if len(obs_bu_positions) > 0:
            # create n dynamic obstacles, put them into the environment
            static_obstacle_group = StaticObstacleGroup(self.p, self.occ_map, self.grid_res).create(
                obs_bu_positions, type="irrelevant")

            self.obstacle_collections.add(static_obstacle_group, dynamic=False)

            self.obstacle_ids.extend(self.obstacle_collections.get_obstacle_ids())

    def sample_segment_index(self, used_indexes, seg_start_indexes, seg_end_indexes):
        length_start = len(seg_start_indexes)
        length_end = len(seg_end_indexes)
        if len(used_indexes) == min(length_start, length_end):
            used_indexes = []
        index = np.random.randint(0, min(length_start, length_end))
        while index in used_indexes:
            index = np.random.randint(0, min(length_start, length_end))
        used_indexes.append(index)
        return index

    def randomize_dynamic_obstacles(self):
        if self.n_dynamic_obstacle_num == 0:
            return

        logging.debug(
            "randomize dynamic obstacles... length of original path:{}".format(len(self.path_manager.original_path)))

        # randomize the start and end position for occupancy map
        obs_bu_starts = []
        obs_bu_ends = []
        obs_bu_paths = []
        start_index = self.n_from_start
        end_index = len(self.om_path) - self.n_to_end
        equal_n = (end_index - start_index) / 4

        seg_start_indexes = np.arange(start_index, end_index - equal_n, equal_n)
        seg_end_indexes = np.arange(start_index + equal_n, end_index, equal_n)

        assert len(seg_start_indexes) == len(seg_end_indexes)
        used_indexes = []
        count = 0
        while count < 30 and len(obs_bu_starts) < self.n_dynamic_obstacle_num - 1:
            count += 1
            index = self.sample_segment_index(used_indexes, seg_start_indexes, seg_end_indexes)
            seg_start, seg_end = seg_start_indexes[index], seg_end_indexes[index]
            [obs_om_start, obs_om_end], sample_success = self.dynamic_obs_sampler(
                occupancy_map=dilate_image(self.occ_map, 2),
                door_map=self.door_occ_map,
                robot_om_path=self.om_path,
                margin=self.dilation_size + 1,
                radius=self.n_radius,
                start_index=seg_start,
                end_index=seg_end,
                kept_distance=self.n_kept_distance,
                kept_distance_to_start=self.n_kept_distance_to_start)

            # if sample obstacle start position and end position failed, continue to next loop and resample
            if not sample_success:
                continue

            # plan a global path for this (start, end) pair
            obs_om_path = AStar(dilate_image(self.occ_map, 2)).search_path(tuple(self.s_om_pose), tuple(self.g_om_pose))

            if len(obs_om_path) == 0:
                continue
            logging.debug("There are now {} sampled obstacles".format(len(obs_bu_starts)))
            obs_bu_path = cvt_to_bu(obs_om_path, self.grid_res)
            obs_bu_start = cvt_to_bu(obs_om_start, self.grid_res)
            obs_bu_end = cvt_to_bu(obs_om_end, self.grid_res)

            obs_bu_paths.append(obs_bu_path)
            obs_bu_starts.append(obs_bu_start)
            obs_bu_ends.append(obs_bu_end)

            if self.render:
                plot_trajectory(self.p, obs_bu_path, color=get_random_color())

            if len(obs_bu_starts) >= self.n_dynamic_obstacle_num - 1:
                break

        # 添加障碍物， 该障碍物从机器人路径倒着走回
        self.add_one_backwards(obs_bu_starts, obs_bu_ends, obs_bu_paths)

        if len(obs_bu_starts) == 0 and self.n_dynamic_obstacle_num > 0:
            return self.restart_environment()

        # create n dynamic obstacles, put them into the environment
        dynamic_obstacle_group = DynamicObstacleGroup(self.p, self.args, self.physical_step_duration).create(
            obs_bu_starts,
            obs_bu_ends,
            obs_bu_paths)

        self.obstacle_collections.add(dynamic_obstacle_group, dynamic=True)
        self.obstacle_ids.extend(self.obstacle_collections.get_obstacle_ids())

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

    def randomize_env_and_plan_path(self):
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
        no_planned_path_available = True
        building_name = ""
        while no_planned_path_available:
            logging.debug("Create the environment...")
            self.reset_simulation()
            self.clear_variables()

            # randomize building
            self.obstacle_ids, self.occ_map, self.dilated_occ_map, self.door_occ_map, samplers, building_name = load_environment_scene(
                p=self.p,
                high_env_config=self.env_config,
                world_config=self.world_config,
                grid_resolution=self.grid_res
            )
            # extract samplers
            self.start_goal_sampler, self.static_obs_sampler, self.dynamic_obs_sampler = samplers

            # sample start position and goal position
            [self.s_om_pose, self.g_om_pose], sample_success = self.start_goal_sampler(
                occupancy_map=self.dilated_occ_map, margin=self.dilation_size + 1)

            # plan a path
            # if no path planned available, resample start position and goal position
            if not sample_success:
                continue

            self.om_path = AStar(self.dilated_occ_map).search_path(tuple(self.s_om_pose), tuple(self.g_om_pose))
            no_planned_path_available = self.om_path is None or len(self.om_path) <= 10

        if self.args.plot_env:
            plot_occupancy_map(self.dilated_occ_map, self.om_path[0], self.om_path[-1],
                               save_name="dilated_om_" + building_name)
            plot_occupancy_map(self.occ_map, self.om_path[0], self.om_path[-1], save_name="om_" + building_name)
            plot_a_star_path(self.dilated_occ_map, self.om_path[0], self.om_path[-1], self.om_path,
                             parent_dir="plot_occupancy_map",
                             save_name="path_dilated_om_" + building_name + "_{}".format(self.episode_count.value))
            plot_a_star_path(self.occ_map, self.om_path[0], self.om_path[-1], self.om_path,
                             parent_dir="plot_occupancy_map",
                             save_name="path_om_" + building_name + "_{}".format(self.episode_count.value))

        # sample start pose and goal pose
        self.s_bu_pose = cvt_to_bu(self.s_om_pose, self.grid_res)
        self.g_bu_pose = cvt_to_bu(self.g_om_pose, self.grid_res)
        self.bu_path = cvt_to_bu(self.om_path, self.grid_res)

        logging.debug("Create the environment, Done...")
        self.path_manager.register_path(self.bu_path[:-5], self.occ_map)
        self.path_manager.update_nearest_waypoint(self.s_bu_pose)

        if self.render:
            plot_trajectory(self.p, self.path_manager.original_path, color=[1, 0, 0])

    def clear_variables(self):
        self.step_count = Counter()
        self.turtle_bot = None
        self.occ_map = None
        self.dilated_occ_map = None
        self.door_occ_map = None
        self.s_om_pose, self.s_bu_pose, self.g_om_pose, self.g_bu_pose = [None] * 4
        self.om_path, self.bu_path = [], []

        self.collection_robot_positions_occ = []
        self.collection_robot_vs = []
        self.collection_robot_ws = []
        self.collection_robot_planned_vs = []
        self.collection_robot_planned_ws = []

        self.obstacle_collections.clear()
        self.obstacle_ids = []

    def logging_action(self, action):
        logging_str = ""
        for key, item in zip(self.action_space_keys, action):
            logging_str += "{}: {}; ".format(key[:-1], item)
        logging.warning(logging_str)
