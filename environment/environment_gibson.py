#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : navigation_icra 
    @Author  : Xiangyu Zeng
    @Date    : 10/28/22 2:49 PM 
    @Description    :
        
===========================================
"""
from collections import deque

import numpy as np
from igibson.render.viewer import Viewer
from matplotlib import pyplot as plt

from agents.action_space.high_level_action_space import AbstractHighLevelActionSpace
from environment.base_gibson_env import GibsonBaseEnv
from environment.gibson.IgibsonConfig import IgibsonConfig
from environment.nav_utilities.coordinates_converter import cvt_polar_to_cartesian
from environment.nav_utilities.counter import Counter
from environment.nav_utilities.pybullet_debug_gui_helper import plot_bullet_path, plot_trajectory
from environment.nav_utilities.state_helper import StateHelper
from environment.path_manager import PathManager
import pybullet as p

from traditional_planner.a_star.astar import AStar
from utils.math_helper import compute_yaw
import pybullet as p_


class EnvironmentGibson(GibsonBaseEnv):
    def __init__(self, args, action_space):
        config_igibson = IgibsonConfig.get_config()
        mode = "gui_interactive" if args.render else "headless"
        super(GibsonBaseEnv, self).__init__(config_file=config_igibson.yaml_data,
                                            mode=mode,
                                            action_timestep=config_igibson["action_timestep"],
                                            physics_timestep=config_igibson["physics_timestep"],
                                            use_pb_gui=config_igibson["pb"])
        # if not args.train:
        #     EvaluationHelper.__init__(self, args)
        self.action_space: AbstractHighLevelActionSpace = action_space

        self.args = args
        self.path_manager = PathManager(self.args)
        self.step_count = Counter()

        self.inference_per_duration = self.args.running_config["inference_per_duration"]
        step_duration = self.args.env_config["step_duration"]
        self.physical_step_duration = step_duration
        self.lidar_scan_interval: float = self.args.robot_config["lidar_scan_interval"]
        self.n_lidar_scan_step = np.round(self.lidar_scan_interval / step_duration)

        self.visible_zone_limit = self.args.robot_config["visible_width"]
        self.seq_len = 4
        self.hit_vector_list = deque(maxlen=self.seq_len)
        self.polar_positions_list = deque(maxlen=self.seq_len)
        self.cartesian_coordinates_list = deque(maxlen=self.seq_len)
        self.ray_num = self.args.robot_config["ray_num"]
        self.visible_zone_limit = self.args.robot_config["visible_width"]

        self.robot = None
        self.view_offset = np.pi / 2
        self.v_ctrl_factor: float = self.args.robot_config["v_ctrl_factor"]
        self.w_ctrl_factor: float = self.args.robot_config["w_ctrl_factor"]
        self.wheel_base = 0.23
        self.state_helper = None
        self.physical_steps = Counter()
        self.floor_num = 0

    def init_states(self):
        for i in range(self.seq_len):
            self.hit_vector_list.append(np.zeros((self.ray_num,)))
            self.polar_positions_list.append(np.zeros((self.ray_num, 2)))
            self.cartesian_coordinates_list.append(np.zeros((self.ray_num, 2)))

    def visualize_occupancy_map(self, occ_map):
        if self.args.visualize:
            ax = plt.gca()
            ax.imshow(occ_map, cmap="Greys")
            plt.show()

    def step(self, action):
        # 应用动作
        print("step")
        self.step_count += 1
        action = self.action_space.to_force(action=action)

        print("action:{}".format(action))
        # action = [0.2, 0]
        success, collision = self.iterate_steps(action)

        robot_position = self.robot.get_position()
        robot_yaw = self.get_yaw(self.robot.get_orientation())
        # 获得下一个状态
        state = self.state_helper.get_next_state(self.path_manager, self.cartesian_coordinates_list,
                                                 self.polar_positions_list,
                                                 self.hit_vector_list,
                                                 robot_position,
                                                 robot_yaw,
                                                 self.visible_zone_limit)
        # 计算奖励
        # compute reward
        over_max_step = self.step_count >= 1000000000

        reward, info_for_sum = self.state_helper.compute_reward(self.robots[0].get_position()[:2], self.path_manager,
                                                                self.cartesian_coordinates_list,
                                                                success, collision, over_max_step)
        done = collision or success or over_max_step
        # store information
        info_for_last = {"collision": collision, "a_success": success,
                         "over_max_step": over_max_step, "step_count": self.step_count.value}

        return state, reward, done, info_for_sum, info_for_last

    def time_to_scan(self):
        # lidar_scan_interval = 0.2, 这意味着每走四步 small_step, 进行一次扫描
        remainder = np.round(self.physical_steps.value % self.n_lidar_scan_step, 2)
        if abs(remainder) <= 1e-5:
            return True
        return False

    def convert_v_w_to_wheel_velocities(self, v, w):
        v = self.v_ctrl_factor * v * 2
        w = self.w_ctrl_factor * w
        vl = v - w * self.wheel_base / 2
        vr = v + w * self.wheel_base / 2
        return np.array([vl, vr])

    def p_step_simulation(self):
        collision_links = self.run_simulation()
        self.physical_steps += 1
        return collision_links

    def iterate_steps(self, action):
        iterate_count = 0

        reach_goal, collision = False, False

        n_step = np.round(self.inference_per_duration / self.physical_step_duration)

        while iterate_count < n_step and not reach_goal and not collision:
            # action = self.convert_v_w_to_wheel_velocities(*action)
            v, w = action
            v = v * 0.5
            w = - w * 3.14
            self.robot.apply_action((v, w))
            collision_links = self.p_step_simulation()
            self.task.step(self)
            # self.target_pos = self.robot.get_position()+

            print("robot position:{}".format(self.robot.get_position()))
            print("robot v:{}; robot w:{}".format(self.get_v(), self.get_w()))
            if self.time_to_scan():
                print("time to scan")
                state = self.get_state()
                self.collect_observation(state)
            position = self.robots[0].get_position()[:2]
            self.path_manager.update_nearest_waypoint(position)
            reach_goal = self.path_manager.check_reach_goal(position)
            iterate_count += 1

        # check collision
        # collision_links = self.run_simulation()
        # self.collision_links = collision_links
        # self.collision_step += int(len(collision_links) > 0)
        # collision = self.collision_step > 0
        collision = False
        return reach_goal, collision

    def get_velocity(self):
        v_transition, v_rotation = p.getBaseVelocity(self.robot.get_body_ids()[0])
        return v_transition, v_rotation

    def get_v(self):
        cur_v, cur_w = p.getBaseVelocity(self.robot.get_body_ids()[0])
        speed = np.linalg.norm(cur_v[:2])
        return speed

    def get_w(self):
        cur_v, cur_w = p.getBaseVelocity(self.robot.get_body_ids()[0])
        return cur_w[2]

    def reset_variables(self):
        super(EnvironmentGibson, self).reset_variables()
        self.physical_steps = Counter()
        self.init_states()
        self.state_helper = StateHelper(self.args, self.step_count)

    def reset(self):
        #  1. restart the environment,
        #  2. re plan from current to the goal,
        #  3. re initialize the state helper.
        print("reset..............")

        super(GibsonBaseEnv, self).reset()
        self.simulator.viewer = Viewer(initial_pos=[self.task.initial_pos[0], self.task.initial_pos[1], 2],
                                       initial_up=[0, 0, 0.5],
                                       initial_view_direction=[0.8, -0.1, 0.6],
                                       simulator=self.simulator,
                                       renderer=self.simulator.renderer)
        self.simulator.sync(force_sync=True)

        self.robot = self.robots[0]
        self.land(self.robot, pos=self.task.initial_pos, orn=self.task.start_euler)
        self.run_simulation()
        print("robot position:{}".format(self.robot.get_position()))

        self.path_manager.register_path(self.task.path_world)
        plot_trajectory(p_, self.task.path_world, color=[1, 0, 0])

        # 从机器人当前位置 到目标位置
        self.collect_observation(self.get_state())
        state = self.state_helper.get_next_state(self.path_manager, self.cartesian_coordinates_list,
                                                 self.polar_positions_list,
                                                 self.hit_vector_list,
                                                 self.robot.get_position()[:2],
                                                 self.get_yaw(self.robot.get_orientation()),
                                                 self.visible_zone_limit)
        return state

    def get_yaw(self, robot_orientation):
        # Euler angles in radians ( roll, pitch, yaw )
        euler_orientation = p.getEulerFromQuaternion(robot_orientation)

        yaw = euler_orientation[2]  # - self.default_orn_euler[2]
        return yaw

    def visualize(self, xs, ys, title, c):
        plt.scatter(xs, ys, lw=2, c=c)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        plt.clf()

    def visualize_polar(self, thetas, dists, title="Plot lidar polar positions", c='r'):
        plt.polar(thetas, dists, 'ro', lw=2, c=c)
        plt.title(title)
        plt.ylim(0, 1)
        plt.show()

    def collect_observation(self, state):

        ranges = np.array(state["scan"], dtype=np.float32).flatten()
        # 1440 -> 360
        ranges = ranges[::4]
        # 开头得是从机器人面对的方向开始
        ranges = np.roll(ranges, -int(len(ranges) / 4 * 3))
        # 获得角度
        hit_thetas = (np.linspace(0, 2 * np.pi, len(ranges), endpoint=False) + np.pi) % (2 * np.pi)
        # self.visualize_polar(hit_thetas[:200], ranges[:200])
        # self.visualize_polar(hit_thetas, ranges)

        hit_vector = np.where(ranges < 1, 1, 0)
        polar_positions = np.array([hit_thetas, ranges]).transpose()
        # relative cartesian coordinates
        cartesian_coordinates = cvt_polar_to_cartesian(polar_positions)

        # plot_gibson_lidar_ray(p, hit_vector, self.robot.get_position()[:2],
        #                       cartesian_coordinates + self.robot.get_position()[:2],
        #                       missRayColor=[0, 1, 0],
        #                       hitRayColor=[1, 0, 0])

        self.polar_positions_list.append(polar_positions)
        self.cartesian_coordinates_list.append(cartesian_coordinates)
        self.hit_vector_list.append(hit_vector)

    def render(self, mode="human"):
        super(EnvironmentGibson, self).render(mode)
