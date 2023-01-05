#!/usr/bin/env python3
import rospy
import gym
import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

from agents.action_space.high_level_action_space import AbstractHighLevelActionSpace
from agents.mapping import Env
from environment.nav_utilities.counter import Counter
from environment.nav_utilities.state_helper import StateHelper
from environment.path_manager import PathManager
from environment.robots.ros_robot import RosRobot
from environment.ros_debugger import RosDebugger
from traditional_planner.a_star.astar import AStar
from utils.image_utility import dilate_image


class Status(Enum):
    Collision = 0
    ReachGoal = 1
    Timeout = 2
    Other = 3


class EnvironmentROS(gym.Env):
    def __init__(self, args, action_space):
        rospy.init_node("ros_env_xy")
        self.args = args
        self.env_config = args.env_config
        self.action_space: AbstractHighLevelActionSpace = action_space
        self.target_pose = None
        self.path_manager = PathManager(args=self.args)
        self.state_helper = None

        self.rate = rospy.Rate(2)
        self.robot = RosRobot(args)
        self.ros_debugger = RosDebugger()
        # 订阅地图主题
        rospy.Subscriber("/map", OccupancyGrid, callback=self.map_callback)
        # 订阅nav goal主题，获取终点
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, callback=self.nav_goal_callback)

        self.map_origin = None
        self.occupancy_map = None
        self.map_width, self.map_height = None, None
        self.grid_resolution = None
        self.goal_position = None

        self.episode_count = Counter()
        self.step_count = Counter()

    def map_callback(self, msg: OccupancyGrid):
        origin = msg.info.origin.position
        self.map_origin = np.array([origin.x, origin.y])
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.grid_resolution = msg.info.resolution
        occupancy_map = np.array(msg.data).reshape((self.map_height, self.map_width)).T
        occupancy_map[occupancy_map <= 50] = 0
        occupancy_map[occupancy_map > 50] = 1
        occupancy_map = dilate_image(occupancy_map.astype("uint8"))
        self.occupancy_map = occupancy_map
        # plt.imshow(self.occupancy_map)
        # plt.show()

    def nav_goal_callback(self, msg: PoseStamped):
        pos = msg.pose.position
        self.goal_position = np.array([pos.x, pos.y])

    def world_to_occu_coord(self, position: np.ndarray):
        pos = (position - self.map_origin) / self.grid_resolution
        return pos.astype(np.int32)

    def occu_to_world_coord(self, position: np.ndarray):
        pos = position * self.grid_resolution + self.map_origin
        return pos

    def update_target_pose(self, target_pose: np.ndarray):
        """
        Parameters
        ----------
        target_pose: [x,y,yaw]
        """
        self.target_pose = target_pose

    def plan_path(self, start_world_pos, end_world_pos):
        print("111")
        start_occu_pos = self.world_to_occu_coord(start_world_pos)
        end_occu_pos = self.world_to_occu_coord(end_world_pos)
        print("222")
        print("起止点-world：{} - {}".format(start_world_pos, end_world_pos))
        print("起止点-occu：{} - {}".format(start_occu_pos, end_occu_pos))

        global_planner = AStar(self.occupancy_map)
        occu_path = global_planner.search_path(start_occu_pos, end_occu_pos)
        if occu_path is None or len(occu_path) == 0:
            self.robot.stop()
            return
        occu_path = np.array(occu_path)
        print("333")
        # self.plot_occu_path(occu_path)
        env_path = np.array([self.occu_to_world_coord(p) for p in occu_path])
        self.path_manager.register_path(env_path)
        self.ros_debugger.draw_lines2(self.path_manager.original_path)

    def plot_occu_path(self, occu_path):
        plt.imshow(self.occupancy_map)
        plt.plot(occu_path[:, 0], occu_path[:, 1])
        plt.show()

    def plan_from_current_to_goal(self):
        self.plan_path(self.robot.get_x_y(), self.goal_position)

    def reset(self, *args):
        self.clear_variables()

        print("初始化开始")
        while True:
            if not rospy.is_shutdown() and self.map_origin is not None and self.goal_position is not None:
                print("接收到final goal，开始规划")
                # 获取机器人的初始坐标，终点，规划A*路径
                self.state_helper = StateHelper(self.args, self.step_count)
                self.plan_from_current_to_goal()
                state = self.state_helper.get_next_state(self.path_manager, self.robot.cartesian_coordinates_list,
                                                         self.robot.polar_positions_list, self.robot.hit_vector_list,
                                                         self.robot.get_x_y(), self.robot.get_yaw(),
                                                         self.robot.visible_zone_limit)
                return state
                break
            self.rate.sleep()
        print("初始化结束")

    def clear_variables(self):
        self.step_count = Counter()

    def step(self, action: np.ndarray):
        self.step_count += 1
        print("action:{}".format(action))
        action = self.action_space.to_force(action=action)
        success, collision = self.iterate_steps(*action)

        over_max_step = self.step_count >= 1000000000

        state = self.state_helper.get_next_state(self.path_manager, self.robot.cartesian_coordinates_list,
                                                 self.robot.polar_positions_list, self.robot.hit_vector_list,
                                                 self.robot.get_x_y(), self.robot.get_yaw(),
                                                 self.robot.visible_zone_limit)
        # compute reward
        reward, info_for_sum = self.state_helper.compute_reward(self.robot.get_position()[:2], self.path_manager,
                                                                success, collision,
                                                                over_max_step)
        if self.args.env == Env.Real:
            done = success or over_max_step
        else:
            done = collision or success or over_max_step

        info_for_last = {"collision": collision, "a_success": success,
                         "over_max_step": over_max_step, "step_count": self.step_count.value}
        print(info_for_last)
        return state, reward, done, info_for_sum, info_for_last

    def iterate_steps(self, planned_v, planned_w):
        reach_goal, collision = False, False
        self.robot.apply_action(planned_v, planned_w)
        position = self.robot.get_x_y()
        self.path_manager.update_nearest_waypoint(position)
        reach_goal = self.path_manager.check_reach_goal(position)
        if reach_goal:
            self.robot.stop()
            self.goal_position = None

        # 讲这些算好，
        return reach_goal, collision

    def _compute_reward_and_status(self, current_pose: np.ndarray):
        dist_to_target = np.linalg.norm(current_pose[:2] - self.target_pose[:2])

        reward = 0
        # penalty on dis_to_target
        reward += self.env_config["low_reward"]["dist_to_target"] * dist_to_target  # / self.max_dist_to_target

        # TODO 碰撞检测
        if self.robot.is_collide():
            done = True
            info = {"status": Status.Collision}
            self.robot.stop()
        elif dist_to_target < 0.1:
            reward += self.env_config["low_reward"]["reach_target"]
            done = True
            info = {"status": Status.ReachGoal}
            # stop the robot
            self.robot.stop()
        else:
            done = False
            info = {"status": Status.Other}
        return reward, done, info

    def render(self, mode="human"):
        pass


if __name__ == "__main__":
    rospy.init_node("ros_env")
    env = EnvironmentROS(None, None)
    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        if env.map_origin is not None and env.goal_position is not None:
            env.reset()
            occu_corrd = env.world_to_occu_coord(np.array([0, 0]))
            print(occu_corrd)
            print(env.occu_to_world_coord(occu_corrd))
        rate.sleep()
