import logging
from typing import List

import numpy as np
import logging as logger

from pybullet_utils.bullet_client import BulletClient

from environment.nav_utilities.counter import Counter
from environment.path_manager import PathManager
from environment.robots.base_robot import BaseRobot
from environment.robots.robot_roles import RobotRoles
from environment.robots.robot_types import init_robot
from utils.math_helper import swap_value, compute_yaw, compute_distance


class Npc:
    def __init__(self, p: BulletClient, client_id: int, args, step_duration, path):
        # bullet_client, occupancy map, grid resolution
        self.p = p
        self.client_id = client_id
        self.args = args
        self.running_config = args.running_config
        self.grid_res = args.running_config["grid_res"]
        self.physical_step_duration = step_duration

        self.npc_robot_name = self.running_config["npc_robot_name"]
        self.npc_robot_config = args.robots_config[self.running_config["npc_robot_name"]]
        self.sensor_config = args.sensors_config[self.running_config["sensor_name"]]

        self.inference_every_duration = self.running_config["inference_duration"]

        self.speed_range = self.args.running_config["npc_speed_range"]
        self.lfd = 0.4
        self.speed = np.random.random() * (self.speed_range[1] - self.speed_range[0]) + self.speed_range[0]
        self.step_count = Counter()

        self.robot: BaseRobot = None

        # create obstacle entity
        self.start_position = None
        self.end_position = None
        # plan a path
        self.path_manager = PathManager(self.args)
        self.create(path)

    def create(self, path):
        logger.debug("create a dynamic obstacle")
        yaw = compute_yaw(path[0], path[-1])
        self.robot = init_robot(self.p, self.client_id, self.npc_robot_name, RobotRoles.NPC, self.physical_step_duration,
                                self.npc_robot_config,
                                self.sensor_config, path[0], yaw)

        self.path_manager.register_path(path)
        self.path_manager.update_nearest_waypoint(self.robot.get_position())
        # set speed
        logging.info("pedestrian id:{}; pedestrian speed:{}".format(self.robot.robot_id, self.speed))

    def small_step(self):
        """
        if reach, turn
        :return:
        """
        self.step_count += 1

        # reset the obstacle location
        cur_position = self.robot.get_position()

        self.path_manager.update_nearest_waypoint(cur_position)

        if self.path_manager.check_reach_goal(cur_position):
            self.path_manager.reverse()
            self.path_manager.update_nearest_waypoint(cur_position)

        target_wp_position = self.path_manager.get_waypoint_by_distance(self.lfd)
        # compute delta_y, delta_x
        direction = target_wp_position - cur_position
        theta = np.arctan2(direction[1], direction[0]) - self.robot.get_yaw()
        # compute the next position where the obstacle should be set
        planned_w = theta / self.physical_step_duration
        planned_v = self.speed
        return self.robot.small_step(planned_v, planned_w)


class NpcGroup:
    def __init__(self, p, client_id, args, step_duration, paths):
        super().__init__()
        self.p = p
        self.client_id = client_id
        self.args = args
        self.step_duration = step_duration
        self.npc_robots: List[Npc] = []
        self.npc_robot_ids: List[int] = []

        self.create(paths)

    def create(self, paths):
        # p, _human_control, grid_res, high_env_config, step_duration)
        i = 0
        for path in paths:
            i += 1
            npc_robot = Npc(self.p, self.client_id, self.args, self.step_duration, path)
            self.npc_robots.append(npc_robot)
            self.npc_robot_ids.append(npc_robot.robot.robot_id)
            logger.debug("The {}-th obstacle with obstacle id : {} created!".format(i, npc_robot.robot.robot_id))
        return self

    def step(self):
        for npc in self.npc_robots:
            npc.small_step()

    def clear(self):
        self.npc_robots: List[Npc] = []
        self.npc_robot_ids: List[int] = []
