from typing import Dict

import numpy as np
from pybullet_utils.bullet_client import BulletClient

from environment.gen_scene.build_office_world import create_cylinder
from environment.nav_utilities.pybullet_helper import place_object
from environment.robots.base_robot import BaseRobot


class ObjectRobot(BaseRobot):
    def __init__(self, p: BulletClient, client_id: int, step_duration: float, robot_config: Dict, sensor_config: Dict,
                 start_position, start_yaw):
        super().__init__(p, client_id)

        self.p = p
        self.client_id = client_id
        self.robot_config = robot_config
        self.sensor_config = sensor_config

        self.physical_step_duration = step_duration

        self.robot_id = None
        self.shape = self.robot_config["shape"]
        self.radius_range = self.robot_config["radius_range"]
        self.height_range = self.robot_config["height_range"]

        self.radius = np.random.random() * (self.radius_range[1] - self.radius_range[0]) + self.radius_range[0]
        self.height = np.random.random() * (self.height_range[1] - self.height_range[0]) + self.height_range[0]

        self.load_object(start_position[0], start_position[1], start_yaw)
        self.cur_yaw = start_yaw
        self.cur_v = 0
        self.cur_w = 0

    def small_step(self, planned_v, planned_w):
        """
        assume that the robot can reach the (planned_v, planned_w) immediately
        """
        cur_position = self.get_position()
        cur_theta = self.get_yaw()
        next_yaw = cur_theta + planned_w * self.physical_step_duration

        delta_y = planned_v * np.sin(next_yaw) * self.physical_step_duration
        delta_x = planned_v * np.cos(next_yaw) * self.physical_step_duration

        # compute the next position where the obstacle should be set
        next_position = [cur_position[0] + delta_x, cur_position[1] + delta_y]

        place_object(self.p, self.robot_id, *next_position)

        self.update_yaw(next_yaw)
        self.update_v_w(planned_v, planned_w)

        return planned_v, planned_w

    def update_yaw(self, yaw):
        self.cur_yaw = yaw

    def update_v_w(self, v, w):
        self.cur_v = v
        self.cur_w = w

    def get_position(self):
        cur_position, cur_orientation_quat = self.p.getBasePositionAndOrientation(self.robot_id)
        return np.array([cur_position[0], cur_position[1]])

    def get_yaw(self):
        return self.cur_yaw

    def get_v(self):
        return self.cur_v

    def get_w(self):
        return self.cur_w

    def load_object(self, cur_x, cur_y, cur_yaw=0):
        self.cur_yaw = cur_yaw
        start_position = np.array([cur_x, cur_y])
        self.robot_id = create_cylinder(self.p, start_position, height=self.height, radius=self.radius)
