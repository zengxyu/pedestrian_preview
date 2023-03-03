import os
from typing import Dict

import numpy as np
from pybullet_utils.bullet_client import BulletClient

from environment.gen_scene.build_office_world import create_cylinder
from environment.nav_utilities.pybullet_helper import place_object
from environment.robots.base_robot import BaseRobot
from environment.robots.robot_roles import get_role_color
from environment.sensors.sensor_types import init_sensors
from utils.fo_utility import get_project_path


class ObjectRobot(BaseRobot):
    def __init__(self, p: BulletClient, client_id: int, robot_role: str, step_duration: float, robot_config: Dict,
                 sensor_names: str, sensors_config: Dict, start_position, start_yaw):
        super().__init__(p, client_id)

        self.p = p
        self.client_id = client_id
        self.robot_role = robot_role
        self.robot_config = robot_config
        self.sensors_config = sensors_config

        self.physical_step_duration = step_duration

        self.robot_id = None
        self.color = None
        self.shape = self.robot_config["shape"]
        self.radius_range = self.robot_config["radius_range"]

        self.radius = np.random.random() * (self.radius_range[1] - self.radius_range[0]) + self.radius_range[0]
        self.height = 1.7

        self.with_collision = self.robot_config["with_collision"]
        # self.load_object(start_position[0], start_position[1], start_yaw)
        self.load_urdf(start_position[0], start_position[1], start_yaw)
        self.sensors = init_sensors(robot_id=self.robot_id, sensor_names=sensor_names,
                                    sensors_config=self.sensors_config)

        self.cur_v = 0
        self.cur_w = 0

    def small_step_pose_control(self, delta_x, delta_y, delta_yaw):
        cur_position = self.get_position()
        cur_yaw = self.get_yaw()
        (_, _, z), _ = self.p.getBasePositionAndOrientation(self.robot_id)
        target_position = np.array([*cur_position, z]) + np.array([delta_x, delta_y, 0])
        target_yaw = cur_yaw + delta_yaw
        target_orientation = np.array([0., 0., target_yaw])
        self.p.resetBasePositionAndOrientation(
            physicsClientId=self.client_id,
            bodyUniqueId=self.robot_id,
            posObj=target_position,
            ornObj=self.p.getQuaternionFromEuler(target_orientation)
        )

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

        place_object(self.p, self.robot_id, next_position[0], next_position[1], next_yaw)

        self.update_v_w(planned_v, planned_w, cur_position, next_position)

        return planned_v, planned_w

    def update_v_w(self, v, w, cur_position, next_position):
        self.cur_v = np.linalg.norm(cur_position - next_position) / self.physical_step_duration
        self.cur_w = w

    def get_v(self):
        return self.cur_v

    def get_w(self):
        return self.cur_w

    def load_object(self, cur_x, cur_y, cur_yaw=0):
        start_position = np.array([cur_x, cur_y])
        self.color = get_role_color(self.robot_role)
        self.robot_id = create_cylinder(self.p, start_position, with_collision=self.with_collision, height=self.height,
                                        radius=self.radius, color=self.color)

    def load_urdf(self, cur_x, cur_y, cur_yaw=0):
        """
        load turtle bot from urdf, turtlebot urdf is from rl_utils
        :return:
        """
        race_car_path = os.path.join(
            get_project_path(),
            "environment",
            "robots",
            "urdf",
            "object_robot.urdf",
        )
        robot_id = self.p.loadURDF(
            race_car_path,
            [cur_x, cur_y, self.height / 2],
            self.p.getQuaternionFromEuler([0, 0, cur_yaw]),
            flags=self.p.URDF_USE_INERTIA_FROM_FILE
                  | self.p.URDF_USE_IMPLICIT_CYLINDER,
        )

        base_link = -1
        # Find relevant joints
        for j in range(self.p.getNumJoints(robot_id)):
            if "base_link" in str(self.p.getJointInfo(robot_id, j)[1]):
                base_link = j

        self.color = get_role_color(self.robot_role)
        self.p.changeVisualShape(robot_id, base_link, rgbaColor=self.color)

        self.robot_id = robot_id
