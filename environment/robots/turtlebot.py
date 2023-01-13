import math
import os
from typing import Dict

import numpy as np
from numpy import pi
from pybullet_utils.bullet_client import BulletClient

from environment.sensors.lidar_sensor import LidarSensor
from environment.robots.base_differential_robot import BaseDifferentialRobot
from environment.sensors.vision_sensor import VisionSensor
from utils.config_utility import read_yaml
from utils.fo_utility import get_project_path

np.set_printoptions(precision=3, suppress=True)


class TurtleBot(BaseDifferentialRobot):
    def __init__(self, p: BulletClient, client_id: int, step_duration: float, robot_config: Dict, args,
                 start_position, start_yaw):
        super().__init__(p, client_id)
        self.lidar_joint_id = None
        self.robot_config = robot_config
        self.wheel_base = 0.23
        # self.robot_id, self.left_wheel_id, self.right_wheel_id initialized
        # 1.5, 2.3
        # 原本的.75, .75
        self.load_turtle_bot(start_position[0], start_position[1], start_yaw)
        self.v_ctrl_factor: float = self.robot_config["v_ctrl_factor"]
        self.w_ctrl_factor: float = self.robot_config["w_ctrl_factor"]

        self.joint_speed = 5

        self.visible_zone_limit = self.robot_config["visible_width"]

        self.sensor_config = args.sensors_config[self.robot_config["sensor"]]

        self.sensor = VisionSensor(robot_id=self.robot_id, sensor_config=self.sensor_config)

        self.physical_step_duration = step_duration

        self.lidar_scan_interval: float = self.robot_config["lidar_scan_interval"]

        self.n_lidar_scan_step = np.round(self.lidar_scan_interval / step_duration)

    def convert_v_w_to_wheel_velocities(self, v, w):
        v = self.v_ctrl_factor * v * 2
        w = self.w_ctrl_factor * w
        vl = v - w * self.wheel_base / 2
        vr = v + w * self.wheel_base / 2
        return np.array([vl, vr])

    def small_step(self, planned_v, planned_w):
        left_v, right_v = self.convert_v_w_to_wheel_velocities(planned_v, planned_w)

        motor_force = 10000
        joint_indices = [self.left_wheel_id, self.right_wheel_id]
        self.p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=joint_indices,
            controlMode=self.p.VELOCITY_CONTROL,
            targetVelocities=[left_v, right_v],
            forces=[motor_force, motor_force],
            physicsClientId=self.client_id)
        return left_v, right_v

    def get_velocity(self):
        v_transition, v_rotation = self.p.getBaseVelocity(self.robot_id)
        v = math.sqrt(v_transition[0] ** 2 + v_transition[1] ** 2)
        return v

    def load_turtle_bot(self, cur_x, cur_y, cur_yaw=0):
        """
        load turtle bot from urdf, turtlebot urdf is from rl_utils
        :return:
        """
        turtle_bot_path = os.path.join(
            get_project_path(),
            "environment",
            "urdf",
            "turtlebot",
            "turtlebot.urdf",
        )
        turtle = self.p.loadURDF(
            turtle_bot_path,
            [cur_x, cur_y, 0],
            self.p.getQuaternionFromEuler([0, 0, cur_yaw]),
            flags=self.p.URDF_USE_INERTIA_FROM_FILE
                  | self.p.URDF_USE_IMPLICIT_CYLINDER,
        )

        left_joint, right_joint, lidar_joint = -1, -1, -1
        # Find relevant joints
        for j in range(self.p.getNumJoints(turtle)):
            if "wheel_left_joint" in str(self.p.getJointInfo(turtle, j)[1]):
                left_joint = j
            if "wheel_right_joint" in str(self.p.getJointInfo(turtle, j)[1]):
                right_joint = j
            if "rplidar_joint" in str(self.p.getJointInfo(turtle, j)[1]):
                lidar_joint = j

        self.wheel_dist = 0.115 * 2

        self.robot_id, self.left_wheel_id, self.right_wheel_id, self.lidar_joint_id = turtle, left_joint, right_joint, lidar_joint
