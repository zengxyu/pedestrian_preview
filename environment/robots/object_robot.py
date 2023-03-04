import math
import os
from typing import Dict

import numpy as np
from pybullet_utils.bullet_client import BulletClient

from environment.gen_scene.build_office_world import create_cylinder
from environment.nav_utilities.pybullet_helper import place_object, plot_robot_direction_line
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

        self.radius = 0.25
        self.height = 1.7

        self.with_collision = self.robot_config["with_collision"]
        # self.load_object(start_position[0], start_position[1], start_yaw)
        self.load_urdf(start_position[0], start_position[1], start_yaw)
        self.sensors = init_sensors(robot_id=self.robot_id, sensor_names=sensor_names,
                                    sensors_config=self.sensors_config)

        self.cur_v = 0
        self.cur_w = 0
        self.debug_line_id = None

    def compute_hit_distance(self, cur_position, ray_theta):
        # 探测起点
        s0 = self.radius
        x0 = s0 * np.cos(ray_theta)
        y0 = s0 * np.sin(ray_theta)

        # 障碍物的高度要比0.5高，不然探测不到
        begin = [cur_position[0] + x0, cur_position[1] + y0]
        ray_begins = [(begin[0], begin[1], 0.2)]

        # 探测终点
        distance = 10  # 探测距离
        x1 = (distance) * np.cos(ray_theta)
        y1 = (distance) * np.sin(ray_theta)

        ray_ends = [(begin[0] + x1, begin[1] + y1, 0.2)]

        # 调用激光探测函数
        results = self.p.rayTestBatch(ray_begins, ray_ends)
        # plot_lidar_ray(self._bullet_client, results, rayFroms, rayTos, missRayColor, hitRayColor)
        results = np.array(results, dtype=object)
        hit_fraction = results[:, 2].astype(float)
        hit_distance = hit_fraction * distance

        return hit_distance

    def prob_distance(self, delta_x, delta_y):
        # 机器人当前位置
        cur_position, cur_orientation_quat = self.p.getBasePositionAndOrientation(self.robot_id)
        cur_yaw = self.p.getEulerFromQuaternion(cur_orientation_quat)[2]

        # 探测机器人行走方向的障碍物距离
        # 机器人行走方向
        theta = np.arctan2(delta_y, delta_x)
        hit_distance = self.compute_hit_distance(cur_position, theta)
        # 发射另外两根射线(robot 半径)
        intent_distance = np.linalg.norm([delta_x, delta_y])

        # 如果探测到的障碍物距离小于意图移动的距离
        if hit_distance - self.radius - 0.1 < intent_distance:
            ratio = hit_distance / intent_distance
            delta_x = 0
            delta_y = 0
        else:
            print()
        pose = np.array([*cur_position, np.arctan2(delta_y, delta_x)])
        self.debug_line_id = plot_robot_direction_line(self.p, self.debug_line_id, current_pose=pose, color=[0, 0, 1])
        return delta_x, delta_y

    def small_step_pose_control(self, delta_x_, delta_y_, delta_yaw):
        delta_x, delta_y = self.prob_distance(delta_x_, delta_y_)
        if abs(delta_x - delta_x_) < 0.00001 and abs(delta_y - delta_y_) < 0.00001:
            print("delta_x == delta_x_ and delta_y == delta_y_")
        else:
            print()
        # 检测该方向障碍物的距离
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
