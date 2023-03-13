import math
import os
from typing import Dict

import numpy as np
from pybullet_utils.bullet_client import BulletClient

from environment.gen_scene.build_office_world import create_cylinder
from environment.nav_utilities.pybullet_helper import place_object, plot_robot_direction_line
from environment.robots.base_robot import BaseRobot
from environment.robots.robot_environment_bridge import RobotEnvBridge
from environment.robots.robot_roles import get_role_color
from environment.sensors.sensor_types import init_sensors
from utils.fo_utility import get_project_path


class ObjectRobot(BaseRobot):
    def __init__(self, p: BulletClient, client_id: int, robot_role: str, step_duration: float, robot_config: Dict,
                 sensor_names: str, sensors_config: Dict, start_position, goal_position, start_yaw):
        super().__init__(p, client_id)
        # RobotEnvBridge.__init__(self)
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
        self.start = start_position
        self.goal = goal_position

        self.cur_v = 0
        self.cur_w = 0
        self.debug_line_id = None
        self.bridge: RobotEnvBridge = None

    def set_bridge(self, bridge):
        self.bridge = bridge

    def compute_move_xy(self, delta_x, delta_y):
        """
        delta_x, delta_y世界坐标上移动delta_x, delta_y
        方法1：根据x,y方向上与坐标轴的距离来判断
        """
        distance = 10  # 探测距离
        measure_height = 0.2  # 测量高度

        # 机器人当前位置
        cur_position, _ = self.p.getBasePositionAndOrientation(self.robot_id)

        # 测量机器人与x,y方向墙的距离
        sign_x = 1 if delta_x > 0 else -1
        sign_y = 1 if delta_y > 0 else -1
        ray_begins = [(*cur_position[:2], measure_height)] * 2
        ray_ends = [(cur_position[0] + sign_x * distance, cur_position[1], measure_height),
                    (cur_position[0], cur_position[1] + sign_y * distance, measure_height)]
        results = self.p.rayTestBatch(ray_begins, ray_ends)
        results = np.array(results, dtype=object)
        hit_fraction = results[:, 2]
        hit_distances = hit_fraction * distance - self.radius
        # if hit_distances[0] <= abs(delta_x) or hit_distances[1] <= abs(delta_y):
        #     return 0, 0
        # return delta_x, delta_y
        # 分别走x,y轴上分量更小的
        return min(hit_distances[0], abs(delta_x)) * sign_x, min(hit_distances[1], abs(delta_y)) * sign_y

    def compute_move_xy2(self, delta_x, delta_y):
        """
        方法2：根据delta_x和delta_y的值扫描90度范围的距离
        """
        distance = 10  # 探测距离
        measure_height = 0.2  # 测量高度

        # 机器人当前位置
        cur_position, _ = self.p.getBasePositionAndOrientation(self.robot_id)

        # 判断要测量的角度
        if delta_x >= 0:
            if delta_y >= 0:
                theta_range = (0, np.pi / 2)
            else:
                theta_range = (-np.pi / 2, 0)
        else:
            if delta_y >= 0:
                theta_range = (np.pi / 2, np.pi)
            else:
                theta_range = (-np.pi, -np.pi / 2)

        theta_size = 100
        # 加angle_step是为了遍历到最后一个值
        thetas = np.linspace(theta_range[0], theta_range[1], theta_size)
        x_ends = [distance * np.cos(theta) for theta in thetas]
        y_ends = [distance * np.sin(theta) for theta in thetas]

        ray_begins = [(*cur_position[:2], measure_height)] * theta_size
        ray_ends = [(cur_position[0] + x_end, cur_position[1] + y_end, measure_height) for x_end, y_end in
                    zip(x_ends, y_ends)]
        results = self.p.rayTestBatch(ray_begins, ray_ends)
        results = np.array(results, dtype=object)
        hit_fraction = results[:, 2]
        ind_min_dist = np.argmin(hit_fraction)
        hit_distance = hit_fraction[ind_min_dist] * distance - self.radius - 0.1

        intent_distance = np.linalg.norm([delta_x, delta_y])
        hit_distance = max(hit_distance, 0)
        if hit_distance <= intent_distance:
            theta = np.arctan2(delta_y, delta_x)
            # 尽量往前走一段
            delta_x, delta_y = hit_distance * np.cos(theta), hit_distance * np.sin(theta)
        return delta_x, delta_y

    def compute_hit_distance(self, cur_position, ray_theta, height=0.2):
        # 探测起点
        s0 = self.radius
        # x0 = s0 * np.cos(ray_theta)
        # y0 = s0 * np.sin(ray_theta)
        x0, y0 = 0, 0
        # 障碍物的高度要比height高，不然探测不到
        begin = [cur_position[0] + x0, cur_position[1] + y0]
        ray_begins = [(begin[0], begin[1], height)]

        # 探测终点
        distance = 10  # 探测距离
        x1 = distance * np.cos(ray_theta)
        y1 = distance * np.sin(ray_theta)

        ray_ends = [(begin[0] + x1, begin[1] + y1, height)]

        # 调用激光探测函数
        results = self.p.rayTestBatch(ray_begins, ray_ends)
        # plot_lidar_ray(self._bullet_client, results, rayFroms, rayTos, missRayColor, hitRayColor)
        results = np.array(results, dtype=object)
        hit_fraction = results[:, 2].astype(float)
        hit_distance = hit_fraction * distance - self.radius

        return hit_distance

    def prob_distance(self, delta_x, delta_y):
        # 机器人当前位置
        cur_position, cur_orientation_quat = self.p.getBasePositionAndOrientation(self.robot_id)
        cur_yaw = self.p.getEulerFromQuaternion(cur_orientation_quat)[2]

        # 探测机器人行走方向的障碍物距离
        # 机器人行走方向
        theta = np.arctan2(delta_y, delta_x)
        hit_distance = self.compute_hit_distance(cur_position, theta)
        # 机器人计划走的距离
        intent_distance = np.linalg.norm([delta_x, delta_y])

        # 如果探测到的障碍物距离小于意图移动的距离
        if hit_distance - self.radius - 0.1 < intent_distance:
            ratio = hit_distance / intent_distance
            delta_x = 0
            delta_y = 0
        else:
            print()
        pose = np.array([*cur_position, theta])
        self.debug_line_id = plot_robot_direction_line(self.p, self.debug_line_id, current_pose=pose, color=[0, 0, 1],
                                                       height=0.5, line_length=1.5)

        return delta_x, delta_y

    def small_step_xy_yaw_control(self, delta_x_, delta_y_, delta_yaw):
        delta_x, delta_y = self.compute_move_xy(delta_x_, delta_y_)
        # if abs(delta_x - delta_x_) < 0.00001 and abs(delta_y - delta_y_) < 0.00001:
        #     print("delta_x == delta_x_ and delta_y == delta_y_")
        # else:
        #     print()
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

    def small_step_xy_control(self, delta_x_, delta_y_):
        delta_x, delta_y = self.compute_move_xy(delta_x_, delta_y_)
        # 检测该方向障碍物的距离
        cur_position = self.get_position()
        cur_yaw = self.get_yaw()
        (_, _, z), _ = self.p.getBasePositionAndOrientation(self.robot_id)
        target_position = np.array([*cur_position, z]) + np.array([delta_x, delta_y, 0])
        target_yaw = np.arctan2(delta_y, delta_x)
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
