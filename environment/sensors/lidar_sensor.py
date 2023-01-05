import math
import numpy as np
import pybullet as p

hitRayColor = [0, 1, 0]
missRayColor = [1, 0, 0]


class LidarSensor:
    def __init__(self, robot_id, sensor_config):
        self.robot_id = robot_id
        self.ray_num = sensor_config["ray_num"]
        self.angle = np.deg2rad(sensor_config["angle"])

        self.ray_length = sensor_config["ray_length"]
        self.height = sensor_config["height"]
        self.lidar_debug_line_ids = []

    def get_obs(self):
        cur_position, cur_orientation_quat = p.getBasePositionAndOrientation(self.robot_id)
        cur_yaw = p.getEulerFromQuaternion(cur_orientation_quat)[2]
        # cur_yaw: 机器人x轴所在的方向
        # 从机器人x轴所在地方向发射第一根射线
        min_angle = cur_yaw - np.pi / 2
        begins = (cur_position[0], cur_position[1], self.height)
        rayFroms = [begins for _ in range(self.ray_num)]

        rayTos = [
            [
                begins[0] + self.ray_length * math.cos(self.angle * float(i) / self.ray_num + min_angle),
                begins[1] + self.ray_length * math.sin(self.angle * float(i) / self.ray_num + min_angle),
                begins[2]
            ]
            for i in range(self.ray_num)]
        # 机器人x轴方向为0度
        hit_thetas = np.array([(self.angle * float(i) / self.ray_num) % (2 * np.pi) for i in range(self.ray_num)])
        # 调用激光探测函数
        results = p.rayTestBatch(rayFroms, rayTos)
        # plot_lidar_ray(self._bullet_client, results, rayFroms, rayTos, missRayColor, hitRayColor)
        results = np.array(results, dtype=object)
        hit_fractions = results[:, 2]
        # 添加高斯噪声
        hit_fractions = hit_fractions + np.random.randn(*hit_fractions.shape) * 0.005
        hit_thetas = hit_thetas + np.random.randn(*hit_thetas.shape) * 0.005
        # plot_lidar_ray(self._bullet_client, results, rayFroms, rayTos, missRayColor=[1, 0, 0], hitRayColor=[0, 0, 1])
        return hit_thetas, hit_fractions
