import math
import numpy as np
import pybullet as p
from matplotlib import pyplot as plt

from environment.nav_utilities.coordinates_converter import cvt_to_om

hitRayColor = [0, 1, 0]
missRayColor = [1, 0, 0]


class PenetrateRaySensor:
    def __init__(self, robot_id, sensor_config):
        self.robot_id = robot_id
        self.ray_num = sensor_config["ray_num"]
        self.angle = np.deg2rad(sensor_config["angle"])

        self.ray_length = sensor_config["ray_length"]
        self.height = sensor_config["height"]
        self.lidar_debug_line_ids = []
        self.occupancy_map = None
        self.grid_res = None
        self.image_width = sensor_config["image_width"]

    def register_occupancy_map(self, occupancy_map, grid_res):
        self.occupancy_map = occupancy_map
        self.grid_res = grid_res

    def get_obs(self):
        cur_position, cur_orientation_quat = p.getBasePositionAndOrientation(self.robot_id)
        cur_yaw = p.getEulerFromQuaternion(cur_orientation_quat)[2]
        # cur_yaw: 机器人x轴所在的方向
        # 从机器人x轴所在地方向发射第一根射线
        min_angle = cur_yaw - np.pi / 4
        begins = (cur_position[0], cur_position[1], self.height)
        # rayFroms = [begins for _ in range(self.ray_num)]
        # 得到四个角点的坐标
        angles = [self.angle * float(i) / self.ray_num + min_angle for i in range(4)]
        rayTos = [
            [
                begins[0] + self.ray_length * math.cos(angles[i]),
                begins[1] + self.ray_length * math.sin(angles[i])
            ]
            for i in range(4)]

        if self.occupancy_map is None or self.grid_res is None:
            raise AttributeError(
                "self.occupancy_map is None or self.grid_res is None; please call function register_occupancy_map() to register")

        rayTos = np.array(rayTos)
        cur_position = cur_position[:2]
        # occupancy map
        obs = self.penetrate_rays(rayTos, cur_position)
        # 调用激光探测函数
        # results = p.rayTestBatch(rayFroms, rayTos)
        # # plot_lidar_ray(self._bullet_client, results, rayFroms, rayTos, missRayColor, hitRayColor)
        # results = np.array(results, dtype=object)
        # hit_fractions = results[:, 2]
        # # 添加高斯噪声
        # hit_fractions = hit_fractions + np.random.randn(*hit_fractions.shape) * 0.005
        # hit_thetas = hit_thetas + np.random.randn(*hit_thetas.shape) * 0.005
        # # plot_lidar_ray(self._bullet_client, results, rayFroms, rayTos, missRayColor=[1, 0, 0], hitRayColor=[0, 0, 1])
        return obs

    def penetrate_rays(self, corners, center):
        corner0 = cvt_to_om(corners[0], self.grid_res)
        corner1 = cvt_to_om(corners[1], self.grid_res)
        corner2 = cvt_to_om(corners[2], self.grid_res)
        corner3 = cvt_to_om(corners[3], self.grid_res)

        # center = cvt_to_om((corners[0] + corners[2]) / 2, self.grid_res)
        center = cvt_to_om(center, self.grid_res)
        center = np.round(center).astype(int)

        local_occupancy_map = np.zeros((self.image_width, self.image_width))
        points_in_corner_01 = np.array(
            [np.linspace(corner0[0], corner1[0], self.image_width),
             np.linspace(corner0[1], corner1[1], self.image_width)]).transpose((1, 0))
        points_in_corner_23 = np.array(
            [np.linspace(corner2[0], corner3[0], self.image_width),
             np.linspace(corner2[1], corner3[1], self.image_width)]).transpose((1, 0))
        plt.imshow(self.occupancy_map)
        plt.plot(points_in_corner_01[:, 0], points_in_corner_01[:, 1])
        plt.plot(points_in_corner_23[:, 0], points_in_corner_23[:, 1])

        plt.show()
        points_in_corner_01 = np.round(points_in_corner_01).astype(int)
        points_in_corner_23 = np.round(points_in_corner_23).astype(int)
        for point_left, point_right in zip(points_in_corner_01, points_in_corner_23):
            points_x = np.linspace(point_left[0], point_right[0], self.image_width)
            points_y = np.linspace(point_left[1], point_right[1], self.image_width)
            points_x = np.round(points_x).astype(int)
            points_y = np.round(points_y).astype(int)
            # points = np.array([points_x, points_y]).transpose((1, 0))
            local_points_x = points_x - center[0]
            local_points_y = points_y - center[1]
            local_points_x = local_points_x + int(self.image_width / 2)
            local_points_y = local_points_y + int(self.image_width / 2)

            local_points_x = np.clip(local_points_x, 0, self.image_width - 1)
            local_points_y = np.clip(local_points_y, 0, self.image_width - 1)
            local_occupancy_map[local_points_x, local_points_y] = self.occupancy_map[points_x, points_y]
        plt.imshow(local_occupancy_map)
        plt.show()
        print()
        # points_x = np.linspace(rays_from_om[0], rays_to_om[0], radius)
        # points_y = np.linspace(rays_from_om[1], rays_to_om[1], radius)
        #
        # points_x = points_x.astype(int)
        # points_y = points_y.astype(int)
        #
        # image = np.zeros((self.image_width, self.image_height))

        # 求解
        # frac =
        return local_occupancy_map
