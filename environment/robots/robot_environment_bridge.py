import numpy as np
from environment.nav_utilities.check_helper import check_collision, CollisionType
from environment.nav_utilities.coordinates_converter import cvt_to_om
from environment.nav_utilities.pybullet_helper import plot_robot_direction_line
from utils.math_helper import compute_distance


class RobotEnvBridge:
    def __init__(self, robot, env, robot_index):
        from environment.environment_bullet import EnvironmentBullet
        from environment.robots.object_robot import ObjectRobot
        self.robot: ObjectRobot = robot
        self.env: EnvironmentBullet = env
        self.geodesic_distance_dict = env.geodesic_distance_dict_list[robot_index]
        self.force_u1_x = env.force_u1_x
        self.force_u1_y = env.force_u1_y
        self.force_vx = env.force_vxs[robot_index]
        self.force_vy = env.force_vys[robot_index]
        self.obstacle_distance_map = env.obstacle_distance_map
        self.robot_direction_id = None
        self.last_distance = None
        self.last_geodesic_distance = None
        self.last_position = None
        self.trajectories = []
        self.times = []

    def compute_euclidean_distance(self):
        """
        计算欧式距离
        Returns:

        """
        euclidean_distance = compute_distance(self.robot.get_position(), self.robot.goal)
        return euclidean_distance

    def compute_delta_euclidean_distance(self):
        """
        计算欧式距离的变化
        Returns:

        """
        if self.last_distance is None:
            self.last_distance = self.compute_euclidean_distance()

        distance = self.compute_euclidean_distance()
        delta_distance = self.last_distance - distance
        return delta_distance

    def compute_delta_geodesic_distance(self):
        """
        计算测地距离的变化
        Returns:

        """
        if self.last_geodesic_distance is None:
            self.last_geodesic_distance = self.compute_geodesic_distance()

        geodesic_distance = self.compute_geodesic_distance()
        delta_geo_distance = (self.last_geodesic_distance - geodesic_distance)
        self.last_geodesic_distance = geodesic_distance
        return delta_geo_distance

    def compute_geodesic_distance(self):
        """
        计算测地距离
        Returns:

        """
        occ_pos = cvt_to_om(self.robot.get_position(), self.env.grid_res)
        occ_pos = tuple(occ_pos)

        if self.geodesic_distance_dict is None:
            return 0
        if occ_pos in self.geodesic_distance_dict.keys():
            geodesic_distance = self.geodesic_distance_dict[occ_pos]
        else:
            geodesic_distance = 100
        geodesic_distance = geodesic_distance * self.env.grid_res
        # print("geodesic_distance:{}".format(geodesic_distance))
        return geodesic_distance

    def get_force_u1(self):
        """
        地图障碍合力方向
        """
        pos = cvt_to_om(self.robot.get_position(), self.env.grid_res)

        pos[0] = np.clip(pos[0], 0, self.env.occ_map.shape[0] - 1)
        pos[1] = np.clip(pos[1], 0, self.env.occ_map.shape[1] - 1)
        fx = self.force_u1_x[pos[0], pos[1]]
        fy = self.force_u1_y[pos[0], pos[1]]
        f = np.array([fy, fx])

        return f

    def get_force_u2(self):
        """
        动态障碍物合力方向
        """
        return

    def get_force_v(self):
        """
        价值合力方向（测地距离）
        """
        pos = cvt_to_om(self.robot.get_position(), self.env.grid_res)

        pos[0] = np.clip(pos[0], 0, self.env.occ_map.shape[0] - 1)
        pos[1] = np.clip(pos[1], 0, self.env.occ_map.shape[1] - 1)
        fx = self.force_vx[pos[0], pos[1]]
        fy = self.force_vy[pos[0], pos[1]]
        f = np.array([fy, fx])
        return f

    def compute_move_s(self):
        """
        获得occupancy map上的移动向量
        Returns:
            s: 移动距离
            s_direction: 移动方向

        """
        if self.last_position is None:
            self.last_position = self.robot.get_position()
        cur_position = self.robot.get_position()
        cur_position_om = cvt_to_om(cur_position, self.env.grid_res)
        last_position_om = cvt_to_om(self.last_position, self.env.grid_res)
        self.last_position = cur_position

        s = np.linalg.norm(cur_position_om - last_position_om)
        s_direction = cur_position_om - last_position_om
        return s, s_direction

    def compute_obstacle_distance(self):
        """
        计算机器人距离障碍物的最近距离
        Returns:

        """
        occ_pos = cvt_to_om(self.robot.get_position(), self.env.grid_res)
        if self.obstacle_distance_map is None:
            return 0
        x = np.clip(occ_pos[0], 0, self.obstacle_distance_map.shape[0] - 1)
        y = np.clip(occ_pos[1], 0, self.obstacle_distance_map.shape[1] - 1)

        obstacle_distance = self.obstacle_distance_map[x, y]
        # print("obstacle distance:{}".format(obstacle_distance))
        obstacle_distance = obstacle_distance * self.env.grid_res
        return obstacle_distance

    def check_collision(self):
        if check_collision(self.env.p, [self.robot.robot_id], self.env.wall_ids):
            return CollisionType.CollisionWithWall
        elif check_collision(self.env.p, [self.robot.robot_id], self.env.npc_ids):
            return CollisionType.CollisionWithPedestrian
        elif check_collision(self.env.p, [self.robot.robot_id], self.env.agent_robot_ids):
            return CollisionType.CollisionWithAgent
        else:
            return False

    def check_reach_goal(self):
        reach_goal = compute_distance(self.robot.get_position(), self.robot.goal) < self.env.running_config[
            "goal_reached_thresh"]
        return reach_goal

    def plot_robot_direction_line(self, render):
        if render:
            id = plot_robot_direction_line(self.env.p, self.robot_direction_id, self.robot.get_x_y_yaw())
            self.robot_direction_id = id

    def clear_itself(self):
        self.env.p.removeBody(self.robot.robot_id)
        if self.robot_direction_id is not None:
            self.env.p.removeUserDebugItem(self.robot_direction_id)

    def add_to_trajectories(self):
        self.trajectories.append(self.robot.get_position())
        self.times.append(self.env.step_count)
