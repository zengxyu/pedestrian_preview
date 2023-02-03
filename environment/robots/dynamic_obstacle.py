import logging
from typing import List

import numpy as np
import logging as logger

from environment.gen_scene.build_office_world import create_cylinder
from environment.nav_utilities.counter import Counter
from environment.nav_utilities.pybullet_helper import place_object
from environment.robots.base_obstacle import BaseObstacleGroup, BaseObstacle

from environment.path_manager import PathManager
from utils.math_helper import swap_value


class DynamicObstacle(BaseObstacle):
    def __init__(self, p, args, step_duration, type="dynamic"):
        # bullet_client, occupancy map, grid resolution
        super().__init__()
        self.p = p
        self.args = args
        self.env_config = args.env_config
        self._human_control = args.render
        self.grid_res = args.env_config["grid_res"]
        self.step_duration = step_duration

        # create obstacle entity
        self.start_position = None
        self.end_position = None
        self.cur_position = None
        self.theta = 0
        self.obstacle_id = -1

        # plan a path
        self.path = None
        self.type = type

        # set speed
        self.pedestrian_speed = self.env_config["pedestrian_speed_range"]
        self.pedestrian_lfd = self.env_config["pedestrian_lfd"]
        self.pedestrian_radius_range = self.env_config["pedestrian_radius_range"]
        self.pedestrian_height_range = self.env_config["pedestrian_height_range"]

        self.speed = np.random.random() * (self.pedestrian_speed[1] - self.pedestrian_speed[0]) + self.pedestrian_speed[
            0]
        self.radius = np.random.random() * (self.pedestrian_radius_range[1] - self.pedestrian_radius_range[0]) + \
                      self.pedestrian_radius_range[0]
        self.height = np.random.random() * (self.pedestrian_height_range[1] - self.pedestrian_height_range[0]) + \
                      self.pedestrian_height_range[0]
        logging.info("pedestrian id:{}; pedestrian speed:{}".format(self.obstacle_id, self.speed))
        self.verbose = True
        self.path_manager = PathManager(self.args)
        self.step_count = Counter()

    def create(self, start_position, end_position, path):
        logger.debug("create a dynamic obstacle")

        self.obstacle_id, self.start_position = create_cylinder(self.p, start_position, height=self.height, radius=self.radius)
        self.cur_position = self.start_position

        self.end_position = end_position
        self.path = path
        self.path_manager.register_path(path)
        self.path_manager.update_nearest_waypoint(self.cur_position)
        self.theta = self.get_theta()

    def get_theta(self):
        target_wp_position = self.path_manager.get_waypoint_by_distance(self.pedestrian_lfd)
        # compute delta_y, delta_x
        direction = target_wp_position - self.cur_position
        theta = np.arctan2(direction[1], direction[0])
        return theta

    def get_direction(self):
        target_wp_position = self.path_manager.get_waypoint_by_distance(self.pedestrian_lfd)
        # compute delta_y, delta_x
        direction = target_wp_position - self.cur_position
        direction = direction / np.linalg.norm(direction)
        return direction

    def get_cur_position(self):
        position, _ = self.p.getBasePositionAndOrientation(self.obstacle_id)
        self.cur_position = position[:2]
        return position[:2]

    def turn(self):
        # switch start point and end point
        self.start_position, self.end_position = swap_value(self.start_position, self.end_position)

        # reverse self.path
        if isinstance(self.path, List):
            self.path.reverse()
        else:
            self.path = self.path[::-1]

        # new a target course
        self.path_manager.register_path(self.path)

    def step(self):
        """
        if reach, turn
        :return:
        """
        self.step_count += 1

        # reset the obstacle location
        cur_position = self.get_cur_position()

        self.path_manager.update_nearest_waypoint(cur_position)

        if self.path_manager.check_reach_goal(cur_position):
            self.turn()
            if len(np.shape(self.path)) < 2:
                print("self.path_manager.deformed_path:{}".format(self.path_manager.deformed_path))
                print("cur_position:{}".format(cur_position))
            self.path_manager.update_nearest_waypoint(cur_position)

        target_wp_position = self.path_manager.get_waypoint_by_distance(self.pedestrian_lfd)
        # compute delta_y, delta_x
        direction = target_wp_position - cur_position
        theta = np.arctan2(direction[1], direction[0])
        self.theta = theta
        delta_y = self.speed * np.sin(theta) * self.step_duration
        delta_x = self.speed * np.cos(theta) * self.step_duration
        # compute the next position where the obstacle should be set
        position_next = [cur_position[0] + delta_x, cur_position[1] + delta_y]
        self.cur_position = position_next
        place_object(self.p, self.obstacle_id, *position_next)


class DynamicObstacleGroup(BaseObstacleGroup):
    def __init__(self, p, args, step_duration):
        super().__init__()
        self.p = p
        self.args = args
        self.step_duration = step_duration

    def create(self, start_positions, end_positions, paths):
        # p, _human_control, grid_res, high_env_config, step_duration)
        i = 0
        for start_position, end_position, path in zip(start_positions, end_positions, paths):
            i += 1
            obstacle = DynamicObstacle(self.p, self.args, self.step_duration)
            obstacle.create(start_position, end_position, path)
            self.obstacles.append(obstacle)
            self.obstacle_ids.append(obstacle.obstacle_id)
            logger.debug("The {}-th obstacle with obstacle id : {} created!".format(i, obstacle.obstacle_id))
        return self

    def step(self):
        for obstacle in self.obstacles:
            obstacle.step()
