from typing import List

import numpy as np

from utils.math_helper import compute_distance, gaussian, compute_yaw
from utils.sampling_interpolating import extract_points, equidistant_sampling_from_path


class PathManager:
    """
    search the next waypoint
    """

    def __init__(self, args):
        """
        :param path:
        """
        self.args = args
        self.env_config = args.env_config

        # distance between waypoints [m]
        self.waypoints_distance = self.env_config["waypoints_distance"]

        self.original_path = None
        self.nearest_ind = 0

    def register_path(self, bu_path):
        self.nearest_ind = 0
        self.original_path = equidistant_sampling_from_path(bu_path, self.waypoints_distance)

        return self

    def get_nearest_ind(self, path, nearest_ind, x, y):
        # update the closest waypoint ind
        dx = [x - icx for icx in path[nearest_ind:, 0]]
        dy = [y - icy for icy in path[nearest_ind:, 1]]
        d = np.hypot(dx, dy)
        ind = np.argmin(d)
        ind += nearest_ind
        return ind

    def update_nearest_waypoint(self, position):
        x, y = position[0], position[1]
        # update_nearest_ind
        self.nearest_ind = self.get_nearest_ind(self.original_path, self.nearest_ind, x, y)

    def get_waypoint_by_distance(self, LFD):
        target_ind = min(self.nearest_ind + int(LFD / self.waypoints_distance), len(self.original_path) - 1)
        return self.get_waypoints(self.original_path, target_ind)

    def get_waypoints(self, waypoints, index):
        """
        get waypoints by indexes
        :param waypoints:
        :param index:
        :return:
        """

        if isinstance(index, list):
            waypoints_in_index = [waypoints[ind] for ind in index]
            return np.array(waypoints_in_index)
        else:
            return waypoints[index]

    def check_reach_goal(self, robot_position):
        reach_goal = compute_distance(robot_position, self.original_path[-1]) < self.env_config["goal_reached_thresh"]
        return reach_goal

    def reverse(self):
        # switch start point and end point
        if isinstance(self.original_path, List):
            self.original_path.reverse()
        else:
            self.original_path = self.original_path[::-1]

        # new a target course
        self.register_path(self.original_path)