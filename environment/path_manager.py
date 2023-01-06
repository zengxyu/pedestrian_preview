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
        self.grid_res = self.args.env_config["grid_res"]
        self.world_config = args.world_config
        self.running_config = args.running_config
        self._human_control = args.render
        # looking-forward target: the next n waypoints
        # distance between waypoints [m]

        self.waypoints_distance = self.env_config["waypoints_distance"]
        self.reach_goal_thresh = self.env_config["goal_reached_thresh"]
        self.ideal_target_distance = self.env_config["ideal_target_distance"]
        self.n_reach_goal_thresh = int(round(self.reach_goal_thresh / self.waypoints_distance))
        self.n_to_ideal_target = int(round(self.ideal_target_distance / self.waypoints_distance))

        self.original_path = None
        self.occ_map = None

        self.nearest_ind = 0
        self.ideal_ind = 0
        self.old_ideal_ind = 0

    def register_path(self, bu_path, occupancy_map=None):
        self.nearest_ind = 0
        self.ideal_ind = 0
        self.old_ideal_ind = 0
        self.occ_map = occupancy_map
        # self.smooth_path(path=bu_path, interval=5)
        equidistant_bu_path = equidistant_sampling_from_path(bu_path, self.waypoints_distance)

        self.original_path = np.array(equidistant_bu_path)
        self.update_ideal_ind()

        return self

    def update_ideal_ind(self):
        max_ind = len(self.original_path) - 1
        self.old_ideal_ind = np.clip(self.ideal_ind, 0, max_ind)
        self.ideal_ind = np.clip(self.nearest_ind + self.n_to_ideal_target, self.ideal_ind, max_ind)

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

    def check_reach_goal(self, cur_position):
        reach_goal = self.nearest_ind >= len(self.original_path) - self.n_reach_goal_thresh or compute_distance(
            cur_position, self.original_path[-1]) < self.reach_goal_thresh
        # print("reach goal:{}; distance:{}".format(reach_goal, compute_distance(
        #     cur_position, self.original_path[-1])))
        # reach_goal = compute_distance(cur_position, self.original_path[-1]) < self.reach_goal_thresh
        return reach_goal

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


def gaussian_curve(p1, p2, t):
    px = (1 - t) * p1[0] + t * p2[0]
    py = (1 - t) * p1[1] + t * p2[1]
    return np.array([px, py])


def linear_curve(p1, p2, t):
    px = (1 - t) * p1[0] + t * p2[0]
    py = (1 - t) * p1[1] + t * p2[1]
    return np.array([px, py])


def tri_bezier(p1, p2, p3, p4, t):
    parm_1 = (1 - t) ** 3
    parm_2 = 3 * (1 - t) ** 2 * t
    parm_3 = 3 * t ** 2 * (1 - t)
    parm_4 = t ** 3

    px = p1[0] * parm_1 + p2[0] * parm_2 + p3[0] * parm_3 + p4[0] * parm_4
    py = p1[1] * parm_1 + p2[1] * parm_2 + p3[1] * parm_3 + p4[1] * parm_4

    return np.array([px, py])


def double_bezier(p1, p2, p3, t):
    parm_1 = (1 - t) ** 2
    parm_2 = 2 * t * (1 - t)
    parm_3 = t ** 2
    px = p1[0] * parm_1 + p2[0] * parm_2 + p3[0] * parm_3
    py = p1[1] * parm_1 + p2[1] * parm_2 + p3[1] * parm_3

    return np.array([px, py])


def compute_distance_point_to_path(point, path):
    distances = np.linalg.norm(path - point, axis=1)
    min_distance = min(distances)
    return min_distance


def compute_distance_two_path(path1, path2):
    """
    compute distance of two paths，为path1路径上的每个点 在path2上找到距离最短的点，将它们的距离保存下来，其中最大的距离就是两条路径的距离
    :param path1:
    :param path2:
    :return:
    """
    min_distances = [0]
    for p1 in path1:
        distances = np.linalg.norm(path2 - p1, axis=1)
        min_distance = min(distances)
        min_distances.append(min_distance)

    max_min_distance = max(min_distances)
    return max_min_distance


def compute_path_length(path):
    """
    compute path length
    :param path:
    :return:
    """
    if len(path) >= 1:
        start_points = path[:-1]
        end_points = path[1:]
        distance = np.sum(np.linalg.norm(end_points - start_points, axis=1))
        return distance
    else:
        return 0
