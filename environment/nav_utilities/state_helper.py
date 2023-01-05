import os.path
import time
import uuid

import numpy as np
from matplotlib import pyplot as plt

from agents.mapping import InputsProcess, get_reward_config, get_input_config
from environment.nav_utilities.bubble_utils import generate_tag_descriptor
from environment.nav_utilities.cast_ray_utils import cast_rays
from environment.nav_utilities.coordinates_converter import transform_robot_to_image, cvt_positions_to_reference, \
    cvt_vectorized_polar_to_cartesian
from environment.nav_utilities.counter import Counter
from environment.nav_utilities.icp import icp
from environment.nav_utilities.match_bbox import filtered_hit_coordinates
from environment.nav_utilities.visualize_utils import visualize_cartesian_list
from utils.math_helper import cartesian_2_polar, compute_distance
from environment.path_manager import PathManager
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class StateHelper:
    def __init__(self, args, step_count: Counter):
        self.args = args
        self._human_control = args.render
        self.visualize = args.visualize
        self.save = args.save
        self.grid_res = self.args.env_config["grid_res"]

        self.env_config = args.env_config
        self.running_config = args.running_config
        self.step_count = step_count

        # 路径上多远作为观察范围
        self.waypoints_distance = self.env_config["waypoints_distance"]
        self.inference_per_duration = self.running_config["inference_per_duration"]
        self.visible_zone_limit = self.args.robot_config["visible_width"]

        # -----------
        # reward
        self.reward_weight_config = get_reward_config(self.args)
        # inputs
        self.input_config = get_input_config(self.args)
        self.seq_indexes = self.input_config["seq_indexes"]

    def get_ros_next_state(self, robot, path_manager: PathManager):
        path_manager.update_ideal_ind()

        cartesian_coordinates_list = np.array(robot.cartesian_coordinates_list)
        polar_positions_list = np.array(robot.polar_positions_list)
        hits = np.array(robot.hit_vector_list)

        cartesian_list = []
        polar_list = []
        hits_list = []
        choose_index = [-3, -1]
        for i in choose_index:
            if self.state_stores[i]:
                hit_vector = hits[i]
                cartesian_positions = cartesian_coordinates_list[i]
                polar_positions = polar_positions_list[i]
                # TODO这个waypoints只用当前时刻的waypoints就可以了
                # waypoints_list.append(waypoints)
                cartesian_list.append(cartesian_positions)
                polar_list.append(polar_positions)
                hits_list.append(hit_vector)
        polar_position_list = np.array(polar_list)
        cartesian_positions_list = np.array(cartesian_list)
        hits_list = np.array(hits_list)
        if self.input_config["process"] == InputsProcess.TagdProcessing:
            reference_polar_coordinates = polar_position_list[-1]
            reference_cartesian_coordinates = cartesian_positions_list[-1]

            reference_coordinates = self.sector_process(reference_polar_coordinates,
                                                        reference_cartesian_coordinates)

            # num_points x 2
            # deformed waypoints
            waypoints = self.get_future_waypoints(path_manager,
                                                  path_manager.original_path,
                                                  path_manager.nearest_ind)
            # cvt deformed waypoints
            waypoints = self.convert_future_polar_waypoints(waypoints, robot.get_x_y(), robot.get_yaw(),
                                                            self.visible_zone_limit)
            # 2, 3(ref)
            masses_s_t, centers = self.generate_tag_descriptor_process(cartesian_positions_list, hits_list)

            visualize_cartesian_list([reference_coordinates, waypoints], title="test",
                                     labels=["reference_coordinates", "waypoints"], folder="output/test",
                                     visualize=self.visualize, save=self.save)
            # seq_len x (points + waypoints)
            return masses_s_t.astype(np.float32), reference_coordinates.astype(np.float32), waypoints.astype(
                np.float32)

    def get_next_state(self, path_manager: PathManager, cartesian_coordinates_list, polar_positions_list,
                       hit_vector_list, robot_position, robot_yaw, scan_range_max):
        path_manager.update_ideal_ind()

        if self.input_config["process"] == InputsProcess.BaselineProcessing:
            cartesian_positions_list = []

            for i in self.input_config["seq_indexes"]:
                coordinates = self.sector_process(polar_positions_list[i]).flatten()
                cartesian_positions_list.append(coordinates)

            cartesian_positions = np.array(cartesian_positions_list)
            waypoints = self.get_future_waypoints(path_manager, robot_position, robot_yaw, scan_range_max)

            results = np.concatenate([cartesian_positions.flatten(), waypoints.flatten()], axis=0)

            return results
        else:
            # num_points x 2
            waypoints = self.get_future_waypoints(path_manager, robot_position, robot_yaw, scan_range_max)

            # 使用最后一帧作为参考帧，计算spacial coordinates
            spacial_coordinates = self.sector_process(polar_positions_list[-1])

            # compute tag descriptor
            cartesian_positions_list = np.array([cartesian_coordinates_list[i] for i in self.seq_indexes])
            hits_list = np.array([hit_vector_list[i] for i in self.seq_indexes])
            tag_descriptors = self.generate_tag_descriptor_process(cartesian_positions_list, hits_list)

            # visualize_cartesian_list([spacial_coordinates, waypoints], title="test",
            #                          labels=["spacial_coordinates", "waypoints"],
            #                          folder="output/test",
            #                          visualize=True, save=self.save)
            return tag_descriptors.astype(np.float32), spacial_coordinates.astype(np.float32), waypoints.astype(
                np.float32)

    def compute_reward(self, robot_position, path_manager: PathManager, cartesian_coordinates_list, success, collision,
                       over_max_step):

        reward = 0
        reward_robot_to_ideal_target = 0
        if self.reward_weight_config["robot_to_ideal_target"] != 0:
            # 机器人到目标点的距离
            # compute the robot's distance to ideal target, 机器人到ideal target距离越大，惩罚越大
            ideal_waypoint = path_manager.get_waypoints(path_manager.original_path, path_manager.old_ideal_ind)
            distance_robot_to_ideal_target = compute_distance(
                robot_position[np.newaxis, :] - ideal_waypoint[np.newaxis, :])
            reward_robot_to_ideal_target = self.reward_weight_config[
                                               "robot_to_ideal_target"] * distance_robot_to_ideal_target
            reward += reward_robot_to_ideal_target

        # 机器人到障碍物的距离
        # compute robot to obstacles
        reward_robot_to_obstacle = 0
        if self.reward_weight_config["robot_to_obstacle"] != 0:
            cartesian_positions = cartesian_coordinates_list[-1]
            distance_robot_to_obstacles = compute_distance(cartesian_positions) * self.visible_zone_limit

            reward_robot_to_obstacle = self.reward_weight_config["robot_to_obstacle"] * (
                    0.5 - np.clip(distance_robot_to_obstacles, 0, 0.5))
            reward += reward_robot_to_obstacle

        # collision reward
        collision_reward = self.reward_weight_config["collision"] * collision
        reward += collision_reward

        #
        reward_over_max_step = self.reward_weight_config["over_max_step"] * over_max_step
        reward += reward_over_max_step

        # 速度为0, 要给惩罚
        reward_success = self.reward_weight_config["success"] * success

        info = {"reward_collision": collision_reward,
                "reward_robot_to_obstacles": reward_robot_to_obstacle,
                "reward_robot_to_ideal_target": reward_robot_to_ideal_target,
                "reward_over_max_step": reward_over_max_step,
                "reward_success": reward_success,
                "reward": reward}

        return reward, info

    def sector_process(self, polar_positions):
        """从每个扇区中选择距离最小的点作为代表这个sector的点"""
        group_num = self.input_config["num_points"]
        half_group_theta_offset = 2 * np.pi / (2 * group_num)
        group_thetas = np.linspace(0, 2 * np.pi, group_num, endpoint=False) + half_group_theta_offset

        distances = polar_positions[:, 1]
        splitted_distance = np.array_split(distances, group_num, axis=0)

        if len(distances) % group_num == 0:
            # 50 x 1
            arg_min_dists = np.argmin(splitted_distance, axis=1)
            result = np.array([[group_thetas[i], splitted_distance[i][arg]] for i, arg in enumerate(arg_min_dists)])
        else:
            # 50 x 1
            arg_min_dists = [np.argmin(hit_sectors_distance) for hit_sectors_distance in splitted_distance]
            result = np.array([[group_thetas[i], splitted_distance[i][arg]] for i, arg in enumerate(arg_min_dists)])

        # convert polar coordinates to cartesian coordinates
        result = cvt_vectorized_polar_to_cartesian(result)
        return result

    def generate_tag_descriptor_process(self, cartesian_positions, hits):
        num_rays = self.input_config["num_rays"]
        dim_points = self.input_config["dim_points"]
        theta_thresh = np.pi * 2 / num_rays / 2
        distance_thresh = self.input_config["distance_thresh"]

        hit_sum = np.array([np.sum(hi) for hi in hits])
        if np.any(hit_sum == 0):
            tag_descriptors = np.zeros((num_rays, 2, dim_points))
            return tag_descriptors

        # filter out the non-hit rays
        hit_coordinates_list = filtered_hit_coordinates(cartesian_positions, hits)
        # 先做ICP对齐
        icp_coordinates_list = self.align_coordinates_by_icp(cartesian_positions_list=hit_coordinates_list)
        # generate tag descriptor
        tag_descriptors, xs_s_t_groups, ys_s_t_groups, virtual_centers = generate_tag_descriptor(icp_coordinates_list,
                                                                                                 num_rays, theta_thresh,
                                                                                                 distance_thresh,
                                                                                                 self.visualize,
                                                                                                 self.save,
                                                                                                 "output/descriptors")
        # save_folder = os.path.join("output", "tag_descriptor")
        # if not os.path.exists(save_folder):
        #     os.makedirs(save_folder)
        # visualize_tag_descriptors(cartesian_positions, hit_coordinates_list, icp_coordinates_list, tag_descriptors,
        #                           xs_s_t_groups, ys_s_t_groups, virtual_centers,
        #                           save_folder=save_folder,
        #                           save_path="{}_tag_{}.png".format(uuid.uuid1(), self.step_count.value))

        return tag_descriptors

    def icp_process(self, cartesian_positions, hits):

        hit_sum = np.array([np.sum(hi) for hi in hits])
        if np.any(hit_sum == 0):
            descriptors = np.zeros((2, self.input_config["num_points"], 2))
            return descriptors
        # 过滤hit点
        coordinates_hits_list = filtered_hit_coordinates(cartesian_positions, hits)

        # 先做ICP对齐
        coordinates_hits_list = self.align_coordinates_by_icp(cartesian_positions_list=coordinates_hits_list)

        # 通过射线投射
        # 90个射线
        cartesian_positions_list = cast_rays(coordinates_hits_list, num_rays=self.input_config["num_points"])
        # cartesian_positions_list = bubble_descriptor(coordinates_hits_list, num_rays=30)

        # visualize_cartesian_list(cartesian_positions_list, "{}_resample".format(self.input_config["num_points"]),
        #                          folder="output/icp_dt_pictures",
        #                          visualize=self.visualize,
        #                          save=self.save)
        # 分格子
        # 求dt
        observation_list = np.array([cartesian_positions_list[0], cartesian_positions_list[1]])

        cartesian_positions = np.array(observation_list)

        return cartesian_positions

    def convert_future_polar_waypoints(self, path_waypoints, robot_position, robot_yaw, visible_zone_limit):
        """

        :param robot_yaw:
        :param robot_position:
        :param path_waypoints:
        :return:
        """

        # 转成相对于机器人坐标系的坐标和角度
        relative_waypoints = cvt_positions_to_reference(positions=path_waypoints,
                                                        reference_position=robot_position,
                                                        reference_yaw=robot_yaw)

        # normalize distance
        relative_waypoints = normalize_cartesian_positions(relative_waypoints, visible_zone_limit)
        results = relative_waypoints

        return results

    def get_future_waypoints(self, path_manager: PathManager, robot_position, robot_yaw, scan_range_max):
        # waypoints_num = int(round(self.input_dims_configs["waypoints"] / 2))
        delta_indexes = np.array([6, 12, 18, 24, 30])

        indexes = np.clip(delta_indexes + path_manager.nearest_ind, 0, len(path_manager.original_path) - 1)
        waypoints = path_manager.get_waypoints(path_manager.original_path, indexes)
        waypoints = self.convert_future_polar_waypoints(waypoints, robot_position, robot_yaw, scan_range_max)

        return waypoints

    def align_coordinates_by_icp(self, cartesian_positions_list):
        reference_positions = cartesian_positions_list[-1]

        cartesian_positions_list_to_be_aligned = cartesian_positions_list[:-1]

        aligned_cartesian_positions_list = []
        for cartesian_positions in cartesian_positions_list_to_be_aligned:
            aligned_cartesian_positions = self.icp(reference_positions, cartesian_positions)
            aligned_cartesian_positions_list.append(aligned_cartesian_positions)
        aligned_cartesian_positions_list.append(reference_positions)
        return aligned_cartesian_positions_list

    def icp(self, reference_points, points_to_be_aligned):
        # run icp
        transformation_history, aligned_points = icp(reference_points, points_to_be_aligned, verbose=False)
        return aligned_points


def normalize_cartesian_positions(positions, visible_zone_limit):
    positions = positions / visible_zone_limit
    return positions


def compute_distance(relative_obstacle_positions):
    """
    compute min distance between robot and obstacles
    :param relative_obstacle_positions:
    :return:
    """
    # print("relative_obstacle_positions:{}".format(relative_obstacle_positions))
    distances = np.linalg.norm(relative_obstacle_positions, axis=1)
    min_distance = np.min(distances)
    return min_distance


def visualize_hit_positions(cvt_hit_positions, visible_zone_limit, scale=10):
    w = scale * 2 * visible_zone_limit
    r, t = transform_robot_to_image(w)
    scan_map = np.zeros(shape=(w, w))
    for position in cvt_hit_positions:
        x, y = r @ position * scale + t
        x = np.clip(int(x), 0, w - 1)
        y = np.clip(int(y), 0, w - 1)
        if x >= w or y >= w:
            print("x:{}; y:{}".format(x, y))
        scan_map[int(x), int(y)] = 1
    plt.imshow(scan_map)
    plt.show()
    return scan_map


colors = ["#B9DB35", "#FAC366",
          "#FF4E0F", "#E000FF", "#3563DB", "#46F2A8", "#B9DB35", "#FAC366",
          "#FF4E0F", "#B9DB35", "#FAC366", "#E000FF", "#3563DB", "#46F2A8",
          "#7B28DE", "#51F5E7", "#CEE04A", "#FCA872"]


def draw_2d_rectangle(x1, y1, x2, y2):
    # diagonal line
    # plt.plot([x1, x2], [y1, y2], linestyle='dashed')
    # four sides of the rectangle
    plt.plot([x1, x2], [y1, y1], color='r')  # -->
    plt.plot([x2, x2], [y1, y2], color='g')  # | (up)
    plt.plot([x2, x1], [y2, y2], color='b')  # <--
    plt.plot([x1, x1], [y2, y1], color='k')  # | (down)


def visualize_descriptors2(coordinates_hits_list, descriptors, folder, visualize):
    N = len(descriptors)
    T = len(descriptors[0])

    if visualize:
        for t in range(T):
            for n in range(N):
                descriptor = descriptors[n][t]
                descriptor = descriptor.reshape((-1, 2))
                plt.scatter(descriptor[:, 0], descriptor[:, 1], color=colors[n % len(colors)])
        plt.show()


def visualize_descriptors(coordinates_hits_list, descriptors, folder, visualize):
    if visualize:
        if len(descriptors.shape) == 1:
            return
        T = len(coordinates_hits_list)

        for i in range(T):
            coordinates_hits = coordinates_hits_list[i]
            plt.scatter(coordinates_hits[:, 0], coordinates_hits[:, 1])
        T, group_num, channel = descriptors.shape
        descriptors = descriptors.reshape((T, group_num, -1, 2))
        for t in range(T):
            for i in range(group_num):
                corners = descriptors[t, i]
                # corners = corners[:-1]
                # corners = np.concatenate([corners, corners[0][np.newaxis, :]])

                plt.plot(corners[:, 0], corners[:, 1], '-', color=colors[t % len(colors)])
        if not os.path.exists(folder):
            os.makedirs(folder)
        # plt.savefig("{}/_{}.png".format(folder, time.time()))
        plt.show()
        plt.clf()


def visualize_bbox(coordinates_groups, corners_groups, centers_group, title, visualize):
    if visualize:
        plt.title(title)

        for points, corners, centers in zip(coordinates_groups, corners_groups, centers_group):
            # plt.scatter(points[:, 0], points[:, 1])
            # plt.scatter(centers[0], centers[1])
            corners = np.concatenate([corners, corners[0][np.newaxis, :]])
            plt.plot(corners[:, 0], corners[:, 1], '-')
        plt.show()
        plt.clf()


def visualize_chunk(coordinates_flatten_list, index_paires, title, visualize):
    if visualize:
        for i, index_pair in enumerate(index_paires):
            start_ind, end_ind = index_pair
            if start_ind < end_ind:
                coords = coordinates_flatten_list[start_ind:end_ind]
            else:
                coords = np.concatenate([coordinates_flatten_list[start_ind:], coordinates_flatten_list[:end_ind]])
            xs = coords[:, 0]
            ys = coords[:, 1]
            color = colors[i % len(colors)]

            plt.scatter(xs, ys, c=color)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        # plt.show()
        directory = "output/chunk"
        if not os.path.exists(directory):
            os.makedirs(directory)
        # plt.show()
        plt.savefig("output/chunk/{}.png".format(time.time()))
        plt.clf()


def visualize_cartesian_list2(coordinates_groups_list, title):
    group_list = []
    markers = ['x', 'o', 's', '.']
    for t, coordinates_groups in enumerate(coordinates_groups_list):
        for i, coordinates_group in enumerate(coordinates_groups):
            group_list.append(coordinates_group.coordinates)
            color = colors[i % len(colors)]
            coordinates_flatten = np.array(coordinates_group.coordinates)
            xs = coordinates_flatten.reshape((-1, 2))[:, 0]
            ys = coordinates_flatten.reshape((-1, 2))[:, 1]
            plt.scatter(xs, ys, lw=0.5, marker=markers[t], c=color)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.show()


def visualize_polar_list(coordinates_flatten_list, colors, title="Plot lidar polar positions"):
    for i, coordinates_flatten in enumerate(coordinates_flatten_list):
        color = colors[i % len(colors)]
        thetas = coordinates_flatten.reshape((-1, 2))[:, 0]
        dists = coordinates_flatten.reshape((-1, 2))[:, 1]
        plt.polar(thetas, dists, 'o', lw=2, c=color)
    plt.title(title)
    plt.ylim(0, 1)
    plt.show()


def visualize_motion_vector(reference_coordinates, matched_motion_vectors, visualize):
    if visualize:
        S = len(reference_coordinates)
        plt.scatter(reference_coordinates[:, 0], reference_coordinates[:, 1], c='r')
        temp_coordinates = reference_coordinates + matched_motion_vectors
        for s in range(S):
            plt.plot([temp_coordinates[s][0], reference_coordinates[s][0]],
                     [temp_coordinates[s][1], reference_coordinates[s][1]])
        plt.show()
