import math
from math import cos, sin
from typing import List

import numpy as np
import random


def gui_observe_entire_environment(pybullet_client, cur_x, cur_y):
    pybullet_client.resetDebugVisualizerCamera(
        cameraDistance=7,
        cameraYaw=0,
        cameraPitch=-60,
        cameraTargetPosition=(
            cur_x,
            cur_y,
            0,
        ),
    )


def plot_mark_spot(_bullet_client, spot_position, size=0.05, color=[0, 0, 1]):
    """

    :param _bullet_client:
    :param spot_position:
    :param color: blue
    :return:
    """
    x, y = spot_position
    line1_start = [x - size, y - size]
    line1_end = [x + size, y + size]
    line2_start = [x - size, y + size]
    line2_end = [x + size, y - size]

    line1_id = _bullet_client.addUserDebugLine(
        lineFromXYZ=[*line1_start, 0],
        lineToXYZ=[*line1_end, 0],
        lineColorRGB=color,
        lineWidth=4
    )
    line2_id = _bullet_client.addUserDebugLine(
        lineFromXYZ=[*line2_start, 0],
        lineToXYZ=[*line2_end, 0],
        lineColorRGB=color,
        lineWidth=4
    )

    return [line1_id, line2_id]


def plot_ground_line(_bullet_client, from_point, to_point, color=[0, 1, 0]):
    return _bullet_client.addUserDebugLine(
        lineFromXYZ=[*from_point, 0.0],
        lineToXYZ=[*to_point, 0.0],
        lineColorRGB=color,
        lineWidth=4
    )


def plot_line(_bullet_client, from_point, to_point, color=[0, 1, 0]):
    """
    plot path
    :param _bullet_client:
    :param from_point:
    :param to_point:
    :return:
    """
    return _bullet_client.addUserDebugLine(
        lineFromXYZ=from_point,
        lineToXYZ=to_point,
        lineColorRGB=color,
        lineWidth=4
    )


def get_random_color():
    color = [random.random() * 0.9, random.random() * 0.9, random.random() * 0.9]
    return color


def plot_bullet_path(_bullet_client, bullet_path, size, color=[0, 1, 0]):
    """
    plot path
    :param color: green
    :param _bullet_client:
    :param bullet_path:
    :return:
    """
    start_points = bullet_path[:-1]
    end_points = bullet_path[1:]
    for (sx, sy), (ex, ey) in zip(start_points, end_points):
        plot_line(_bullet_client, [sx, sy, 0.0], [ex, ey, 0.0], color=color)
        plot_mark_spot(_bullet_client, spot_position=[sx, sy], size=size, color=color)
    if len(bullet_path) > 0:
        plot_mark_spot(_bullet_client, spot_position=bullet_path[-1], size=size, color=color)


def draw_robot_direction(_bullet_client, robot_x, robot_y, robot_yaw, color=[1, 1, 0]):
    end_x = robot_x + cos(robot_yaw)
    end_y = robot_y + sin(robot_yaw)

    line_id = plot_line(_bullet_client, [robot_x, robot_y, 0.5], [end_x, end_y, 0.5], color=color)

    return line_id


def draw_lines(p, cur_position, target_waypoint, target_index, bullet_path):
    """
    draw a line from cur_position to target_waypoint
    and a line from target_waypoint to a waypoint after 5 waypoints from target_position
    :param cur_position:
    :param target_waypoint:
    :param target_index:
    :param bullet_path:
    :return:
    """
    p.addUserDebugLine(
        lineFromXYZ=[*cur_position, 0.5],
        lineToXYZ=[*target_waypoint, 0.5],
        lineColorRGB=[1, 0, 0],
        lineWidth=2
    )
    p.addUserDebugLine(
        lineFromXYZ=[*bullet_path[target_index], 0.5],
        lineToXYZ=[*bullet_path[target_index + 5], 0.5],
        lineColorRGB=[1, 0, 0],
        lineWidth=2
    )


def plot_lidar_ray(_bullet_client, results, rayFroms, rayTos, missRayColor, hitRayColor):
    lidar_debug_line_ids = []
    # 根据results结果给激光染色
    for index, result in enumerate(results):
        if result[0] == -1:
            line_id = _bullet_client.addUserDebugLine(rayFroms[index], rayTos[index], missRayColor)
            lidar_debug_line_ids.append(line_id)

        else:
            line_id = _bullet_client.addUserDebugLine(rayFroms[index], rayTos[index], hitRayColor)
            lidar_debug_line_ids.append(line_id)

    print("line num:{}".format(len(lidar_debug_line_ids)))
    return lidar_debug_line_ids


def plot_gibson_lidar_ray(_bullet_client, hit_vectors, rayFroms, rayTos, missRayColor, hitRayColor):
    lidar_debug_line_ids = []
    # 根据results结果给激光染色
    for index, hit in enumerate(hit_vectors):
        if hit == 0:
            line_id = _bullet_client.addUserDebugLine(rayFroms, rayTos[index], missRayColor)
            lidar_debug_line_ids.append(line_id)

        else:
            line_id = _bullet_client.addUserDebugLine(rayFroms, rayTos[index], hitRayColor)
            lidar_debug_line_ids.append(line_id)

    print("line num:{}".format(len(lidar_debug_line_ids)))
    return lidar_debug_line_ids


def remove_lidar_debug_lines(_bullet_client, client_id, lidar_debug_line_ids):
    _bullet_client.removeAllUserDebugItems()
    for line_id in lidar_debug_line_ids:
        _bullet_client.removeUserDebugItem(physicsClientId=client_id, itemUniqueId=line_id)
        _bullet_client.removeUserDebugItem(physicsClientId=client_id, itemUniqueId=line_id)

    lidar_debug_line_ids = []


def plot_trajectory(_p, path: np.ndarray, color: List[float], step_size=1) -> List[int]:
    """
    Plot trajectory in pybullet environment

    Parameters
    ----------
    path : [[x1,y1],[x2,y2],...]
        numpy array of the coordinates in environment coordinate
    color : Tuple[float, float, float]
        color of the trajectory

    Returns
    -------

    """

    path_ids = []
    last_coord = None
    trajectory_height = 0.05
    for i in range(0, len(path), step_size):
        node = path[i]
        if last_coord is not None:
            id = _p.addUserDebugLine(last_coord, (*node, trajectory_height), color, 5)
            path_ids.append(id)
        last_coord = (*node, trajectory_height)
    return path_ids


def plot_robot_direction_line(pybullet_client, robot_direction_line_id, current_pose):
    if robot_direction_line_id is not None:
        pybullet_client.removeUserDebugItem(robot_direction_line_id)
    robot_direction_line_id = pybullet_client.addUserDebugLine(
        (*current_pose[:2], 0.1),
        (
            current_pose[0] + 0.6 * math.cos(current_pose[2]),
            current_pose[1] + 0.6 * math.sin(current_pose[2]),
            0.1,
        ),
        [1, 0, 0],
        5,
    )
    return robot_direction_line_id

# def draw_parameters(self, i, target_position, robot_position, robot_orientation, distance_to_target, v, w, left_v,
#                     right_v):
#     target_position_str = "i:{}; target position:{}".format(i, str(np.around(target_position, decimals=2)))
#     robot_position_orientation_str = "robot position:{}; robot orientation:{}; distance_to_target:{}".format(
#         str(np.around(robot_position, decimals=2)), str(np.around(robot_orientation, decimals=2)),
#         str(round(distance_to_target, 2)))
#
#     robot_v_w_str = "robot speed:{}; robot w:{}; left v:{}; right v:{}".format(np.around(v, decimals=2),
#                                                                                np.around(w, decimals=2),
#                                                                                np.around(left_v, decimals=2),
#                                                                                np.around(right_v, decimals=2))
#     p.addUserDebugText(target_position_str, [0., 0., 5.], [1, 0, 0])
#     p.addUserDebugText(robot_position_orientation_str, [0., 0., 4.5], [1, 0, 0])
#     p.addUserDebugText(robot_v_w_str, [0., 0., 4.], [1, 0, 0])
