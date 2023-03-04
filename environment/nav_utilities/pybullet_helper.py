import math

import numpy as np


def get_formatted_robot_pose_yaw(_bullet_client, robot_id):
    """
    :return: cur_position : [x, y]
            cur_orientation : [d_x, d_y, d_z] a direction vector where the robot looks at
    """
    cur_position, cur_orientation_quat = _bullet_client.getBasePositionAndOrientation(robot_id)
    cur_euler = _bullet_client.getEulerFromQuaternion(cur_orientation_quat)
    return np.array(cur_position), cur_euler[2]


def place_object(_bullet_client, object_id, x, y, yaw=None):
    """
    place object with object_id at given pose x, y, yaw
    :param _bullet_client:
    :param object_id:
    :param x:
    :param y:
    :param yaw:
    :return:
    """
    (_, _, z), _ = _bullet_client.getBasePositionAndOrientation(object_id)

    _bullet_client.resetBasePositionAndOrientation(
        object_id,
        [x, y, z],
        _bullet_client.getQuaternionFromEuler(
            [0, 0, (np.random.random_sample() * np.pi * 2 if yaw is None else yaw)]
        ),
    )


def plot_robot_direction_line(pybullet_client, robot_direction_line_id, current_pose, color=[1, 0, 0], height=1.5,
                              line_length=0.6):
    if robot_direction_line_id is not None:
        pybullet_client.removeUserDebugItem(robot_direction_line_id)
    robot_direction_line_id = pybullet_client.addUserDebugLine(
        (*current_pose[:2], height),
        (
            current_pose[0] + line_length * math.cos(current_pose[-1]),
            current_pose[1] + line_length * math.sin(current_pose[-1]),
            height,
        ),
        color,
        5,
    )
    return robot_direction_line_id
