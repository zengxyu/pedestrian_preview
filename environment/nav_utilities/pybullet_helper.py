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
