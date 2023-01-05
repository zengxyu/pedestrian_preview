import numpy as np


def print_robot_information(p, robot_id):
    if robot_id is not None:
        print("\n robot information:")
        print("\t robot joints:")
        number_of_joints = p.getNumJoints(robot_id)
        for joint_index in range(number_of_joints):
            info = p.getJointInfo(robot_id, joint_index)
            print(info[0], ": ", info[1])
    else:
        print("please load robot urdf first!")


def get_joint_id_by_name(p, robot_id, name):
    number_of_joints = p.getNumJoints(robot_id)
    for joint_index in range(number_of_joints):
        info = p.getJointInfo(robot_id, joint_index)
        if str(info[1], encoding="utf-8") == name:
            return joint_index
    print("No joint with name : {}".format(name))
    return -1


def get_formatted_robot_pose(_bullet_client, robot_id):
    """
    :return: cur_position : [x, y]
            cur_orientation : [d_x, d_y, d_z] a direction vector where the robot looks at
    """
    cur_position, cur_orientation_quat = _bullet_client.getBasePositionAndOrientation(robot_id)
    cur_orientation_mat = _bullet_client.getMatrixFromQuaternion(cur_orientation_quat)
    cur_orientation = np.array([cur_orientation_mat[0], cur_orientation_mat[3], cur_orientation_mat[6]])
    return np.array(cur_position), np.array(cur_orientation)


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
