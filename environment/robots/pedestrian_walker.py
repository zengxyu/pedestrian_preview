"""
Code based on
https://github.com/epfl-lasa/human-robot-collider
"""
import time

import numpy as np
import os

from environment.nav_utilities.coordinates_converter import cvt_to_bu

index_list = [
    [0, [6, 7, 8], None],
    [1, [3, 4, 5], None],
    [2, [33, 34, 35], None],
    [3, [17, 18, 19], None],
    [4, [36], -1.0],
    [5, [20], -1.0],
    [6, [28, 29, 30], None],
    [7, [12, 13, 14], None],
    [8, [37, 38, 39], None],
    [9, [21, 22, 23], None],
    [10, [31], -1.0],
    [11, [15], -1.0],
    [12, [40, 41], None],
    [13, [24, 25], None],
    [14, [0, 1, 2], None],
    [15, [9, 10, 11], None],
    [16, [43], 1.0],
    [17, [27], 1.0],
    [18, [42], -1.0],
    [19, [26], -1.0],
]


class Pedestrian:
    def __init__(self, bullet_client, s_p, e_p, s_pose, e_pose, frequency=100.0):
        """pedestrian use bullet/world coordinates, not occupancy map coordinates"""
        self._p = bullet_client

        # load pedestrian
        self.robot_id = None
        self._load_pedestrian_urdf()

        # initialize starting location and goal location, c_p is current location
        self.s_p = s_p
        self.e_p = e_p
        self.s_pose = s_pose
        self.e_pose = e_pose
        self.c_pose = s_pose
        self.yaw = self.compute_yaw(self.s_pose, self.e_pose)
        # pose containers
        self.global_xyz = np.zeros([3])
        self.global_rpy = np.zeros([3])

        self.local_xyz = np.zeros([3])
        self.local_rpy = np.zeros([3])

        self.joint_positions = np.zeros([44])

        self.cyclic_joint_positions = None
        self.cyclic_pelvis_rotations = None
        self.cyclic_pelvis_forward_velocity = None
        self.cyclic_pelvis_lateral_position = None
        self.cyclic_pelvis_vertical_position = None
        self._load_walking_config()

        self.gait_phase_step = 0
        # self.steps_per_simStep = 1
        self.steps_per_simStep = int(100.0 / frequency)
        # base velocity : how far the pedestrian move for each sec ; base velocity ~= 1.39
        self.base_velocity = self.compute_pedestrian_step_distance()

        # current pedestrian velocity
        self.velocity = self.base_velocity

        self.setColor()
        self.resetGlobalTransformation(*self.s_pose, self.yaw)

    def compute_yaw(self, s_pose, e_pose):
        """compute the direction towards g_pose from s_pose"""
        d_vec = np.array(e_pose) - np.array(s_pose)
        theta = np.arctan2(d_vec[1], d_vec[0])
        return theta

    def reached(self):
        dist = np.linalg.norm([self.e_pose[0] - self.c_pose[0], self.e_pose[1] - self.c_pose[1]])
        reach = dist < 1
        if reach:
            print("pedestrian reach goal pose:{}".format(self.e_pose))
        return reach

    def turn(self):
        yaw = self.compute_yaw(self.e_pose, self.s_pose)
        self.resetGlobalTransformation(*self.c_pose, yaw)
        self.s_p, self.e_p = self.swap_value(self.s_p, self.e_p)
        self.s_pose, self.e_pose = self.swap_value(self.s_pose, self.e_pose)

        self.yaw = yaw



    def compute_pedestrian_step_distance(self):
        """
        compute the distance for one step/pedestrian make a step from lifting left foot to lifting left foot again
        distance ~= 1.39 which is approximate to real pedestrian walking speed.
        So we set here pedestrian makes a step for each sec
        """
        distance = np.sum(np.abs(self.cyclic_pelvis_forward_velocity)) * 0.01
        return distance

    def set_velocity(self, velocity):
        """
        velocity ~ (0,2.5) m/s; human can walk at most 2.5 m/s
        :param velocity:
        :return:
        """
        if self.velocity != velocity:
            print("self.velocity:{};velocity:{}".format(self.velocity, velocity))

        velocity = np.clip(velocity, 0, 2.5)
        self.velocity = velocity

    def move(self):
        while True:
            self.step()

    def step(self):
        """
        pedestrian steps in velocity ~= 1.4
        :return:
        """

        for gp_step in range(self.cyclic_joint_positions.shape[1]):
            # print("self.get_gp_step_duration():{}".format(self.get_gp_step_duration()))
            # time.sleep(self.get_gp_step_duration())
            # self.steps_per_simStep = self.steps_per_simStep * self.velocity / self.base_velocity
            self.advance()
        cur_position, _ = self._p.getBasePositionAndOrientation(self.robot_id)
        self.c_pose = cur_position[:2]

    def get_gp_step_duration(self):
        gp_step_duration = 1 / self.cyclic_joint_positions.shape[1] * self.base_velocity / self.velocity
        return gp_step_duration

    def delete(self):
        self._p.removeBody(self.robot_id)
        del self.cyclic_joint_positions
        del self.cyclic_pelvis_forward_velocity
        del self.cyclic_pelvis_lateral_position
        del self.cyclic_pelvis_rotations
        del self.cyclic_pelvis_vertical_position

    def setColor(self):
        for index in self._p.getVisualShapeData(self.robot_id):
            self._p.changeVisualShape(
                self.robot_id,
                index[1],
                rgbaColor=[0, 0, 1, 1],
            )

    def resetGlobalTransformation(self, x, y, yaw):
        self.c_pose = (x, y)
        self.gait_phase_step = 0

        self.global_xyz = np.array([x, y, 0.94])
        self.global_rpy[2] = yaw - np.pi
        self.local_xyz[0] = 0.0 + (
                self.cyclic_pelvis_forward_velocity[self.gait_phase_step - 1]
                * 0.01
                * self.steps_per_simStep
        )
        self.local_xyz[1] = self.cyclic_pelvis_lateral_position[self.gait_phase_step]
        self.local_xyz[2] = self.cyclic_pelvis_vertical_position[self.gait_phase_step]
        self.local_rpy[:] = self.cyclic_pelvis_rotations[:, self.gait_phase_step]
        self.joint_positions[:] = self.cyclic_joint_positions[:, self.gait_phase_step]

        self._apply_pose()
        self._apply_pose()

    def advance(self):
        self.gait_phase_step += self.steps_per_simStep
        if self.gait_phase_step >= self.cyclic_joint_positions.shape[1]:
            self.gait_phase_step -= self.cyclic_joint_positions.shape[1]
        self.local_xyz[:] += (
                self.cyclic_pelvis_forward_velocity[self.gait_phase_step - 1]
                * 0.01
                * self.steps_per_simStep
        )
        self.local_xyz[1] = self.cyclic_pelvis_lateral_position[self.gait_phase_step]
        self.local_xyz[2] = self.cyclic_pelvis_vertical_position[self.gait_phase_step]
        self.local_rpy[:] = self.cyclic_pelvis_rotations[:, self.gait_phase_step]
        self.joint_positions[:] = self.cyclic_joint_positions[:, self.gait_phase_step]

        self._apply_pose()

    def _apply_pose(self):
        for ind, array, info in index_list:
            if len(array) > 1:
                if len(array) == 3:
                    q = self._p.getQuaternionFromEuler(
                        [
                            -self.joint_positions[array[0]],
                            self.joint_positions[array[2]],
                            -self.joint_positions[array[1]],
                        ]
                    )
                else:
                    q = self._p.getQuaternionFromEuler(
                        [
                            -self.joint_positions[array[0]],
                            0.0,
                            -self.joint_positions[array[1]],
                        ]
                    )
                q = (q[0], q[1], -q[2], q[3])

                if ind == 0 or ind == 1:
                    _, q = self._p.invertTransform([0, 0, 0], q)

                self._p.resetJointStateMultiDof(self.robot_id, ind, q)
            else:
                self._p.resetJointState(
                    self.robot_id, ind, info * self.joint_positions[array[0]]
                )

        # Base rotation and Zero Translation (for now)
        self._applyToURDFBody()

    # call this function AFTER applying the BT- and BP-joint angles for the urdf (as shown above)
    def _applyToURDFBody(self):
        # get the rotation from the (hypothetical) world to the pelvis
        _, _, _, _, _, rot_pelvis = self._p.getLinkState(self.robot_id, 1)

        # get the translation to the right leg frame
        _, _, _, _, trans_rleg, _ = self._p.getLinkState(self.robot_id, 2)

        # get the translation to the left leg frame
        _, _, _, _, trans_lleg, _ = self._p.getLinkState(self.robot_id, 3)

        # get the rotation from the (hypothetical) world to the chest
        pose_chest, rot_chest = self._p.getBasePositionAndOrientation(self.robot_id)

        # get the rotation from the pelvis to the chest
        _, rot_pelvis = self._p.invertTransform([0, 0, 0], rot_pelvis)
        _, rot_pelvis_to_chest = self._p.multiplyTransforms(
            [0, 0, 0], rot_pelvis, [0, 0, 0], rot_chest
        )
        _, rot_pelvis_to_chest = self._p.multiplyTransforms(
            [0, 0, 0],
            self._p.getQuaternionFromEuler([-np.pi / 2, np.pi, 0]),
            [0, 0, 0],
            rot_pelvis_to_chest,
        )

        # World coordinates
        rot_world_to_pelvis = self._p.getQuaternionFromEuler(self.local_rpy)

        # compose rotation from world to my chest frame
        _, rot_world_to_chest = self._p.multiplyTransforms(
            [0, 0, 0], rot_world_to_pelvis, [0, 0, 0], rot_pelvis_to_chest
        )

        # New pose
        world_pose = (
                self.local_xyz
                + np.array(pose_chest)
                - 0.5 * (np.array(trans_rleg) + np.array(trans_lleg))
        )

        # pre-multiply with global transform
        t_final, r_final = self._p.multiplyTransforms(
            self.global_xyz,
            self._p.getQuaternionFromEuler(self.global_rpy),
            world_pose,
            rot_world_to_chest,
        )

        # apply it to the base together with a zero translation
        self._p.resetBasePositionAndOrientation(self.robot_id, t_final, r_final)

    def _load_pedestrian_urdf(self):
        self.robot_id = self._p.loadURDF(
            os.path.join(os.path.dirname(__file__), "../urdf/pedestrian/model", "pedestrian.urdf"),
            flags=self._p.URDF_MAINTAIN_LINK_ORDER,
            globalScaling=1.0,
        )

    def _load_walking_config(self):
        """
        load the configuration which contains the pelvis local movement, which enables pedestrians to walk like a real human
        :return:
        """
        self.cyclic_joint_positions = np.load(
            os.path.join(
                os.path.dirname(__file__),
                "../urdf/pedestrian/model",
                "motion",
                "cyclic_joint_positions.npy",
            )
        )
        self.cyclic_pelvis_rotations = np.load(
            os.path.join(
                os.path.dirname(__file__),
                "../urdf/pedestrian/model",
                "motion",
                "cyclic_pelvis_rotations.npy",
            )
        )
        self.cyclic_pelvis_forward_velocity = np.load(
            os.path.join(
                os.path.dirname(__file__),
                "../urdf/pedestrian/model",
                "motion",
                "cyclic_pelvis_forward_velocity.npy",
            )
        )
        self.cyclic_pelvis_lateral_position = np.load(
            os.path.join(
                os.path.dirname(__file__),
                "../urdf/pedestrian/model",
                "motion",
                "cyclic_pelvis_lateral_position.npy",
            )
        )
        self.cyclic_pelvis_vertical_position = np.load(
            os.path.join(
                os.path.dirname(__file__),
                "../urdf/pedestrian/model",
                "motion",
                "cyclic_pelvis_vertical_position.npy",
            )
        )


if __name__ == '__main__':
    from pybullet_utils import bullet_client
    import pybullet as p
    import pybullet_data

    p = bullet_client.BulletClient(connection_mode=p.GUI)

    # bullet client id
    client_id = p._client
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf", physicsClientId=client_id)
    s_p = (0, 0)
    e_p = (5, 5)
    s_pose = cvt_to_bu(s_p, 2)
    g_pose = cvt_to_bu(e_p, 2)
    ped = Pedestrian(p, s_p, e_p, s_pose, g_pose)
    ped.set_velocity(1.0)
    ped.resetGlobalTransformation(0, 0, -np.pi / 2)
    for i in range(1000):
        # time.sleep(1)
        start = time.time()
        ped.step()
        duration = time.time() - start
        print("duration:{}".format(duration))
