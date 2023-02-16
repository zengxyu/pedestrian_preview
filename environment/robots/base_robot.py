from pybullet_utils.bullet_client import BulletClient
import numpy as np


class BaseRobot:
    def __init__(self, p: BulletClient, client_id: int):
        self.p = p
        self.client_id = client_id
        self.robot_id = None

    def get_position(self):
        cur_position, cur_orientation_quat = self.p.getBasePositionAndOrientation(self.robot_id)
        return np.array([cur_position[0], cur_position[1]])

    def get_yaw(self):
        cur_position, cur_orientation_quat = self.p.getBasePositionAndOrientation(self.robot_id)
        cur_euler = self.p.getEulerFromQuaternion(cur_orientation_quat)
        return cur_euler[2]

    def get_v(self):
        cur_v, cur_w = self.p.getBaseVelocity(self.robot_id)
        speed = np.linalg.norm(cur_v[:2])
        return speed

    def get_w(self):
        cur_v, cur_w = self.p.getBaseVelocity(self.robot_id)
        return cur_w[2]

    def get_x_y_yaw(self):
        position = self.get_position()
        yaw = self.get_yaw()
        return np.array([position[0], position[1], yaw])

    def reset_base_body(self, linear_velocity, angular_velocity):
        """

        :param linear_velocity:
        :param angular_velocity:
        :return: turtle_bot_id, left_joint_id, right_joint_id, lidar_joint_id
        """

        self.p.resetBaseVelocity(
            physicsClientId=self.p,
            objectUniqueId=self.robot_id,
            linearVelocity=linear_velocity,
            angularVelocity=angular_velocity
        )
        self.p.stepSimulation()

    def reset_to_pose(self, position, orientation=None, check_collision=False):
        """
         move a step by resetting the position and orientation of robot
         It is best only to do this at the start, and not during a running simulation,
         since the command will override the effect of all physics simulation.
        :param position:
        :param orientation: euler angle
        :param check_collision:
        :return:
        """
        position = np.array([0., 0., 0.]) if position is None else position
        orientation = np.array([0., 0., np.pi / 2]) if orientation is None else orientation
        self.p.resetBasePositionAndOrientation(
            physicsClientId=self.client_id,
            bodyUniqueId=self.robot_id,
            posObj=position,
            ornObj=self.p.getQuaternionFromEuler(orientation)
        )
        self.p.stepSimulation()
        if check_collision:
            pass
        return

    def small_step(self, planned_v, planned_w):
        pass
