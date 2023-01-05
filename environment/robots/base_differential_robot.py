from pybullet_utils.bullet_client import BulletClient

from utils.math_helper import clockwise_radian
from environment.robots.base_robot import BaseRobot

import numpy as np

import pybullet as p


class BaseDifferentialRobot(BaseRobot):
    """
    for the movement of differential-type robots
    """

    def __init__(self, p: BulletClient, client_id: int):
        super().__init__(p, client_id)
        self.left_wheel_id, self.right_wheel_id = None, None
        self.wheel_dist = None
        self.low_v = 2
        self.high_v = 8
        self.low_w = 2
        self.high_w = 8

    def adjust_direction(self, target_orientation, alpha_d=0.5, direction_thresh=0.5):
        n_step = 0
        _, cur_orientation = self.get_formatted_robot_pose()
        w = clockwise_radian(target_orientation[:2], cur_orientation[:2])
        while abs(w) > direction_thresh:
            _, cur_orientation = self.get_formatted_robot_pose()
            w = clockwise_radian(target_orientation[:2], cur_orientation[:2])
            v_turn = alpha_d * w * self.wheel_dist
            v_turn = np.clip(v_turn, self.low_w, self.high_w) if v_turn > 0 else np.clip(v_turn, -self.high_w,
                                                                                         -self.low_w)
            p.setJointMotorControl2(
                self.robot_id, self.left_wheel_id,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0,
                physicsClientId=self.client_id
            )
            p.setJointMotorControl2(
                self.robot_id, self.right_wheel_id,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=v_turn,
                physicsClientId=self.client_id
            )

            p.stepSimulation()

            # if n_step % 10000 == 0:
            #     print("adjust direction; n_step:{} ; w:{} ; v_turn:{}".format(n_step, w, v_turn))

            n_step += 1
        return w

    def move_to_pose_by_v_ctrl(self, target_position, orientation=None):
        target_position = np.array([0., 0., 0.]) if target_position is None else target_position
        orientation = np.array([0., 0., 0.]) if orientation is None else orientation

        joint_indices = [self.left_wheel_id, self.right_wheel_id]
        alpha = 500
        alpha_d = 500
        # when the distance between current base position and goal position
        # is less than distance_thresh, the robot arrives at goal position
        distance_thresh = 0.1
        delta_distance = np.inf
        # continue moving until it reaches position
        n_step = 0
        while delta_distance > distance_thresh:
            cur_position, cur_orientation = self.get_formatted_robot_pose()
            diff_position = target_position - cur_position

            v = alpha * np.linalg.norm(diff_position[:2])
            v = np.clip(v, self.low_v, self.high_v)

            w = self.adjust_direction(target_orientation=diff_position, alpha_d=alpha_d, direction_thresh=0.3)

            if n_step % 1000 == 0:
                print(
                    "target position : {}; target_orientation / diff_position: {}; \n"
                    "cur position : {}; cur orientation : {}; \n"
                    "delta_distance : {} \n"
                    "v :{}; w:{}\n \n".format(
                        target_position,
                        diff_position,
                        cur_position,
                        cur_orientation,
                        delta_distance, v, w))
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=joint_indices,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[v, v],
                forces=[1.2] * len(joint_indices),
                physicsClientId=self.client_id
            )
            delta_distance = np.linalg.norm(target_position - np.array(cur_position))
            p.stepSimulation()

            n_step += 1
