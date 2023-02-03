from typing import Dict

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
        self.wheel_base = None
        self.robot_config = None
        self.v_ctrl_factor: float = None
        self.w_ctrl_factor: float = None
