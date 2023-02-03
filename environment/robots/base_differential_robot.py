from pybullet_utils.bullet_client import BulletClient
from environment.robots.base_robot import BaseRobot


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
