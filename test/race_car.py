import gym
import numpy as np
import pybullet as p
import pybullet_data

from utils import get_local_data_path
from environment.nav_utilities.pybullet_helper import print_robot_information


class RaceCar(gym.Env):
    def __init__(self, render, human_control):
        self._render = render
        self._human_control = human_control
        self.first_launch = True
        self._physics_client_id = p.connect(p.GUI if self._render else p.DIRECT)
        self.step_num = 0
        self.robot_id = -1
        self.plane_id = -1
        self.angle_gui_id = -1
        self.throttle_gui_id = -1
        if self._human_control:
            self.addDebugGUI()

    def reset(self):
        self.step_num = 0
        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.setGravity(0., 0., -9.81)
        p.setAdditionalSearchPath(get_local_data_path())
        self.robot_id = p.loadURDF("raceCar.urdf", basePosition=np.array([0., 0., 0.]),
                                   baseOrientation=p.getQuaternionFromEuler(np.array([0., 0., 0.])),
                                   physicsClientId=self._physics_client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf", basePosition=np.array([0., 0., 0.]),
                                   baseOrientation=p.getQuaternionFromEuler(np.array([0., 0., 0.])),
                                   physicsClientId=self._physics_client_id)
        if self.first_launch:
            print_robot_information(p, self.robot_id)
            self.first_launch = False

    def step(self, action):

        pass

    def guiControl(self):
        # joints connected front wheel and backend wheel
        wheel_indices = [1, 3, 4, 5]
        # turning joints 0 , 2
        hinge_indices = [0, 2]
        while True:
            user_angle = p.readUserDebugParameter(self.angle_gui_id)
            user_throttle = p.readUserDebugParameter(self.throttle_gui_id)
            for joint_index in wheel_indices:
                p.setJointMotorControl2(self.robot_id, joint_index,
                                        p.VELOCITY_CONTROL,
                                        targetVelocity=user_throttle)
            for joint_index in hinge_indices:
                p.setJointMotorControl2(self.robot_id, joint_index,
                                        p.POSITION_CONTROL,
                                        targetPosition=user_angle)
            p.stepSimulation()

    def addDebugGUI(self):
        self.angle_gui_id = p.addUserDebugParameter("Steering", -0.5, 0.5, 0)
        self.throttle_gui_id = p.addUserDebugParameter("Throttle", -10, 10, 0)

    def render(self, mode="human"):
        pass
