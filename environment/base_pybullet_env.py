import time
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from pybullet_utils import bullet_client

import gym
import pybullet as p

from environment.nav_utilities.counter import Counter


class PybulletBaseEnv(gym.Env, ABC):
    def __init__(self, args):
        self.render = args.render

        # bullet client
        self.p = bullet_client.BulletClient(connection_mode=p.GUI if self.render else p.DIRECT)

        # bullet client id
        self.client_id = self.p._client

        self.robot_direction_id = None

        self.physical_step_duration = args.env_config["step_duration"]
        self.episode_count = Counter()
        self.step_count = Counter()
        self.occ_map = None
        self.grid_res = None

    def reset_simulation(self):
        self.p.removeAllUserDebugItems()
        self.p.resetSimulation()
        self.p.setGravity(0, 0, -9.81)
        self.p.setPhysicsEngineParameter(enableConeFriction=1)

        # Set control type
        self.p.setRealTimeSimulation(False)  # Manual simulation call
        self.p.setTimeStep(self.physical_step_duration)
        self.p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        self.p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 1)
        self.p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
        self.p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
        self.p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        self.p.resetDebugVisualizerCamera(cameraDistance=8, cameraYaw=0, cameraPitch=-60,
                                          cameraTargetPosition=[3, 3, 0])
        # Create Plane
        self.p.createMultiBody(0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_PLANE))

    def _gui_observe_entire_environment(self):
        self.p.resetDebugVisualizerCamera(
            cameraDistance=50,
            cameraYaw=50,
            cameraPitch=-60,
            cameraTargetPosition=(
                self.occ_map.shape[0] * self.grid_res / 2,
                self.occ_map.shape[1] * self.grid_res / 2,
                5,
            ),
        )

    def close(self):
        self.p.disconnect()

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self, mode="human"):
        pass

    def copy(self):
        return deepcopy(self)
