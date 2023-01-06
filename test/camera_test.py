import time

import numpy as np
import pybullet as p
import pybullet_data

from environment.robots.human import Human, Man


def reset_human(human, distance, robot_angle, human_angle, gait_phase):
    human.reset()
    x = distance * np.cos(-np.pi / 2 - robot_angle)
    y = distance * np.sin(-np.pi / 2 - robot_angle)
    orientation = -np.pi / 2 - robot_angle + human_angle
    human.resetGlobalTransformation(
        xyz=np.array([x, y, 0.94 * human.scaling]),
        rpy=np.array([0, 0, orientation - np.pi / 2]),
        gait_phase_value=0
    )


physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
time_step = 0.01
human = Man(physicsClient, partitioned=True, timestep=time_step)
human.reset()
human.resetGlobalTransformation(
    xyz=np.array([0, 0, 0.94 * human.scaling]),
    rpy=np.array([0, 0, 0]),
    gait_phase_value=0
)
p.setTimeStep(time_step, physicsClient)
p.stepSimulation()
for i in range(1000000000000000000000):
    cubePos, cubeOrn = p.getBasePositionAndOrientation(human.body_id)
    cubePos = np.array(cubePos)
    # human.set_body_velocities_from_gait()
    human.advance(np.array([0, 0, 0]), np.array([0, 0, 0, 1]))
    p.stepSimulation()
    time.sleep(0.01)

p.disconnect()
