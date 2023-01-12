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
human = Man(physicsClient, partitioned=True, timestep=time_step, translation_scaling=0.95 )
human.reset()
human.resetGlobalTransformation(
    xyz=np.array([0, 0, 0.94 * human.scaling]),
    rpy=np.array([0, 0, 0]),
    gait_phase_value=0
)
p.setTimeStep(time_step, physicsClient)
p.stepSimulation()
b = 0
for i in range(1000000000000000000000):
    cubePos, cubeOrn = p.getBasePositionAndOrientation(human.body_id)
    cur_euler = p.getEulerFromQuaternion(cubeOrn)
    b += 0.001
    cubeOrn = p.getQuaternionFromEuler(np.array([0, 0, b]))
    w = np.pi / 2
    v = 1
    human.small_step(v, 0)
    p.stepSimulation()
    print("position:{}".format(human.get_position()))
    print("v:{};".format(human.get_v()))
    time.sleep(0.05)

p.disconnect()
