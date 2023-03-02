import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
import pybullet_robots.panda.panda_sim as panda_sim
import pybullet_data

p.connect(p.GUI)
p.setGravity(0, 0, -10)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
tableUid = p.loadURDF("/home/zj/project/person/pedestrian_preview/environment/robots/urdf/turtlebot/turtlebot.urdf", basePosition =[0, 0, -0.45])




while 1:
    p.setJointMotorControl2(tableUid, 2, p.POSITION_CONTROL, [1, 1])
    p.stepSimulation()
    time.sleep(1 / 240)
    p.getCameraImage(320, 240)
