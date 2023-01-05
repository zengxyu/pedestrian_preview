from environment.robot_arm import RobotArm
import pybullet as p

env = RobotArm(render=True)
env.render()
env.reset()

while True:
    p.stepSimulation()