import gym
import pybullet as p
import environment
from environment.race_car import RaceCar

# environment._register()
# env = gym.make("raceCar-v0")

env = RaceCar(render=True, human_control=True)
env.render()
env.reset()

env.guiControl()
