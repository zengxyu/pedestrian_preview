import numpy as np

from environment.circle_car_flat_env import CircleCarFlatEnv

if __name__ == '__main__':
    env = CircleCarFlatEnv(render=True, human_control=True)

    env.render()
    obs = env.reset()
    env.guiControl(np.inf)
