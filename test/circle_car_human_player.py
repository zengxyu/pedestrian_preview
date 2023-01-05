
from environment.circle_car import CircleCar

if __name__ == '__main__':
    env = CircleCar(render=True, human_control=True)

    env.render()
    obs = env.reset()
    env.guiControl()
