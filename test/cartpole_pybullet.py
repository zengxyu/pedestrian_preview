import pybullet as p
import pybullet_envs
from time import sleep
import gym
from pybullet_envs.bullet import CartPoleBulletEnv
# pip install stable-baselines 基于tensorflow的，
from stable_baselines3.dqn import DQN
# cid = p.connect(p.DIRECT)
# env = gym.make("CartPoleContinuousBulletEnv-v0")
# 使用 CartPoleBulletEnv 创建环境是不需要链接引擎的，
# 因为连接引擎的部分以客户端的形式写在了CartPoleBulletEnv这个类中
env = CartPoleBulletEnv(renders=True, discrete_actions=True)
env.render()
env.reset()

model = DQN(policy="MlpPolicy", env=env)
print("开始训练，稍等片刻")
model.learn(total_timesteps=10000)
model.save("./model")
# for _ in range(10000):
#     sleep(1 / 60)
#     action = env.action_space.sample()
#     obs, reward, done, _ = env.step(action)
# p.disconnect(cid)
