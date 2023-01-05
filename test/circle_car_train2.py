from stable_baselines3 import TD3, PPO, SAC
import torch

from environment.circle_car import CircleCar

# policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=[dict(pi=[32, 64, 32], vf=[32, 64, 32])])
#
# policy_kwargs = dict(activation_fn=torch.nn.Tanh,
#                      net_arch=[dict(pi=[32, 64, 32], vf=[32, 64, 32])])
# environment._register()
# env = gym.make("circleCar-v0")
policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=dict(pi=[64, 128, 64], qf=[64, 128, 64]))
# policy_kwargs = dict(activation_fn=torch.nn.Tanh,
#                      net_arch=[dict(pi=[32, 64, 32], vf=[32, 64, 32])])

env = CircleCar(render=False, human_control=False)
env.render()
env.reset()

model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
print("开始训练，稍等片刻")
for i in range(100):
    model.learn(total_timesteps=100000)
    print("save==============================================")
    model.save("./sac_custom_model")
