import os

from environment.circle_car import CircleCar

""" Deactivate all GPUs, explicitly activate below """
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import gym, pybulletgym, gym_hrl
from agents_hrl.agent_sac import SAC
from agents_hrl.networks.stochastic_actor_critic_small import Actor, Critic


class CustomAgent(SAC):
    """
    You can override or add functions.
    Useful for any kind of preprocessing
    Even additional networks can be added
    Saving and training does not apply to those
    """

    def __init__(self, extra_arg_a=None, **kwargs):
        super().__init__(**kwargs)


if __name__ == "__main__":
    # Initialize environments
    # gym_hrl._register()
    # env = gym.make("InvertedPendulumPyBulletEnv-v0")
    # eval_env = gym.make("InvertedPendulumPyBulletEnv-v0")
    env = CircleCar(render=False, human_control=False)
    # eval_env = CircleCar(render=True, human_control=False)
    # Initialize GPUs without full memory allocation
    # gpu_id = 0
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # for gpu in tf.config.experimental.list_physical_devices("GPU"):
    #     tf.config.experimental.set_memory_growth(gpu, True)

    # Create networks
    actor = Actor(
        env.action_space.shape[0], env.observation_space.shape, batch_size=128
    )
    critic = Critic(
        (env.action_space.shape[0] + env.observation_space.shape[0],), batch_size=128
    )

    """ Create actual agent instance """
    agent = CustomAgent(
        env=env,
        eval_env=None,
        actor=actor,
        critic=critic,
    )
    # for i in range(1000):
    #     agent.train()
    """ Loop over each epoch """
    for news_ in agent.train():
        ...
