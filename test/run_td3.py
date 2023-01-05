import os

from environment.circle_car import CircleCar

""" Deactivate all GPUs, explicitly activate below """
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import gym, pybulletgym
import environment
from agents_hrl.agent_td3 import TD3
from agents_hrl.networks.actor_critic_small import Actor, Critic


class CustomAgent(TD3):
    """
    You can override or add functions by again subclassing the agent
    Useful for any kind of preprocessing
    Even additional networks can be added
    Saving and training does not apply to those
    """

    def __init__(self, extra_arg_a=None, **kwargs):
        super().__init__(**kwargs)


if __name__ == "__main__":
    # Initialize environments
    # env = CircleCar(render=False, human_control=False)
    # eval_env = CircleCar(render=True, human_control=False)

    environment._register()
    #
    env = gym.make("turtlebotFlat-v0")
    eval_env = gym.make("turtlebotFlatGUI-v0")

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
        eval_env=eval_env,
        actor=actor,
        critic=critic,
    )

    """ Loop over each epoch """
    for news_ in agent.train():
        ...
