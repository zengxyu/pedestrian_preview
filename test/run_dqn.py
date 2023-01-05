import os

""" Deactivate all GPUs, explicitly activate below """
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gym

if __name__ == "__main__":
    # Initialize environments
    env = gym.make("HalfCheetah-v2")

    # Initialize GPUs without full memory allocation
    gpu_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Create networks
    critic = Critic(env.action_space.n, env.observation_space.shape, batch_size=128)
    critic_target = Critic(
        env.action_space.n, env.observation_space.shape, batch_size=128
    )

    """ Create actual agent instance """
    agent = CustomAgent(
        env=env,
        eval_env=eval_env,
        critic=critic,
        critic_target=critic_target,
    )

    """ Loop over each epoch """
    for news_ in agent.train():
        ...
