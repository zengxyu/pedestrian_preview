"""
输入：场景地图、所有行人起点，时间参数
输出：从开始时刻到最后时刻t_last中，每一个时刻的所有人的坐标（2D），以及t_last的值
"""

# !/usr/bin/env python3
import logging
import os

from agents.pfrl_agents.agent_builder import build_agent
from config import process_args
from utils.basic_logger import setup_logger
from utils.set_random_seed import set_random_seeds
from warnings import filterwarnings

from environment.environment_bullet import EnvironmentBullet, Phase


def run_deduction(args, occupancy_map, starts, ends):
    """
    加载环境
    Args:
        occupancy_map:
        starts:
        ends:

    Returns:

    """

    agent, action_space, scheduler, replay_buffer = build_agent(args)
    env: EnvironmentBullet = EnvironmentBullet(args=args, action_space=action_space)
    env.phase = Phase.TEST

    # 加载模型
    model_path = os.path.join(args.in_folder, "model", "model_epi_{}".format(args.in_model_index))
    agent.load(model_path)

    state = env.reset()
    done = False
    infos_for_sum = []
    infos_for_last = []
    global_i_step = 0

    with agent.eval_mode():
        while not done:
            actions = agent.batch_act(state)
            state, reward, done, info_for_sum, info_for_last = env.step(actions)
            global_i_step += 1
            infos_for_sum.append(info_for_sum)
            infos_for_last.append(info_for_last)
            # 加到轨迹中
            for robot in env.agent_robots:
                robot.bridge.add_to_trajectories()
    trajectories = robot.bridge.trajectories
    times = robot.bridge.times
    return trajectories, times


def load_args_and_configs():
    filterwarnings(action='ignore', category=DeprecationWarning, message='`np.')
    set_random_seeds(500001)
    args = process_args()
    if args.render:
        setup_logger(log_level=logging.INFO)
    else:
        setup_logger(log_level=logging.WARNING)
    return args


if __name__ == '__main__':
    args = load_args_and_configs()
    # occupancy_map, starts, ends =
    run_deduction(args, None, None, None)
