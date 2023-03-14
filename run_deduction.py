"""
输入：场景地图、所有行人起点，时间参数
输出：从开始时刻到最后时刻t_last中，每一个时刻的所有人的坐标（2D），以及t_last的值
"""

# !/usr/bin/env python3
import logging
import os

from tqdm import tqdm

from agents.pfrl_agents.agent_builder import build_agent
from config import process_args
from learner.pf_learner import add_statistics_to_collector
from utils.basic_logger import setup_logger
from utils.info import EpisodeInfo
from utils.set_random_seed import set_random_seeds
from warnings import filterwarnings

from environment.environment_bullet import EnvironmentBullet, Phase


class PFRunDeductioner:
    def __init__(self, args, env, agent):
        self.args = args
        self.env = env
        self.agent = agent
        self.test_collector = EpisodeInfo()

    def run_deduction(self, occupancy_map, starts, ends):
        """
        加载环境
        Args:
            occupancy_map:
            starts:
            ends:

        Returns:

        """

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
        add_statistics_to_collector(infos_episode_for_sum=infos_for_sum,
                                    infos_episode_for_last=infos_for_last,
                                    agent_statistics=agent.get_statistics(),
                                    episode_info_collector=self.test_collector)
        print("===================================success rate:{}".format(self.test_collector.get_success_rate()))
        pbar.set_description("Success rate:{}".format(self.test_collector.get_success_rate()))
        for robot in env.agent_robots_copy:
            trajectories = robot.bridge.trajectories
            times = robot.bridge.times
            print("trajectories:{}".format(trajectories))
            print("times:{}".format(times))
        print()

        return


def load_args_and_configs():
    filterwarnings(action='ignore', category=DeprecationWarning, message='`np.')
    set_random_seeds(500001)
    args = process_args()
    if args.render:
        setup_logger(log_level=logging.INFO)
    else:
        setup_logger(log_level=logging.WARNING)
    return args


def load_env():
    agent, action_space, scheduler, replay_buffer = build_agent(args)
    env: EnvironmentBullet = EnvironmentBullet(args=args, action_space=action_space)
    env.phase = Phase.TEST

    # 加载模型
    model_path = os.path.join(args.in_folder, "model", "model_epi_{}".format(args.in_model_index))
    agent.load(model_path)
    return env, agent


if __name__ == '__main__':
    args = load_args_and_configs()
    env, agent = load_env()
    run_deductioner = PFRunDeductioner(args, env, agent)
    pbar = tqdm(range(200))

    for i in pbar:
        run_deductioner.run_deduction(None, None, None)
    print()
