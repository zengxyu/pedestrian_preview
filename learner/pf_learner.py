import json
import logging
import os
from typing import List, Dict

import numpy as np

from torch.utils.tensorboard import SummaryWriter

from learner.trainer_helper import add_scalar, save_episodes_info, get_items
from utils.info import EpisodeInfo
from tqdm import tqdm


class PFLearner:
    def __init__(self, env, agent, action_space, scheduler, args):
        self.args = args
        self.running_config = args.running_config
        self.writer = SummaryWriter(log_dir=args.out_board)
        self.env = env
        self.agent = agent
        self.action_space = action_space
        self.scheduler = scheduler

        self.render = self.args.render
        self.train_i_episode = 0
        self.test_i_episode = 0
        self.global_i_step = 0

        self.train_collector = EpisodeInfo()
        self.test_collector = EpisodeInfo()
        self.eval = not args.train

    def run(self):
        print("========================================Start running========================================")
        if self.eval or self.args.resume:
            logging.info("load model from {} {}".format(self.args.in_model, self.args.in_model_index))
            self.agent.load("{}/model_epi_{}".format(self.args.in_model, self.args.in_model_index))

        if self.args.train or self.args.resume:
            print("Start training")
            for i in range(self.args.num_episodes):
                print("\nEpisode:{}".format(i))
                self.scheduler.print()
                self.train_once()

                if (i + 1) % self.running_config["evaluate_every_n_training"] == 0:
                    print("\nTest Episode:{}".format(i))
                    self.evaluate_n_times(self.running_config["evaluate_n_times"])
                    self.agent.save("{}/model_epi_{}".format(self.args.out_model, self.test_i_episode))
                    self.scheduler.lr_schedule(self.test_collector.get_smooth_success_rate())

                # if (i + 1) % self.running_config["learn_segmentation_every_n_training"] == 0:
                #     self.segmentation_learner.run()

        else:
            print("Start evaluating")
            success_num = 0
            collision_num = 0
            timeout_num = 0
            navigation_time_on_success_episodes = {}

            pbar = tqdm(range(self.args.num_episodes))
            for i in pbar:
                info = self.evaluate_once()
                # if info['a_success']:
                #     success_num += 1
                # elif info["collision"]:
                #     collision_num += 1
                # else:
                #     timeout_num += 1
                # navigation_time_on_success_episodes["{}".format(i)] = info['step_count']

                pbar.set_description("Success rate:{}".format(self.test_collector.get_success_rate()))

            test_result = {
                "success": success_num,
                "success_rate": success_num / self.args.num_episodes,
                "collision": collision_num,
                "collision_rate": collision_num / self.args.num_episodes,
                "timeout": timeout_num,
                "timeout_rate": timeout_num / self.args.num_episodes,
                "navigation_time": navigation_time_on_success_episodes,
            }

            save_folder = os.path.join(self.args.out_folder, self.env.get_env_types())
            os.makedirs(save_folder, exist_ok=True)
            # store test result
            filename = os.path.join(
                save_folder,
                "dynamic_{}+static_{}_speed_{}.".format(self.args.dynamic_num, self.args.static_num,
                                                        self.args.max_speed)
            )
            with open(filename + "json", "w") as f:
                json.dump(test_result, f)
            self.env.writer.close()
            self.agent.writer.close()

    def train_once(self):
        phase = "Train"
        self.train_i_episode += 1
        state = self.env.reset()
        infos_for_sum = []
        infos_for_last = []
        done = False
        i_step = 0
        while not done:
            action = self.agent.act(state[0])
            state, reward, done, info_for_sum, info_for_last = self.env.step(action)
            self.agent.observe(obs=state[0], reward=reward, done=done, reset=False)
            self.global_i_step += 1

            i_step += 1
            infos_for_sum.append(info_for_sum)
            infos_for_last.append(info_for_last)

        add_statistics_to_collector(infos_episode_for_sum=infos_for_sum,
                                    infos_episode_for_last=infos_for_last,
                                    agent_statistics=self.agent.get_statistics(),
                                    episode_info_collector=self.train_collector)
        if self.train_i_episode % 100 == 0:
            add_scalar(self.writer, phase, self.train_collector.get_smooth_n_statistics(n=100), self.train_i_episode)
            save_episodes_info(phase, self.train_collector, self.train_i_episode, self.args)

    def evaluate_once(self):
        phase = "ZEvaluation"
        self.test_i_episode += 1
        state = self.env.reset()

        done = False
        infos_for_sum = []
        infos_for_last = []
        i_step = 0
        with self.agent.eval_mode():
            while not done:
                actions = self.agent.batch_act(state)
                state, reward, done, info_for_sum, info_for_last = self.env.step(actions[0])
                self.agent.observe(obs=state, reward=reward, done=done, reset=False)
                self.global_i_step += 1
                i_step += 1
                infos_for_sum.append(info_for_sum)
                infos_for_last.append(info_for_last)
        add_statistics_to_collector(infos_episode_for_sum=infos_for_sum,
                                    infos_episode_for_last=infos_for_last,
                                    agent_statistics=self.agent.get_statistics(),
                                    episode_info_collector=self.test_collector)

        logging.info('Complete evaluation episode {}'.format(self.test_i_episode))

        return info_for_last

    def evaluate_n_times(self, n_times):
        phase = "ZEvaluation"
        for i in range(n_times):
            self.evaluate_once()

        add_scalar(self.writer, phase, self.test_collector.get_smooth_n_statistics(n=n_times), self.test_i_episode)
        save_episodes_info(phase, self.test_collector, self.test_i_episode, self.args)
        logging.info('Complete evaluation episode {}'.format(self.test_i_episode))


def add_statistics_to_collector(infos_episode_for_sum: List[Dict],
                                infos_episode_for_last: List[Dict],
                                agent_statistics,
                                episode_info_collector: EpisodeInfo
                                ):
    new_infos_episode_for_sum: Dict[List] = get_items(infos_episode_for_sum)
    new_infos_episode_for_last: Dict[List] = get_items(infos_episode_for_last)

    for key, item in new_infos_episode_for_sum.items():
        episode_info_collector.add({key: np.mean(new_infos_episode_for_sum[key])})

    for key, item in new_infos_episode_for_last.items():
        episode_info_collector.add({key: new_infos_episode_for_last[key][-1]})

    if not np.isnan(agent_statistics[0][1]):
        episode_info_collector.add({"a_average_q": agent_statistics[0][1]})
        episode_info_collector.add({"a_loss": agent_statistics[1][1]})
