#!/usr/bin/env python3
import logging
import sys
import time

from agents.mapping import Env
from agents.pfrl_agents.agent_builder import build_agent
from config import process_args
from learner.pf_learner import PFLearner
from utils.basic_logger import setup_logger
from utils.set_random_seed import set_random_seeds
from warnings import filterwarnings

from environment.environment_bullet import EnvironmentBullet

filterwarnings(action='ignore', category=DeprecationWarning, message='`np.')
set_random_seeds(10001)
args = process_args()

if args.render:
    setup_logger(log_level=logging.INFO)
else:
    setup_logger(log_level=logging.WARNING)

agent, action_space, scheduler, replay_buffer = build_agent(args)
env = EnvironmentBullet(args=args, action_space=action_space)
start_time = time.time()
for i in range(100):
    env.reset()

print("duration:{}".format(time.time() - start_time))
