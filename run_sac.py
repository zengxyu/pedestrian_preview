#!/usr/bin/env python3
import logging
from agents.mapping import Env
from agents.pfrl_agents.agent_builder import build_agent
from config import process_args
from learner.pf_learner_sac import PFLearner
from utils.basic_logger import setup_logger
from utils.set_random_seed import set_random_seeds
from warnings import filterwarnings

filterwarnings(action='ignore', category=DeprecationWarning, message='`np.')
set_random_seeds(103)
args = process_args()

if args.render:
    setup_logger(log_level=logging.INFO)
else:
    setup_logger(log_level=logging.WARNING)

if __name__ == '__main__':
    if args.env == Env.Pybullet:
        from environment.environment_bullet import EnvironmentBullet
        agent, action_space = build_agent(args)
        env = EnvironmentBullet(args=args, action_space=action_space)
        learner = PFLearner(env=env, agent=agent, action_space=action_space, args=args).run()

    else:
        raise NotImplementedError
