import logging
from typing import Dict

import numpy as np

import pfrl
from agents.mapping import get_agent_name, get_input_config
from agents.pfrl_agents.action_builder import build_action_space
from agents.pfrl_agents.network_builder import build_network
from agents.pfrl_agents.scheduler_builder import get_scheduler, SchedulerHandler
from pfrl.agents import DoubleDQN, DDPG, TD3

from agents.pfrl_agents.explorer_builder import get_explorer_by_name
from agents.pfrl_agents.optimizer_builder import get_optimizer_by_name
from agents.pfrl_agents.replay_buffer_builder import get_replay_buffer_by_name


def build_agent(args):
    action_space = build_action_space(args)
    input_kwargs: Dict = get_input_config(args)
    agent = None
    agent_name = get_agent_name(parser_args=args)

    scheduler, replay_buffer = None, None

    if agent_name in ["ddpg", "ddpg_recurrent"]:
        actor_network, critic_network = build_network(parser_args=args, action_space=action_space,
                                                      input_kwargs=input_kwargs)
        agent, scheduler, replay_buffer = build_ddpg_agent(args, actor_network, critic_network, agent_name,
                                                           action_space)
        args.actor_network = actor_network
        logging.info("actor_network:{}, critic_network:{}".format(actor_network, critic_network))

    elif agent_name in ["td3", "td3_recurrent"]:
        actor_network, critic_network1, critic_network2 = build_network(parser_args=args, action_space=action_space,
                                                                        input_kwargs=input_kwargs)
        agent, scheduler, replay_buffer = build_td3_agent(args, actor_network, critic_network1, critic_network2,
                                                          agent_name, action_space)
        logging.info("actor_network:{}, critic_network1:{}, critic_network2:{}".format(actor_network, critic_network1,
                                                                                       critic_network2))
    elif agent_name in ["sac"]:
        actor_network, critic_network1, critic_network2 = build_network(parser_args=args, action_space=action_space,
                                                                        input_kwargs=input_kwargs)

        logging.info("actor_network:{}, critic_network1:{}, critic_network2:{}".format(actor_network, critic_network1,
                                                                                       critic_network2))
        agent, scheduler, replay_buffer = build_sac_agent(args, actor_network, critic_network1, critic_network2,
                                                          agent_name, action_space)
    return agent, action_space, scheduler, replay_buffer


def build_ddpg_agent(parser_args, policy, q_func, agent_name, action_space):
    agent_config = parser_args.agents_config[agent_name]

    optimizer_actor = get_optimizer_by_name(parser_args, agent_config["optimizer"], policy)
    optimizer_critic = get_optimizer_by_name(parser_args, agent_config["optimizer"], q_func)

    scheduler = SchedulerHandler(parser_args, agent_config["scheduler"], [optimizer_actor, optimizer_critic])

    explorer = get_explorer_by_name(parser_args, agent_config["explorer"], action_space=action_space)

    replay_buffer = get_replay_buffer_by_name(parser_args, agent_config["replay_buffer"])

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        random_action = np.random.uniform(action_space.low, action_space.high).astype(np.float32)
        return random_action

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    agent = DDPG(
        policy,
        q_func,
        optimizer_actor,
        optimizer_critic,
        replay_buffer,
        gamma=agent_config["discount_rate"],
        explorer=explorer,
        replay_start_size=agent_config["replay_start_size"],
        target_update_method=agent_config["target_update_method"],
        target_update_interval=agent_config["target_update_interval"],
        update_interval=agent_config["update_interval"],
        soft_update_tau=agent_config["soft_update_tau"],
        n_times_update=agent_config["n_times_update"],
        gpu=parser_args.gpu,
        minibatch_size=agent_config["batch_size"],
        burnin_action_func=burnin_action_func,
    )
    return agent, scheduler, replay_buffer


def build_td3_agent(parser_args, policy, q_func1, q_func2, agent_name, action_space):
    agent_config = parser_args.agents_config[agent_name]
    optimizer_actor = get_optimizer_by_name(parser_args, agent_config["optimizer"], policy)
    optimizer_critic1 = get_optimizer_by_name(parser_args, agent_config["optimizer"], q_func1)
    optimizer_critic2 = get_optimizer_by_name(parser_args, agent_config["optimizer"], q_func2)
    explorer = get_explorer_by_name(parser_args, agent_config["explorer"], action_space=action_space)
    scheduler = SchedulerHandler(parser_args, agent_config["scheduler"],
                                 [optimizer_actor, optimizer_critic1, optimizer_critic2])

    replay_buffer = get_replay_buffer_by_name(parser_args, agent_config["replay_buffer"])

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_space.low, action_space.high).astype(np.float32)

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    agent = TD3(
        policy,
        q_func1,
        q_func2,
        optimizer_actor,
        optimizer_critic1,
        optimizer_critic2,
        replay_buffer,
        gamma=agent_config["discount_rate"],
        soft_update_tau=agent_config["soft_update_tau"],
        explorer=explorer,
        replay_start_size=agent_config["replay_start_size"],
        gpu=parser_args.gpu,
        minibatch_size=agent_config["batch_size"],
        burnin_action_func=burnin_action_func,
    )
    return agent, scheduler, replay_buffer


def build_sac_agent(parser_args, policy, q_func1, q_func2, agent_name, action_space):
    agent_config = parser_args.agents_config[agent_name]
    optimizer_actor = get_optimizer_by_name(parser_args, agent_config["optimizer"], policy)
    optimizer_critic1 = get_optimizer_by_name(parser_args, agent_config["optimizer"], q_func1)
    optimizer_critic2 = get_optimizer_by_name(parser_args, agent_config["optimizer"], q_func2)
    scheduler = SchedulerHandler(parser_args, agent_config["scheduler"],
                                 [optimizer_actor, optimizer_critic1, optimizer_critic2])

    replay_buffer = get_replay_buffer_by_name(parser_args, agent_config["replay_buffer"])

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_space.low, action_space.high).astype(np.float32)

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    agent = pfrl.agents.SoftActorCritic(
        policy,
        q_func1,
        q_func2,
        optimizer_actor,
        optimizer_critic1,
        optimizer_critic2,
        replay_buffer,
        gamma=agent_config["discount_rate"],
        replay_start_size=agent_config["replay_start_size"],
        gpu=parser_args.gpu,
        minibatch_size=agent_config["batch_size"],
        burnin_action_func=burnin_action_func,
        entropy_target=-len(action_space.low),
        temperature_optimizer_lr=agent_config["temperature_optimizer_lr"],
    )

    return agent, scheduler, replay_buffer
