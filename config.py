import argparse
import os.path

import numpy as np

from utils.config_utility import *
from utils.fo_utility import get_project_path


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_folder", type=str, default="test_folder")
    parser.add_argument("--in_folder", type=str, default=None)
    parser.add_argument("--in_model_index", type=int)

    parser.add_argument("--num_episodes", type=int, default=50000)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--render", action="store_true", default=False)

    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--from_configs", type=str, default="configs")
    parser.add_argument("--gpu", type=int, default=0, help="gpu >=0 : use gpu; gpu <0 : use cpu")
    parser.add_argument('--scene_name', type=str, help='')
    parser.add_argument("--max_speed", type=float, help='')
    parser.add_argument("--dynamic_num", type=int, help='')
    parser.add_argument("--static_num", type=int, help='')
    parser.add_argument("--max_steps", type=int, help='')
    parser.add_argument("--load_coordinates_from", type=str)
    parser.add_argument("--load_map_from", type=str)

    parser.add_argument("--state", type=str)
    
    # motion/motion_speed_control_0.1/model/model_epi_1000
    parser_args = parser.parse_args()
    parser_args.out_folder = os.path.join(get_project_path(), "output", parser_args.out_folder)

    # set out_folder path and in_folder_path
    if not parser_args.train or parser_args.resume:
        assert parser_args.in_folder and parser_args.in_model_index
        parser_args.in_folder = os.path.join(get_project_path(), "output", parser_args.in_folder)

    # config dir
    if parser_args.train:
        # copy configs dir from /project_path/configs to /project_path/output/out_folder/configs
        copy_configs_to_folder(from_folder=get_project_path(), from_configs=parser_args.from_configs,
                               to_folder=parser_args.out_folder)
        configs_folder = os.path.join(parser_args.out_folder, "configs")
    elif parser_args.resume:
        copy_configs_to_folder(from_folder=parser_args.in_folder, to_folder=parser_args.out_folder)
        configs_folder = os.path.join(parser_args.in_folder, "configs")
    else:
        # copy configs dir from /project_path/output/in_folder/configs to /project_path/output/out_folder/configs
        # copy_configs_to_folder(from_folder=parser_args.in_folder, to_folder=parser_args.out_folder)
        configs_folder = os.path.join(parser_args.in_folder, "configs")

    setup_folder(parser_args)

    # load some yaml files
    # configs_folder = os.path.join(parser_args.out_folder, "configs")
    parser_args.configs_folder = configs_folder
    parser_args.agents_config_folder = os.path.join(configs_folder, "agents_config")

    parser_args.agents_config = read_yaml(parser_args.agents_config_folder, "agents.yaml")
    parser_args.inputs_config = read_yaml(parser_args.configs_folder, "inputs_config.yaml")
    parser_args.action_spaces_config = read_yaml(parser_args.configs_folder, "action_spaces_config.yaml")
    parser_args.robots_config = read_yaml(parser_args.configs_folder, "robots_config.yaml")
    parser_args.worlds_config = read_yaml(parser_args.configs_folder, "worlds_config.yaml")
    parser_args.sensors_config = read_yaml(parser_args.configs_folder, "sensors_config.yaml")
    parser_args.env_config = read_yaml(parser_args.configs_folder, "env_config.yaml")
    parser_args.running_config = read_yaml(parser_args.configs_folder, "running_rl.yaml")

    # evaluation时动态配置的环境参数，
    if not parser_args.train or parser_args.resume:
        if parser_args.scene_name is not None:
            if parser_args.env_config["scene_name"] == "random":
                scene_name = np.random.choice(a=["office", "corridor", "cross"])
            else:
                scene_name = parser_args.scene_name
            parser_args.env_config["scene_name"] = scene_name
        if parser_args.max_steps is not None:
            parser_args.running_config["max_steps"] = parser_args.max_steps
        if parser_args.max_speed is not None:
            parser_args.env_config["pedestrian_speed_range"] = [parser_args.max_speed - 0.01,
                                                                parser_args.max_speed]

        if parser_args.dynamic_num is not None:
            parser_args.env_config["pedestrian_dynamic_num"] = parser_args.dynamic_num

        if parser_args.static_num is not None:
            parser_args.env_config["pedestrian_static_num"] = parser_args.static_num

    print("\nYaml env_config config:", parser_args.env_config)
    print("\nYaml training config:", parser_args.running_config)
    print("\n==============================================================================================\n")
    return parser_args
