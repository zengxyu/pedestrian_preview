#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 3/29/22 9:53 PM 
    @Description    :
        
===========================================
"""
import os
import shutil
import logging
import yaml
import sys


def create_folders(folders):
    """create out folder"""
    for folder in folders:
        if not os.path.exists(folder):
            print("Create folder:{}", folder)
            os.makedirs(folder)


def check_folders_exist(folders):
    """check if folders exist"""
    for folder in folders:
        assert os.path.exists(folder), "Path to folder : {} not exist!".format(folder)


def get_log_level(name):
    log_level_mapping = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO,
                         "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
    return log_level_mapping.get(name)


def read_yaml(config_dir, config_name):
    yaml_path = os.path.join(config_dir, config_name)

    # read configs from yaml path
    if not os.path.exists(yaml_path):
        logging.error("yaml_abs_path : {} not exist".format(yaml_path))

    with open(yaml_path, 'r') as f:
        yaml_config = yaml.load(f, Loader=yaml.SafeLoader)
    return yaml_config


def copy_configs_to_folder(from_folder, to_folder, from_configs="configs"):
    from_folder = os.path.join(from_folder, from_configs)
    to_folder = os.path.join(to_folder, "configs")
    if not os.path.exists(to_folder):
        shutil.copytree(from_folder, to_folder)
    else:
        logging.info("File exists:{}".format(to_folder))
        key = input(
            "Output directory already exists! \nFrom {} to {}. \nOverwrite the folder? (y/n).".format(from_folder,
                                                                                                      to_folder))
        if key.lower() == 'y':
            shutil.rmtree(to_folder)
            shutil.copytree(from_folder, to_folder)
        else:
            logging.info("Please respecify the folder.")

            sys.exit(1)


def setup_folder(parser_args):
    """
    config all output folders and input folders, setup model folder, board folder, result_folder
    :param parser_args:
    :return:
    """
    # out_folder given by parser_args
    parser_args.out_model = os.path.join(parser_args.out_folder, "model")
    parser_args.out_board = os.path.join(parser_args.out_folder, "board_log")
    create_folders(
        [parser_args.out_folder, parser_args.out_model, parser_args.out_board])

    if not parser_args.train:
        if parser_args.in_folder is not None and parser_args.in_folder != "":
            parser_args.in_model = os.path.join(parser_args.in_folder, "model")
            check_folders_exist([parser_args.in_folder, parser_args.in_model])
