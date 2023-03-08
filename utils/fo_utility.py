import os

import yaml


def get_office1500_goal_outdoor_path():
    return os.path.join(get_project_path(), "data", "office_1500_goal_outdoor")


def get_office1500_path():
    return os.path.join(get_project_path(), "data", "office_1500")


def get_p2v_path():
    return os.path.join(get_project_path(), "data", "p2v")


def get_data_path():
    return os.path.join(get_project_path(), "data")


def get_project_path():
    return os.path.dirname(os.path.dirname(__file__))


def get_local_data_path():
    return os.path.join(get_project_path(), "urdf")


def get_flat_path():
    return os.path.join(get_project_path(), "urdf", "flat")


def create_dirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        print("path:{} already exists, so not to create!".format(dir_path))
