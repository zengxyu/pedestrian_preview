import os.path

from configs.env_config import EnvConfig
from environment.nav_utilities.coordinates_converter import cvt_to_bu
from environment.nav_utilities.debug_plot_helper import plot_bullet_path
from environment.turtlebot_office_environment import TurtleBotOfficeEnv
import pickle

# TODO 激光返回的结果可视化
from utils.fo_utility import get_project_path


def plan_global_path(env):
    # global planner use a star algorithm to plan a global path to goal
    path_save_to_dir = "output/path"

    # om_path = g_planner.plan(Planner.A_star, env.dilated_occ_map, env.s_om_pose, env.g_om_pose, save_to_dir=path_save_to_dir)
    om_path = pickle.load(open(os.path.join(path_save_to_dir, "path.obj"), "rb"))
    bullet_path = cvt_to_bu(om_path, env.grid_res)
    bullet_path = bullet_path[10:]
    # bullet_path = [[2.75, 2.75], [3, 3], [3.25, 3.75], [3.5, 4], [3.75, 4.25], [6.75, 5.25]]
    plot_bullet_path(env.p, bullet_path)
    # env.start(bullet_path)
    env.path_tracking(bullet_path)


# load yaml config
yaml_path = os.path.join(get_project_path(), "configs/env_default.yaml")
configs = EnvConfig().load_configs(yaml_path)

# initialize environment
env = TurtleBotOfficeEnv(render=True, configs=configs)
# env.reset()
plan_global_path(env)
