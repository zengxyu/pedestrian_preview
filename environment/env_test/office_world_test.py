import os.path

from environment.sub_turtlebot_office_env import SubTurtlebotOfficeEnv
from environment.utils.gen_office_map import (
    configs_2021Mar11 as world_configs_office,
    funcs_2021Mar11 as world_create_funcs_office,
)

from environment.turtlebot_office_env import TurtlebotOfficeEnv

if __name__ == '__main__':
    save_om_dir = "output"
    if not os.path.exists(save_om_dir):
        os.makedirs(save_om_dir)
        print("create a directory:{}".format(save_om_dir))
    save_om_path = os.path.join(save_om_dir, "../../output/occupancy_map.obj")
    env = SubTurtlebotOfficeEnv(render=True,
                                schedule=False,
                                evaluation=True,
                                world_configs=world_configs_office,
                                world_create_funcs=world_create_funcs_office,
                                exact_complexity_level=True,
                                save_om=True,
                                save_om_path=save_om_path
                                )
