import os.path

from environment.utils.gen_turtlebot_world import (
    configs_2021Jan05 as world_configs_office,
    funcs_2021Jan05 as world_create_funcs_office,
)

from environment.turtlebot_flat_env import TurtlebotFlatEnv

if __name__ == '__main__':
    save_om_dir = "output"
    if not os.path.exists(save_om_dir):
        os.makedirs(save_om_dir)
        print("create a directory:{}".format(save_om_dir))
    save_om_path = os.path.join(save_om_dir, "../../output/occupancy_map.obj")
    env = TurtlebotFlatEnv(render=True,
                           schedule=False,
                           evaluation=True,
                           world_configs=world_configs_office,
                           world_create_funcs=world_create_funcs_office,
                           exact_complexity_level=True,
                           save_om=True,
                           save_om_path=save_om_path
                           )
    print()
