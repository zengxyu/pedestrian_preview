import os

import matplotlib.pyplot as plt

from config import process_args
from environment.gen_scene.common_sampler import *
from environment.gen_scene.gen_map_util import make_exit_door
from utils.config_utility import read_yaml
from utils.fo_utility import get_project_path


def create_u_shape_map(configs):
    grid_resolution = 0.1
    # 场景大小
    random_index = np.random.randint(0, len(configs["outer_range"]))
    outer_range = configs["outer_range"][random_index]
    min_outer_length = int(outer_range[0] / grid_resolution)
    max_outer_length = int(outer_range[1] / grid_resolution)
    outer_length = np.random.randint(min_outer_length, max_outer_length) if min_outer_length < max_outer_length else min_outer_length

    # U型墙的长
    wall_range = configs["wall_range"]
    min_wall_length = int(wall_range[0] / grid_resolution)
    max_wall_length = int(wall_range[1] / grid_resolution)
    wall_length = np.random.randint(min_wall_length, max_wall_length)

    # U型墙壁内部区域宽
    hall_range = configs["hall_range"]
    min_hall_width = int(hall_range[0] / grid_resolution)
    max_hall_width = int(hall_range[1] / grid_resolution)

    # 墙体厚度
    thick_range = configs["thick_range"]
    min_thick = int(thick_range[0] / grid_resolution)
    max_thick = int(thick_range[1] / grid_resolution)
    thick = np.random.randint(min_thick, max_thick)

    # 占据栅格
    occupancy_map = np.zeros((outer_length, outer_length), dtype=bool)

    # U型墙壁从哪边墙壁长出
    direction = np.random.choice([0, 1, 2, 3])

    # 生成U型区域走廊两边墙壁的位置
    ind_xy1 = np.random.randint(thick, outer_length - thick)
    ind_xy2 = np.random.randint(thick, outer_length - thick)

    while not min_hall_width < abs(ind_xy1 - ind_xy2) < max_hall_width:
        ind_xy1 = np.random.randint(thick, outer_length - thick)
        ind_xy2 = np.random.randint(thick, outer_length - thick)

    #
    half_thick = int(thick / 2)

    if direction == 0:
        occupancy_map[ind_xy1 - half_thick:ind_xy1 + half_thick + 1, :wall_length] = True
        occupancy_map[ind_xy2 - half_thick:ind_xy2 + half_thick + 1, :wall_length] = True
    elif direction == 2:
        occupancy_map[ind_xy1 - half_thick:ind_xy1 + half_thick + 1, -wall_length:] = True
        occupancy_map[ind_xy2 - half_thick:ind_xy2 + half_thick + 1, -wall_length:] = True
    elif direction == 1:
        occupancy_map[:wall_length, ind_xy1 - half_thick:ind_xy1 + half_thick + 1] = True
        occupancy_map[:wall_length, ind_xy2 - half_thick:ind_xy2 + half_thick + 1] = True
    else:
        occupancy_map[-wall_length:, ind_xy1 - half_thick:ind_xy1 + half_thick + 1] = True
        occupancy_map[-wall_length:, ind_xy2 - half_thick:ind_xy2 + half_thick + 1] = True
    # 设置外围围墙
    occupancy_map[:, [0, -1]] = True
    occupancy_map[[0, -1], :] = True
    # 开门
    make_exit_door(occupancy_map, configs, grid_resolution)
    return occupancy_map


if __name__ == '__main__':
    from environment.gen_scene.world_generator import get_world_config

    world_name = "u_shape"
    worlds_config = read_yaml(os.path.join(get_project_path(), "configs"), "worlds_config.yaml")
    world_config = get_world_config(worlds_config, world_name)

    for i in range(100):
        occupancy_map = create_u_shape_map(world_config)
        plt.imshow(occupancy_map)
        plt.show()
        print()
