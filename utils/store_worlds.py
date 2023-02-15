import os

import matplotlib.pyplot as plt
import numpy as np

from environment.gen_scene.sampler_mapping import get_sampler_class
from environment.gen_scene.world_generator import get_world_config
from environment.gen_scene.worlds_mapping import get_world_creator_func
from utils.config_utility import read_yaml
from utils.fo_utility import get_project_path
from utils.image_utility import dilate_image


def generate_world(num_starts):
    # read specified world config
    world_name = "office"
    worlds_config = read_yaml(os.path.join(get_project_path(), "configs"), "worlds_config.yaml")
    world_config = get_world_config(worlds_config, world_name)

    # read specified sampler config
    agent_sampler_name = "agent_sg_opposite_baffle_sampler1"
    samplers_config = read_yaml(os.path.join(get_project_path(), "configs"), "samplers_config.yaml")
    agent_sampler_config = samplers_config[agent_sampler_name]
    agent_sg_sampler_class = get_sampler_class(agent_sampler_config["sampler_name"])
    agent_sg_sampler_params = agent_sampler_config["sampler_params"]

    # create world occupancy map
    create_world = get_world_creator_func(world_name)
    occupancy_map = create_world(world_config)
    # dilate occupancy map
    dilated_occ_map = dilate_image(occupancy_map, dilation_size=10)

    starts = []
    ends = []
    while len(starts) < num_starts:
        sample_success = False
        while not sample_success:
            # sample start and goal
            [start, end], sample_success = agent_sg_sampler_class(dilate_occupancy_map=dilated_occ_map,
                                                                  occupancy_map=occupancy_map,
                                                                  **agent_sg_sampler_params)
        starts.append(start)
        ends.append(end)
    return occupancy_map, np.array(starts), np.array(ends)


def display(occupancy_map, starts, ends):
    plt.imshow(occupancy_map)
    plt.scatter(starts[:, 0], starts[:, 1], c='g')
    plt.scatter(ends[:, 0], ends[:, 1], c='g')
    for s, e in zip(starts, ends):
        plt.plot([s[0], e[0]], [s[1], e[1]])
    plt.show()


if __name__ == '__main__':
    occupancy_map, starts, ends = generate_world(num_starts=20)
    display(occupancy_map, starts, ends)
