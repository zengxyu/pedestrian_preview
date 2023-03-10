import os
import pickle
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

from utils.compute_u_forces import compute_u_force
from utils.compute_v_forces import compute_v_force
from utils.fo_utility import *


def compute_force_u(potential_maps_path):
    force_u_x, force_u_y, force_u_scalar = pickle.load(open(potential_maps_path, 'rb'))

    return force_u_scalar, force_u_x, force_u_y


def compute_force_v(potential_maps_path):
    force_v_x, force_v_y, force_v_scalar = pickle.load(open(potential_maps_path, 'rb'))

    return force_v_scalar, force_v_x, force_v_y


if __name__ == '__main__':
    folder_name = "sg_walls"
    phase = "train"
    index = 0
    env_path = os.path.join(get_office_evacuation_path(), folder_name, phase, "envs", "env_{}.pkl".format(index))
    u_path = os.path.join(get_office_evacuation_path(), folder_name, phase, "u_forces", "env_{}.pkl".format(index))
    v_path = os.path.join(get_office_evacuation_path(), folder_name, phase, "v_forces", "env_{}.pkl".format(index))
    # scene_path = "/data/office_1500/train/envs/env_0.pkl"
    # geo_path = "/data/office_1500/train/geodesic_distance/env_0.pkl"
    # potential_maps_path = "/data/office_1500/train/uv_forces/env_0.pkl"

    save_path = os.path.join(
        "/data/office_1500/train/geodesic_distance_images",
        "env_0.txt")

    occupancy_map, _, _ = pickle.load(open(env_path, "rb"))
    force_ux, force_uy, force_u = pickle.load(open(u_path, "rb"))
    v_force_map_dict = pickle.load(open(v_path, "rb"))

    goal = (38, 9)
    force_vx, force_vy, force_v = v_force_map_dict[tuple(goal)]
    w = 0.1
    force_x = force_ux + w * force_vx
    force_y = force_uy + w * force_vy
    XX, YY = np.meshgrid(np.arange(1, occupancy_map.shape[0] + 1), np.arange(1, occupancy_map.shape[1] + 1))
    plt.figure(100)
    plt.quiver(XX, YY, force_vx, force_vy)
    plt.title('Force V')
    plt.savefig("force_v.png")
    plt.show()

    plt.quiver(XX, YY, force_ux, force_uy)
    plt.title('Force U')
    plt.savefig("force_u.png")

    plt.show()

    plt.quiver(XX, YY, force_x, force_y)
    plt.title('Force')
    plt.savefig("force.png")

    plt.show()

    print()
