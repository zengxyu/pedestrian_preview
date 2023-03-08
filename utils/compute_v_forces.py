import os
import pickle
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

from utils.compute_u_forces import compute_u_force
from utils.fo_utility import get_project_path, get_p2v_path


def compute_v_force(occupancy_map, geo_distance_map, k1=0.01, k2=0.005, k3=1):
    force_v = k1 * geo_distance_map + k2 * geo_distance_map ** 2 + k3
    force_v[geo_distance_map > 39] = 11

    force_vx = np.zeros_like(force_v)
    force_vx = force_vx.astype(float)
    for i in range(0, force_v.shape[0]):
        for j in range(0, force_v.shape[1]):
            if occupancy_map[i][j] == 1 or occupancy_map[i][j - 1] == 1:
                continue
            force_vx[i][j] = np.sign(geo_distance_map[i][j - 1] - geo_distance_map[i][j]) * force_v[i][j]

    force_vy = np.zeros_like(force_v).astype(float)
    force_vy = force_vy.astype(float)

    for i in range(0, force_v.shape[0]):
        for j in range(0, force_v.shape[1]):
            if occupancy_map[i][j] == 1 or occupancy_map[i - 1][j] == 1:
                continue
            force_vy[i][j] = np.sign(geo_distance_map[i - 1][j] - geo_distance_map[i][j]) * force_v[i][j]

    return force_vx, force_vy, force_v


def compute_v_force_const(occupancy_map, geo_distance_map):
    """

    Args:
        occupancy_map: 占据栅格图
        geo_distance_map: 测地距离图

    Returns:

    """
    force_vx = np.zeros_like(occupancy_map).astype(float)
    force_vy = np.zeros_like(occupancy_map).astype(float)
    force_v = np.zeros_like(occupancy_map).astype(float)

    for i in range(0, force_v.shape[0]):
        for j in range(0, force_v.shape[1]):
            if occupancy_map[i][j] == 1 or occupancy_map[i][j - 1] == 1:
                continue
            force_vx[i][j] = np.sign(geo_distance_map[i][j - 1] - geo_distance_map[i][j]) * force_v[i][j]

    for i in range(0, force_v.shape[0]):
        for j in range(0, force_v.shape[1]):
            if occupancy_map[i][j] == 1 or occupancy_map[i - 1][j] == 1:
                continue
            force_vy[i][j] = np.sign(geo_distance_map[i - 1][j] - geo_distance_map[i][j]) * force_v[i][j]

    return force_vx, force_vy, force_v


if __name__ == '__main__':
    scene_path = os.path.join(get_p2v_path(), "train", "envs", "env_{}.pkl".format(0))
    geo_path = os.path.join(get_p2v_path(), "train", "envs", "env_{}.pkl".format(0))
    # scene_path = "/data/office_1500/train/envs/env_0.pkl"
    # geo_path = "/data/office_1500/train/geodesic_distance/env_0.pkl"
    potential_maps_path = "/data/office_1500/train/uv_forces/env_0.pkl"

    save_path = os.path.join(
        "/data/office_1500/train/geodesic_distance_images",
        "env_0.txt")

    occupancy_map, _, _ = pickle.load(open(scene_path, "rb"))
    obj = pickle.load(open(geo_path, "rb"))
    geo_distance_map = np.zeros_like(occupancy_map).astype(float)
    goal = (38, 9)
    geodesic_distance: Dict = obj[goal]

    for key in geodesic_distance.keys():
        distance = geodesic_distance[key]
        geo_distance_map[key] = distance

    force_v, force_vx, force_vy = compute_v_force(occupancy_map, geo_distance_map)
    # force_u_scalar_map, force_u_x, force_u_y = compute_u_force(potential_maps_path)

    # force_x = force_u_x + 0.1 * force_vx
    # force_y = force_u_y + 0.1 * force_vy
    XX, YY = np.meshgrid(np.arange(1, occupancy_map.shape[0] + 1), np.arange(1, occupancy_map.shape[1] + 1))
    plt.figure(100)
    plt.quiver(XX, YY, force_vx, force_vy)
    plt.title('Force V')
    plt.savefig("force_v.png")
    plt.show()

    plt.quiver(XX, YY, force_vx, force_vy)
    plt.title('Force U')
    plt.savefig("force_u.png")

    plt.show()

    plt.quiver(XX, YY, force_vx, force_vy)
    plt.title('Force')
    plt.savefig("force.png")

    plt.show()

    print()
