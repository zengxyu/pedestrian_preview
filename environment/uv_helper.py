import os
import pickle
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt


def compute_force_v(occupancy_map, geo_distance_map, k1=0.01, k2=0.005, k3=1):
    force_v_scalar = k1 * geo_distance_map + k2 * geo_distance_map ** 2 + k3
    force_v_scalar[geo_distance_map > 39] = 11

    force_v_x = np.zeros_like(force_v_scalar)
    force_v_x = force_v_x.astype(float)
    for i in range(0, force_v_scalar.shape[0]):
        for j in range(0, force_v_scalar.shape[1]):
            if occupancy_map[i][j] == 1 or occupancy_map[i][j - 1] == 1:
                continue
            force_v_x[i][j] = np.sign(geo_distance_map[i][j - 1] - geo_distance_map[i][j]) * force_v_scalar[i][j]

    force_v_y = np.zeros_like(force_v_scalar).astype(float)
    force_v_y = force_v_y.astype(float)

    for i in range(0, force_v_scalar.shape[0]):
        for j in range(0, force_v_scalar.shape[1]):
            if occupancy_map[i][j] == 1 or occupancy_map[i - 1][j] == 1:
                continue
            force_v_y[i][j] = np.sign(geo_distance_map[i - 1][j] - geo_distance_map[i][j]) * force_v_scalar[i][j]

    return force_v_scalar, force_v_x, force_v_y


def compute_force_u(potential_maps_path):
    force_u_x, force_u_y, force_u_scalar = pickle.load(open(potential_maps_path, 'rb'))

    return force_u_scalar, force_u_x, force_u_y


if __name__ == '__main__':
    scene_path = "/data/office_1500/train/envs/env_0.pkl"
    geo_path = "/data/office_1500/train/geodesic_distance/env_0.pkl"
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

    force_v_scalar_map, force_v_x, force_v_y = compute_force_v(occupancy_map, geo_distance_map)
    force_u_scalar_map, force_u_x, force_u_y = compute_force_u(potential_maps_path)

    force_x = force_u_x + 0.1 * force_v_x
    force_y = force_u_y + 0.1 * force_v_y
    XX, YY = np.meshgrid(np.arange(1, occupancy_map.shape[0] + 1), np.arange(1, occupancy_map.shape[1] + 1))
    plt.figure(100)
    plt.quiver(XX, YY, force_v_x, force_v_y)
    plt.title('Force V')
    plt.savefig("force_v.png")
    plt.show()

    plt.quiver(XX, YY, force_u_x, force_u_y)
    plt.title('Force U')
    plt.savefig("force_u.png")

    plt.show()

    plt.quiver(XX, YY, force_x, force_y)
    plt.title('Force')
    plt.savefig("force.png")

    plt.show()

    print()
