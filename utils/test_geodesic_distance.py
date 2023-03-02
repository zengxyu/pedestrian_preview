import os.path
import pickle
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

scene_path = "/home/zeng/workspace/pycharm_workspace/navigation/pedestrian_preview/data/office_1000/train/random_envs/env_0.pkl"
geo_path = "/home/zeng/workspace/pycharm_workspace/navigation/pedestrian_preview/data/office_1000/train/geodesic_distance/env_0.pkl"
save_path = os.path.join(
    "/home/zeng/workspace/pycharm_workspace/navigation/pedestrian_preview/data/office_1000/train/geodesic_distance_images",
    "env_0.txt")

occupancy_map, _, _ = pickle.load(open(scene_path, "rb"))
obj = pickle.load(open(geo_path, "rb"))

K = 0.05
goal = (38, 9)
sigma1 = 50
sigma2 = 50
geodesic_distance: Dict = obj[goal]
image1 = np.zeros((70, 70)).astype(float)
image2 = np.zeros((70, 70)).astype(float)
k1 = 0.01
k2 = 0.005
k3 = 1
for key in geodesic_distance.keys():
    distance = geodesic_distance[key]
    image1[key] = distance
    image2[key] = k1 * distance + k2 * distance ** 2 + k3
    if distance > 39:
        image2[key] = 11

image_x = np.zeros_like(image2)
image_x = image_x.astype(float)
for i in range(0, image2.shape[0]):
    for j in range(0, image2.shape[1]):
        if occupancy_map[i][j] == 1 or occupancy_map[i][j - 1] == 1:
            continue
        image_x[i][j] = np.sign(image1[i][j - 1] - image1[i][j]) * image2[i][j]

image_y = np.zeros_like(image2).astype(float)
image_y = image_y.astype(float)

for i in range(0, image2.shape[0]):
    for j in range(0, image2.shape[1]):
        if occupancy_map[i][j] == 1 or occupancy_map[i - 1][j] == 1:
            continue
        image_y[i][j] = np.sign(image1[i - 1][j] - image1[i][j]) * image2[i][j]

save_path_image_x = os.path.join(
    "/home/zeng/workspace/pycharm_workspace/navigation/pedestrian_preview/data/office_1000/train/geodesic_distance_images",
    "env_0_image_x.txt")
save_path_image_y = os.path.join(
    "/home/zeng/workspace/pycharm_workspace/navigation/pedestrian_preview/data/office_1000/train/geodesic_distance_images",
    "env_0_image_y.txt")
np.savetxt(save_path_image_x, image_x.tolist())
np.savetxt(save_path_image_y, image_y.tolist())

plt.imshow(image2)
plt.show()
XX, YY = np.meshgrid(np.arange(1, image2.shape[0] + 1), np.arange(1, image2.shape[1] + 1))
plt.figure(20)
plt.quiver(XX, YY, image_x, image_y)
plt.title('Convolved Map Vectors')
plt.show()

print()
