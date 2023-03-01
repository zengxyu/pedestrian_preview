import pickle

import numpy as np

scene_path = "/home/zeng/workspace/pycharm_workspace/navigation/pedestrian_preview/data/office_1000/train/random_envs/env_0.pkl"
occupancy_map, starts, ends = pickle.load(open(scene_path, 'rb'))
occupancy_map.astype(bool).tolist()
np.savetxt("occupancy_map_0.txt", occupancy_map)