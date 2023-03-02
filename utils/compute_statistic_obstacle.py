import os.path
import pickle

import numpy as np
from numpy import mean

from utils.fo_utility import get_project_path

random_env_folder = os.path.join(get_project_path(), "data", "office_1000", "train", "random_envs")
filenames = os.listdir(random_env_folder)
paths = [os.path.join(random_env_folder, filename) for filename in filenames]
obstacle_num_list = []
for path in paths:
    occupancy_map, _, _ = pickle.load(open(path, "rb"))
    obstacle_num_list.append(np.sum(occupancy_map))
print("min:{}".format(min(obstacle_num_list)))
print("max:{}".format(max(obstacle_num_list)))
print("mean:{}".format(mean(obstacle_num_list)))
