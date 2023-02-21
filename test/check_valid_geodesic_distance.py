import os.path
import pickle

import numpy as np

from utils.fo_utility import get_project_path


def check_invalid_geodesic_distance():
    for i in range(1000):
        file_path = os.path.join(get_project_path(), "data", "office_1000", "geodesic_distance", "env_{}.pkl".format(i))
        # file_path = "/home/zeng/workspace/pycharm_workspace/navigation/pedestrian_preview/data/office_1000/geodesic_distance/env_{}.pkl".format(
        #     i)
        file = pickle.load(open(file_path, "rb"))
        invalid_distance = False
        for k1, v1 in file.items():
            for k2, v2 in v1.items():
                if np.isinf(v2):
                    invalid_distance = True
                if invalid_distance:
                    break
            if invalid_distance:
                break
        if invalid_distance:
            print(file_path)


if __name__ == '__main__':
    check_invalid_geodesic_distance()
