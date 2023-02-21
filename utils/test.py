import pickle

import numpy as np


def check_invalid_geodesic_distance():
    for i in range(1000):
        file_path = "/home/zeng/workspace/pycharm_workspace/navigation/pedestrian_preview/data/office_1000/geodesic_distance/env_{}.pkl".format(
            i)
        file = pickle.load(open(file_path, "rb"))
        flag = False
        for k1, v1 in file.items():
            for k2, v2 in v1.items():
                if np.isinf(v2):
                    flag = True
                if flag:
                    break
            if flag:
                break
        if flag:
            print(file_path)


if __name__ == '__main__':
    check_invalid_geodesic_distance()
