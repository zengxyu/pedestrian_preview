import os.path
import pickle

import numpy as np

from environment.gen_scene.compute_door import compute_door
from utils.fo_utility import get_project_path
from utils.office_1000_generator import display_and_save


def convert_office_1000_goal_outdoor():
    office_1000_parent_folder = os.path.join(get_project_path(), "data", "office_1000")
    office_1000_geodesic_distance_folder = os.path.join(office_1000_parent_folder, "geodesic_distance")
    office_1000_random_envs_folder = os.path.join(office_1000_parent_folder, "random_envs")
    office_1000_random_envs_images_folder = os.path.join(office_1000_parent_folder, "random_envs_images")

    out_office_1000_goal_outdoor = os.path.join(get_project_path(), "data", "office_1000_goal_outdoor")
    out_office_1000_goal_outdoor_envs_folder = os.path.join(out_office_1000_goal_outdoor, "random_envs")
    out_office_1000_goal_outdoor_images_folder = os.path.join(out_office_1000_goal_outdoor, "random_envs_images")

    if not os.path.exists(out_office_1000_goal_outdoor):
        os.makedirs(out_office_1000_goal_outdoor)
    if not os.path.exists(out_office_1000_goal_outdoor_envs_folder):
        os.makedirs(out_office_1000_goal_outdoor_envs_folder)
    if not os.path.exists(out_office_1000_goal_outdoor_images_folder):
        os.makedirs(out_office_1000_goal_outdoor_images_folder)

    # 读取每一个occupancy map, 计算门位置， 保存到新的文件夹中
    for filename in os.listdir(office_1000_random_envs_folder):
        print("Processing file:{}".format(filename))
        file_path = os.path.join(office_1000_random_envs_folder, filename)
        occ_map, starts, _ = pickle.load(open(file_path, "rb"))
        door_center = compute_door(occ_map).tolist()
        ends = np.array([door_center])
        out_file_path = os.path.join(out_office_1000_goal_outdoor_envs_folder, filename)
        out_image_path = os.path.join(out_office_1000_goal_outdoor_images_folder,
                                      filename[:filename.index(".")] + ".png")

        pickle.dump([occ_map, starts, ends], open(out_file_path, "wb"))
        display_and_save(occ_map, starts, ends, save=True, save_path=out_image_path)

    print("Done!")


if __name__ == '__main__':
    convert_office_1000_goal_outdoor()
