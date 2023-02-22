import os.path
import pickle

import numpy as np

from environment.gen_scene.common_sampler import sg_opposite_baffle_sampler3
from environment.gen_scene.compute_door import compute_door
from utils.fo_utility import get_project_path
from utils.image_utility import dilate_image
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

    num_starts = 20
    # 读取每一个occupancy map, 计算门位置， 保存到新的文件夹中
    for filename in os.listdir(office_1000_random_envs_folder):
        print("Processing file:{}".format(filename))
        file_path = os.path.join(office_1000_random_envs_folder, filename)
        occupancy_map, _, _ = pickle.load(open(file_path, "rb"))
        dilated_occ_map = dilate_image(occupancy_map, dilation_size=5)

        door_center = compute_door(occupancy_map).tolist()

        # sample start point
        count = 0
        starts = []
        while len(starts) < num_starts and count < 100:
            # print("start point number:{}".format(len(starts)))

            start, sample_success = sg_opposite_baffle_sampler3(dilate_occupancy_map=dilated_occ_map,
                                                                occupancy_map=occupancy_map,
                                                                goal=door_center)

            if sample_success:
                starts.append(start)
            count += 1

        ends = np.array([door_center]).astype(int)
        starts = np.array(starts).astype(int)
        out_file_path = os.path.join(out_office_1000_goal_outdoor_envs_folder, filename)
        out_image_path = os.path.join(out_office_1000_goal_outdoor_images_folder,
                                      filename[:filename.index(".")] + ".png")

        pickle.dump([occupancy_map, starts, ends], open(out_file_path, "wb"))
        ends_tile = np.tile(ends, (len(starts), 1))
        display_and_save(occupancy_map, starts, ends_tile, save=True, save_path=out_image_path)

    print("Done!")


if __name__ == '__main__':
    convert_office_1000_goal_outdoor()
