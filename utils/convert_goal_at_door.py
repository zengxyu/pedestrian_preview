import os.path
import pickle

import numpy as np

from environment.gen_scene.common_sampler import sg_opposite_baffle_sampler3
from environment.gen_scene.compute_door import compute_door
from utils.fo_utility import *
from utils.image_utility import dilate_image
from utils.gen_fixed_envs import display_and_save


# 将 office 1500 中的目标点转为门，起点是和终点仍然是有隔板的点, 之后需要重新计算obstacle distance, geodesic distance, uv_forces
def convert_office_1000_goal_outdoor(dataset_path, folder_name, out_folder_name, phase, indexes):
    env_folder = os.path.join(dataset_path, folder_name, phase, "envs")
    out_env_folder = os.path.join(dataset_path, out_folder_name, phase, "envs")
    out_image_folder = os.path.join(dataset_path, out_folder_name, phase, "envs_images")

    num_starts = 20
    if not os.path.exists(out_env_folder):
        os.makedirs(out_env_folder)
    if not os.path.exists(out_image_folder):
        os.makedirs(out_image_folder)

    file_name_template = "env_{}.pkl"
    image_name_template = "image_{}.png"
    # 读取每一个occupancy map, 计算门位置， 保存到新的文件夹中
    for i in indexes:
        file_name = file_name_template.format(i)
        image_name = image_name_template.format(i)
        print("Processing file:{}".format(file_name))
        in_file_path = os.path.join(env_folder, file_name)
        out_file_path = os.path.join(out_env_folder, file_name)
        out_image_path = os.path.join(out_image_folder, image_name)

        occupancy_map, _, _ = pickle.load(open(in_file_path, "rb"))
        dilated_occ_map = dilate_image(occupancy_map, dilation_size=5)

        door_center = compute_door(occupancy_map).tolist()

        # sample start point
        count = 0
        starts = []
        # 采样起点，和终点之间有障碍物
        while len(starts) < num_starts and count < 100:
            # print("start point number:{}".format(len(starts)))
            start, sample_success = sg_opposite_baffle_sampler3(dilate_occupancy_map=dilated_occ_map,
                                                                occupancy_map=occupancy_map,
                                                                goal=door_center)

            if sample_success:
                starts.append(start)
            count += 1

        # 如果不能采样成功，采样起点，和终点之间没有障碍物
        count = 0
        while len(starts) < num_starts and count < 100:
            # print("start point number:{}".format(len(starts)))

            start, sample_success = sg_opposite_baffle_sampler3(dilate_occupancy_map=dilated_occ_map,
                                                                occupancy_map=occupancy_map,
                                                                goal=door_center)

            starts.append(start)
            count += 1

        ends = np.array([door_center]).astype(int)
        starts = np.array(starts).astype(int)
        # 将目标转为门以后，保存该occupancy map
        pickle.dump([occupancy_map, starts, ends], open(out_file_path, "wb"))
        ends_tile = np.tile(ends, (len(starts), 1))
        # 保存图像
        display_and_save(occupancy_map, starts, ends_tile, save=True, save_path=out_image_path)

    print("Done!")


if __name__ == '__main__':
    dataset_path = get_office_evacuation_path()
    folder_name = "sg_walls"
    out_folder_name = "goal_at_door"
    phase = "train"
    # 要处理从哪个到哪个文件
    indexes = [i for i in range(1200, 1700)]

    convert_office_1000_goal_outdoor(dataset_path, folder_name, out_folder_name, phase, indexes)
