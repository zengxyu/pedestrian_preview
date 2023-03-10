import os.path
import pickle

import numpy as np

from environment.gen_scene.common_sampler import sg_opposite_baffle_sampler3, sg_opposite_baffle_sampler4
from environment.gen_scene.compute_door import compute_door
from utils.fo_utility import get_project_path, get_office_evacuation_path
from utils.image_utility import dilate_image
from utils.gen_fixed_envs import display_and_save


# 将 office 1500 中的目标点转为门，起点是和终点仍然是有隔板的点, 之后需要重新计算obstacle distance, geodesic distance, uv_forces
def convert_sg_no_walls():
    """
    将环境中的起点终点重新采样
    之前起点终点之间有障碍物
    转为起点终点之间没有障碍物
    Returns:

    """
    phase = "train"
    in_folder_name = "sg_walls"
    out_folder_name = "sg_no_walls"
    num_starts = 20
    indexes = [i for i in range(0, 1200)]

    # in
    sg_walls_folder = os.path.join(get_office_evacuation_path(), in_folder_name)
    sg_walls_envs_folder = os.path.join(sg_walls_folder, phase, "envs")
    # out
    sg_no_walls = os.path.join(get_office_evacuation_path(), out_folder_name)
    sg_no_walls_envs_folder = os.path.join(sg_no_walls, phase, "envs")
    sg_no_walls_images_folder = os.path.join(sg_no_walls, phase, "envs_images")

    if not os.path.exists(sg_no_walls_envs_folder):
        os.makedirs(sg_no_walls_envs_folder)
    if not os.path.exists(sg_no_walls_images_folder):
        os.makedirs(sg_no_walls_images_folder)

    file_name_template = "env_{}.pkl"
    image_name_template = "image_{}.png"
    # 读取每一个occupancy map, 计算门位置， 保存到新的文件夹中
    for i in indexes:
        file_name = file_name_template.format(i)
        image_name = image_name_template.format(i)
        print("Processing file:{}".format(file_name))
        in_file_path = os.path.join(sg_walls_envs_folder, file_name)
        out_file_path = os.path.join(sg_no_walls_envs_folder, file_name)
        out_image_path = os.path.join(sg_no_walls_images_folder, image_name)

        occupancy_map, _, _ = pickle.load(open(in_file_path, "rb"))
        dilated_occ_map = dilate_image(occupancy_map, dilation_size=5)

        # sample start point
        starts = []
        ends = []
        # 如果不能采样成功，采样起点，和终点之间没有障碍物
        count = 0
        while len(starts) < num_starts and count < 100:
            # print("start point number:{}".format(len(starts)))
            [start, end], sample_success = sg_opposite_baffle_sampler4(dilate_occupancy_map=dilated_occ_map,
                                                                       occupancy_map=occupancy_map)

            if sample_success:
                starts.append(start)
                ends.append(end)
            count += 1


        ends = np.array(ends).astype(int)
        starts = np.array(starts).astype(int)
        # 将目标转为门以后，保存该occupancy map
        pickle.dump([occupancy_map, starts, ends], open(out_file_path, "wb"))
        # 保存图像
        display_and_save(occupancy_map, starts, ends, save=True, save_path=out_image_path)

    print("Done!")


if __name__ == '__main__':
    convert_sg_no_walls()
