import os
import pickle

from environment.gen_scene.world_loader import read_occupancy_map
from utils.fo_utility import get_data_path


def convert_p2v_to_occupancy_map():
    p2v_map_folder = os.path.join(get_data_path(), "p2v", "env1", "maps")
    p2v_occupancy_map_folder = os.path.join(get_data_path(), "p2v", "env1", "occupancy_map")
    if not os.path.exists(p2v_occupancy_map_folder):
        os.makedirs(p2v_occupancy_map_folder)

    # 地图读入路径
    p2v_map_path = os.path.join(p2v_map_folder, "map.txt")
    # 地图写出路径
    p2v_occupancy_map_path = os.path.join(p2v_occupancy_map_folder, "occupancy_map.pkl")
    ratio = 0.25 / 0.1
    occupancy_map = read_occupancy_map(p2v_map_path, ratio)
    p2v_occ_map_file = open(p2v_occupancy_map_path, "wb")
    pickle.dump(occupancy_map, p2v_occ_map_file)


if __name__ == '__main__':
    convert_p2v_to_occupancy_map()
