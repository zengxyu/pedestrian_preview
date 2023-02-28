import os
import pickle

from utils.office_1000_generator import display_and_save_only_env

if __name__ == '__main__':
    env_parent_folder = '../data/office_1000/train/random_envs'
    occupancy_maps_folder = '../data/office_1000/train/occupancy_maps'
    if not os.path.exists(occupancy_maps_folder):
        os.makedirs(occupancy_maps_folder)

    env_names = os.listdir(env_parent_folder)
    length = len(env_names)
    template = "env_{}.pkl"
    image_template = "env_{}.png"
    indexes = [i for i in range(1000)]
    for i in indexes:
        env_name = template.format(i)
        image_name = image_template.format(i)
        env_path = os.path.join(env_parent_folder, env_name)
        fr = open(env_path, 'rb')
        env, _, _ = pickle.load(fr)
        image_out_path = os.path.join(occupancy_maps_folder, image_name)
        display_and_save_only_env(env, save=True, save_path=image_out_path)
        print("Save to {}!".format(image_out_path))
    print("Done!")
