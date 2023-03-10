import pickle
import shutil

from multiprocessing import Pool
import os, time, random

from utils.compute_geodesic_distance import compute_geodesic_distance
from utils.fo_utility import get_project_path, get_office_evacuation_path


def compute_and_save_geodesic_distance(envs_folder, geo_dist_folder, start_index, end_index):
    print('Run task from {} to {}'.format(start_index, end_index))
    start = time.time()
    template = "env_{}.pkl"
    indexes = [i for i in range(start_index, end_index)]
    for i in indexes:
        env_name = template.format(i)
        env_path = os.path.join(envs_folder, env_name)
        out_path = os.path.join(geo_dist_folder, env_name)
        print("out_path:{}".format(out_path))
        print("Computing geodesic distance for {}...".format(env_name))
        out = compute_geodesic_distance(file_name=env_path)
        pickle.dump(out, open(out_path, 'wb'))
        print("Save to {}!".format(out_path))
    print("Done!")

    end = time.time()
    print('Task from {} to {} runs {} seconds.'.format(start_index, end_index, end - start))


def multi_process(folder_name, phase, indexes):
    # in
    folder = os.path.join(get_office_evacuation_path(), folder_name)
    # in
    in_envs_folder = os.path.join(folder, phase, "envs")
    # out
    out_geo_dist_folder = os.path.join(folder, phase, "geodesic_distance")

    if not os.path.exists(out_geo_dist_folder):
        os.makedirs(out_geo_dist_folder)

    print('Parent process %s.' % os.getpid())
    # 进程数量
    num_process = 5
    p = Pool(num_process)
    num_batch = int((indexes[1] - indexes[0]) / num_process)
    split_env_indexes = [[indexes[0] + i * num_batch, indexes[0] + (i + 1) * num_batch] for i in range(num_process)]
    for start_index, end_index in split_env_indexes:
        p.apply_async(compute_and_save_geodesic_distance,
                      args=(in_envs_folder, out_geo_dist_folder, start_index, end_index,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')


if __name__ == '__main__':
    phase = "test"
    folder_name = "sg_no_walls"
    # 要处理从哪个到哪个文件
    indexes = [12, 192]

    multi_process(folder_name, phase, indexes)
