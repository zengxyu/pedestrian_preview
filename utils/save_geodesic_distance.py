import pickle

from multiprocessing import Pool
import os, time, random

from utils.compute_geodesic_distance import compute_geodesic_distance


def compute_and_save_geodesic_distance(start_index, end_index):
    print('Run task from {} to {}'.format(start_index, end_index))
    start = time.time()
    template = "env_{}.pkl"
    indexes = [i for i in range(start_index, end_index, 1)]
    for i in indexes:
        env_name = template.format(i)
        env_path = os.path.join(env_parent_folder, env_name)
        print("Computing geodesic distance for {}...".format(env_name))
        out = compute_geodesic_distance(file_name=env_path)
        out_path = os.path.join(geodesic_distance_parent_folder, env_name)
        pickle.dump(out, open(out_path, 'wb'))
        print("Save to {}!".format(out_path))
    print("Done!")

    end = time.time()
    print('Task from {} to {} runs {} seconds.'.format(start_index, end_index, end - start))


def multi_process():
    env_parent_folder = '../data/office_1000/random_envs'
    geodesic_distance_parent_folder = '../data/office_1000/geodesic_distance'
    if not os.path.exists(geodesic_distance_parent_folder):
        os.makedirs(geodesic_distance_parent_folder)

    env_names = os.listdir(env_parent_folder)
    length = len(env_names)

    print('Parent process %s.' % os.getpid())
    num_process = 5

    p = Pool(num_process)

    env_indexes = [0, 400]
    part = int((env_indexes[1] - env_indexes[0]) / num_process)
    split_env_indexes = [[env_indexes[0] + i * part, env_indexes[0] + (i + 1) * part] for i in range(num_process)]
    for start_index, end_index in split_env_indexes:
        p.apply_async(compute_and_save_geodesic_distance, args=(start_index, end_index,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    return


if __name__ == '__main__':
    multi_process()
