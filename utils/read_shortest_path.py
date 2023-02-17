import pickle


def get_shortest_path(path):
    shortest_path_obj = pickle.load(open(path, 'rb'))
    print()
    return


if __name__ == '__main__':
    get_shortest_path(path="../data/shortest_path_distance_0.pkl")
