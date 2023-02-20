import os.path
import pickle

from utils.fo_utility import get_project_path


def get_shortest_path(path):
    shortest_path_obj = pickle.load(open(path, 'rb'))
    return shortest_path_obj


# def exchange_data_structure(obj):
#     position_dict = {}
#     for e in effective_end_point:
#         e = node_position2_num[e]
#         distance = bellman_ford(graph_ksp, e)
#         # 调整输出格式
#         for s_point, dist in distance.items():
#             position = node_num2_position[s_point]
#             if node_num2_position[e] not in position_dict:
#                 position_dict[node_num2_position[e]] = {position: dist}
#             else:
#                 position_dict[node_num2_position[e]][position] = dist
#     return position_dict


if __name__ == '__main__':
    geodesic_parent_folder = os.path.join(get_project_path(), "data", "office_1000", "geodesic_distance")
    file_names = os.listdir(geodesic_parent_folder)
    for file_name in file_names:
        path = os.path.join(get_project_path(), "data", "office_1000", "geodesic_distance", "env_2.pkl")
        shortest_path_obj = get_shortest_path(path)

    print()
