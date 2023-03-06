import logging
import os
import pickle
import heapq
from collections import defaultdict

node_adj = {}
node_num2_position = {}
node_position2_num = {}


def trans_data(_array):
    """
    """
    # 遍历二维数组
    for i in range(_array.shape[0]):
        for j in range(_array.shape[1]):

            # 建立点位置和序号的相互映射
            num = i * _array.shape[1] + j
            node_num2_position[num] = (i, j)
            node_position2_num[(i, j)] = num

            # 找所有点的邻接点
            # True代表不可行的点
            if _array[i, j]:
                node_adj[(i, j)] = []
            else:
                # False代表这个点是可行点，将它的邻居加入邻接表
                neighbors = []
                if i > 0 and not _array[i - 1, j]:  # 上邻居
                    neighbors.append((i - 1, j))
                if i < _array.shape[0] - 1 and not _array[i + 1, j]:  # 下邻居
                    neighbors.append((i + 1, j))
                if j > 0 and not _array[i, j - 1]:  # 左邻居
                    neighbors.append((i, j - 1))
                if j < _array.shape[1] - 1 and not _array[i, j + 1]:  # 右邻居
                    neighbors.append((i, j + 1))
                node_adj[(i, j)] = neighbors


def k_shortest_paths(graph, start, end, k):
    """
    """
    paths = []
    heap = [(0, [start])]
    visited = defaultdict(set)

    while heap and len(paths) < k:
        (cost, path) = heapq.heappop(heap)
        node_ = path[-1]
        if node_ == end and path not in paths:
            paths.append((path, cost))
        elif node_ not in visited or cost < min(visited[node_]):
            visited[node_].add(cost)
            for neighbor_, weight in graph[node_].items():
                heapq.heappush(heap, (cost + weight, path + [neighbor_]))

    return paths


def bellman_ford(graph, start):
    # 初始化距离字典
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    # 进行V-1轮松弛操作
    for i in range(len(graph) - 1):
        for u in graph:
            for v, weight in graph[u].items():
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight

    # 检查是否存在负权回路
    for u in graph:
        for v, weight in graph[u].items():
            if distances[u] + weight < distances[v]:
                raise ValueError('图中存在负权回路')

    return distances


def compute_geodesic_distance(file_name):
    # 读取数据
    fr = open(file_name, 'rb')
    inf = pickle.load(fr)
    world_map = inf[0]
    start_point, end_point = inf[1].tolist(), inf[2].tolist()

    # 将数据转换成内置格式
    trans_data(world_map)

    # 转换graph的格式
    graph_ksp = {}
    for node_num, position in node_num2_position.items():
        if len(node_adj[position]) == 0:
            continue
        neighbor = {}
        for neighbor_position in node_adj[position]:
            neighbor[node_position2_num[neighbor_position]] = 1
        graph_ksp[node_num] = neighbor

    # 检查终点是否有效
    effective_end_point = []
    for i in range(len(end_point)):
        e = tuple(end_point[i])
        if e not in node_adj:
            print('point is out of map')
            continue
        else:
            if len(node_adj[e]) == 0:
                print('point no path')
            else:
                effective_end_point.append(e)

    # 调用bellman ford算法
    position_dict = {}
    for e in effective_end_point:
        e = node_position2_num[e]
        distance = bellman_ford(graph_ksp, e)
        # 调整输出格式
        for s_point, dist in distance.items():
            position = node_num2_position[s_point]
            if node_num2_position[e] not in position_dict:
                position_dict[node_num2_position[e]] = {position: dist}
            else:
                position_dict[node_num2_position[e]][position] = dist
    return position_dict


if __name__ == '__main__':
    env_parent_folder = '../data/office_1500/envs'
    geodesic_distance_parent_folder = '../data/office_1500/geodesic_distance'
    if not os.path.exists(geodesic_distance_parent_folder):
        os.makedirs(geodesic_distance_parent_folder)

    env_names = os.listdir(env_parent_folder)
    length = len(env_names)
    template = "env_{}.pkl"
    indexes = [303, 553, 678]
    for i in indexes:
        env_name = template.format(i)
        env_path = os.path.join(env_parent_folder, env_name)
        print("Computing geodesic distance for {}...".format(env_name))
        out = compute_geodesic_distance(file_name=env_path)
        out_path = os.path.join(geodesic_distance_parent_folder, env_name)
        pickle.dump(out, open(out_path, 'wb'))
        print("Save to {}!".format(out_path))
    print("Done!")
