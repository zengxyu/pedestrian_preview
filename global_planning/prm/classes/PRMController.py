from collections import defaultdict
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import shapely.geometry
import argparse

from .Dijkstra import Graph, dijkstra, to_array
from .Utils import Utils


class PRMController:
    def __init__(self, num_of_random_coordinates, obstacles, destination, max_map_size):
        self.num_of_random_coordinates = num_of_random_coordinates
        self.coordinates_list = np.array([])
        self.obstacles = obstacles
        # self.current = np.array(current)
        self.destination = np.array(destination)
        self.max_map_size = max_map_size
        self.graph = Graph()
        self.utils = Utils()
        self.solution_found = False
        self.collision_free_points = []

    def run_prm(self, initial_random_seed, save_image=True):
        seed = initial_random_seed
        # Keep resampling if no solution found
        # while not self.solution_found:
        print("Trying with random seed {}".format(seed))
        np.random.seed(seed)

        # Generate n random samples called milestones
        self.gen_milestones(max_map_size=self.max_map_size)

        # Check if milestones are collision free
        self.check_if_collision_free()

        # Link each milestone to k nearest neighbours.
        # Retain collision free links as local paths.
        self.find_nearest_neighbour()

        self.compute_dijkstra_from_destination()
        # Search for shortest path from start to end node - Using Dijksta's shortest path alg
        # path, dist, path_to_end = self.get_shortest_path()
        # self.display_path(path, dist, path_to_end)
        # seed = np.random.randint(1, 100000)
        # self.coordinates_list = np.array([])
        # self.graph = Graph()

        # if save_image:
        #     plt.savefig("{}_samples.png".format(self.num_of_random_coordinates))
        # plt.show()
        # return path

    def display_path(self, path_points):
        x = [int(item[0]) for item in path_points]
        y = [int(item[1]) for item in path_points]
        plt.plot(x, y, c="blue", linewidth=3.5, linestyle='--')

        print("****Output****")

        # print("The quickest path from {} to {} is: \n {} \n with a distance of {}".format(
        #     start,
        #     self.destination,
        #     " \n ".join(points_to_end),
        #     str(dist[self.end_node])
        # )
        # )

    def gen_milestones(self, max_map_size=100):
        self.coordinates_list = np.random.randint(
            max_map_size, size=(self.num_of_random_coordinates, 2))
        # Adding begin and end points
        # self.current = self.current.reshape(1, 2)
        self.destination = self.destination.reshape(1, 2)
        self.coordinates_list = np.concatenate(
            (self.coordinates_list, self.destination), axis=0)

    def check_if_collision_free(self):
        for point in self.coordinates_list:
            collision = self.check_point_collision(point)
            if not collision:
                self.collision_free_points.append(point)
        self.collision_free_points = np.array(self.collision_free_points)
        self.plot_points(self.collision_free_points)

    def find_nearest_neighbour(self, k=10):
        X = np.array(self.collision_free_points)
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X)
        distances, indices = knn.kneighbors(X)
        collision_free_path = []

        for i, p in enumerate(X):
            # Ignoring nearest neighbour - nearest neighbour is the point itself
            for j, neighbour in enumerate(X[indices[i][1:]]):
                start_line = p
                end_line = neighbour
                if not self.check_point_collision(start_line) and not self.check_point_collision(end_line):
                    if not self.check_line_collision(start_line, end_line):
                        collision_free_path.append(p)
                        collision_free_path.append(neighbour)
                        a = str(self.find_node_index(p))
                        b = str(self.find_node_index(neighbour))
                        self.graph.add_node(a)
                        self.graph.add_edge(a, b, distances[i, j + 1])
                        x = [p[0], neighbour[0]]
                        y = [p[1], neighbour[1]]
                        plt.plot(x, y)
        # plt.show()

    def compute_dijkstra_from_destination(self):
        self.destination_node = str(self.find_node_index(self.destination))
        self.dist, self.prev = dijkstra(self.graph, self.destination_node)

    def get_shortest_path_from_start(self, start):
        self.start_node = str(self.find_nearest_node_index(start))
        path_to_destination = to_array(self.prev, self.start_node)
        shortest_path = [(self.find_points_from_node(path)) for path in path_to_destination]
        return shortest_path

    # def get_shortest_path(self):
    #     self.start_node = str(self.find_node_index(self.current))
    #     self.end_node = str(self.find_node_index(self.destination))
    #
    #     dist, prev = dijkstra(self.graph, self.start_node)
    #
    #     path_to_end = to_array(prev, self.end_node)
    #
    #     if len(path_to_end) > 1:
    #         self.solution_found = True
    #     else:
    #         return
    #
    #     # Plotting shortest path
    #     shortest_path = [(self.find_points_from_node(path)) for path in path_to_end]
    #     # self.display_path(shortest_path, dist, path_to_end)
    #     # plt.show()
    #     return shortest_path, dist, path_to_end

    def check_line_collision(self, start_line, end_line):
        collision = False
        line = shapely.geometry.LineString([start_line, end_line])
        for obs in self.obstacles:
            if self.utils.isWall(obs):
                unique_coords = np.unique(obs.allCords, axis=0)
                wall = shapely.geometry.LineString(
                    unique_coords)
                if line.intersection(wall):
                    collision = True
            else:
                obstacle_shape = shapely.geometry.Polygon(
                    obs.allCords)
                collision = line.intersects(obstacle_shape)
            if collision:
                return True
        return False

    def find_node_index(self, p):
        return np.where((self.collision_free_points == p).all(axis=1))[0][0]

    def find_nearest_node_index(self, p):
        nearest_index = np.argmin(np.linalg.norm(self.collision_free_points - p, axis=1))
        return nearest_index

    def find_points_from_node(self, n):
        return self.collision_free_points[int(n)]

    def plot_points(self, points):
        x = [item[0] for item in points]
        y = [item[1] for item in points]
        plt.scatter(x, y, c="black", s=10)

    def check_collision(self, obs, point):
        p_x = point[0]
        p_y = point[1]
        if obs.bottomLeft[0] <= p_x <= obs.bottomRight[0] and obs.bottomLeft[1] <= p_y <= obs.topLeft[1]:
            return True
        else:
            return False

    def check_point_collision(self, point):
        for obs in self.obstacles:
            collision = self.check_collision(obs, point)
            if collision:
                return True
        return False
