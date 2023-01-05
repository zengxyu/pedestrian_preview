from math import cos, sin

import cv2
import numpy as np

from environment.nav_utilities.coordinates_converter import cvt_to_bu, transform_point_to_image_coord
from environment.scenes.scene_helper import load_scene
from utils.fo_utility import get_project_path
from utils.image_utility import dilate_image


def display_image(result_image, scale=5):
    shape = result_image.shape
    resized_shape = (shape[0] * scale, shape[1] * scale)
    result_image = cv2.resize(result_image, resized_shape, cv2.INTER_NEAREST)
    cv2.imshow("image", result_image)
    cv2.waitKey()
    cv2.destroyAllWindows()


class DrawGlobalMap:
    def __init__(self, occupancy_map, dilated_occupancy_map, om_path):
        self.occupancy_map = occupancy_map
        self.dilated_occupancy_map = dilated_occupancy_map
        self.om_path = om_path

        self.shape = np.shape(self.occupancy_map)
        self.res_image = np.zeros((self.shape[0], self.shape[1], 3))

        self.draw_image()
        self.draw_path(self.res_image, om_path, color=[255, 0, 255])

    def draw_image(self):
        diff_map = self.occupancy_map != self.dilated_occupancy_map
        self.res_image[occupancy_map] = [255, 255, 0]
        self.res_image[diff_map] = [255, 0, 0]
        return self.res_image

    def draw_path(self, res_image, om_path, color):
        for p in om_path:
            res_image[p[0], p[1], :] = color
        return res_image

    def draw_robot(self, robot_x, robot_y, robot_yaw, color=(0, 0, 255)):
        radius = 5
        start = transform_point_to_image_coord(np.array([robot_x, robot_y]))
        res_image = cv2.circle(self.res_image.copy(), center=start, radius=radius, color=color)
        end = start + 2 * radius * transform_point_to_image_coord(np.array([cos(robot_yaw), sin(robot_yaw)]))
        res_image = cv2.line(res_image, start, end, color)
        return res_image

    def draw(self, robot_x, robot_y, robot_yaw, deformed_waypoint_ind, deformed_waypoint, deforming_force):
        # draw image
        res_image = self.draw_robot(robot_x, robot_y, robot_yaw)
        self.om_path[deformed_waypoint_ind] = [deformed_waypoint[0], deformed_waypoint[1]]
        res_image[deformed_waypoint[0], deformed_waypoint[1], :] = [255, 255, 255]
        return res_image



if __name__ == '__main__':
    import os
    import pickle

    from_path = os.path.join(get_project_path(), "output", "office.obj")
    _, _, occupancy_map, grid_res = load_scene(from_path)
    dilated_occ_map = dilate_image(occupancy_map, dilation_size=4)

    path_save_to_dir = "environment/env_test/output/path"

    # om_path = g_planner.plan(Planner.A_star, env.dilated_occ_map, env.s_om_pose, env.g_om_pose, save_to_dir=path_save_to_dir)
    om_path = pickle.load(open(os.path.join(get_project_path(), path_save_to_dir, "path.obj"), "rb"))

    draw_global_map = DrawGlobalMap(occupancy_map, dilated_occ_map, om_path)
    res_image = draw_global_map.draw(25, 50, np.pi / 2, 30, om_path[30] + np.array([10, 10]), 2)
    display_image(res_image)
