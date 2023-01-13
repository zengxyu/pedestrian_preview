import logging

from utils.image_utility import dilate_image
from traditional_planner.a_star.astar import AStar


def plan_a_star_path(occ_map, robot_occ_start, robot_occ_end):
    if robot_occ_start is None or robot_occ_end is None:
        raise Exception("Check if robot_occ_start and robot_occ_end is None")
    if len(robot_occ_start) != 2 or len(robot_occ_end) != 2:
        raise Exception("Check if len(robot_occ_start) !=2 or len(robot_occ_end) !=2")
    robot_occ_start = robot_occ_start.astype(int)
    robot_occ_end = robot_occ_end.astype(int)
    obs_om_path = AStar(dilate_image(occ_map, 2)).search_path(tuple(robot_occ_start), tuple(robot_occ_end))

    if obs_om_path is None or len(obs_om_path) == 0:
        return None
    return obs_om_path
