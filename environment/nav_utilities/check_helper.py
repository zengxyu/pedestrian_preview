from typing import List

from pybullet_utils.bullet_client import BulletClient


class CollisionType:
    NoCollision = -1
    CollisionWithWall = 1
    CollisionWithPedestrian = 2
    CollisionWithAgent = 3


def check_collision(_bullet_client: BulletClient, robot_ids: List[int], obstacle_ids: List[int]) -> bool:
    """
    check if robot id in robot_ids collision
    :param _bullet_client:
    :param robot_ids:
    :param obstacle_ids:
    :return: True -> collision
    """
    collision = False
    for robot_id in robot_ids:
        for obstacle_id in obstacle_ids:
            if _bullet_client.getContactPoints(robot_id, obstacle_id):
                collision = True
    return collision


distance_threshes = [0.1, 0.2, 0.3, 0.4, 0.5]
penalties = [5, 4, 3, 2, 1]


def compute_proxemic_obstacle_penalty(_bullet_client: BulletClient, robot_ids: List[int],
                                      obstacle_ids: List[int]) -> int:
    max_penalty = 0

    for robot_id in robot_ids:
        for obstacle_id in obstacle_ids:
            for penalty, distance_thresh in zip(penalties, distance_threshes):
                if _bullet_client.getClosestPoints(robot_id, obstacle_id, distance_thresh):
                    # penalty_ += penalty
                    if penalty > max_penalty:
                        max_penalty = penalty
                    break
    return max_penalty


def check_goals_reached(_bullet_client: BulletClient, robot_id: int, goal_ids: List[int],
                        goal_reached_thresh: float) -> bool:
    """

    :param _bullet_client:
    :param robot_id:
    :param goal_ids:
    :param goal_reached_thresh:
    :return: True if robot with robot_id reaches goal with goal_id in goal_ids
    """
    for goal_id in goal_ids:
        if _bullet_client.getClosestPoints(
                robot_id, goal_id, goal_reached_thresh
        ):
            return True

    return False


def check_goals_visible(_bullet_client: BulletClient, goal_ids: List[int], goal_poses: List[List],
                        pose: List[int]) -> bool:
    """
    check if goals are visible
    :param _bullet_client:
    :param goal_ids:
    :param goal_poses:
    :param pose:
    :return:
    """
    for i, goal_id in enumerate(goal_ids):
        if (goal_id == _bullet_client.rayTest(
                [pose[0], pose[1], 0.75],
                [goal_poses[i], goal_poses[i + 1], 0.75],
        )[0][0]):
            return True
    return False
