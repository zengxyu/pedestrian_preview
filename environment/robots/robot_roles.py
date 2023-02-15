import logging

import numpy as np


class RobotRoles:
    AGENT = "agent"
    NPC = "npc"


def get_role_color(robot_role: str):
    """
    get color by robot role
    """
    # agent color : red
    if robot_role == RobotRoles.AGENT:
        color = [1, 0, 0, 1]
        # color = list(np.random.random(size=3)) + [1]
    elif robot_role == RobotRoles.NPC:
        color = list(np.random.random(size=3)) + [1]
    else:
        logging.error("No such a robot role")
        raise NotImplementedError
    return color
