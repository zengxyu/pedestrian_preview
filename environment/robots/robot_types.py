from environment.robots.differential_race_car import DifferentialRaceCar
from environment.robots.object_robot import ObjectRobot
from environment.robots.turtlebot import TurtleBot


class RobotTypes:
    RaceCar = "race_car"
    Turtlebot = "turtlebot"
    ObjectRobot = "object_robot"


def init_robot(p, client_id, robot_name, robot_role, physical_step_duration, robot_config, sensor_config, start, yaw):
    if robot_name == RobotTypes.RaceCar:
        robot = DifferentialRaceCar(p, client_id, robot_role, physical_step_duration, robot_config,
                                    sensor_config, start, yaw)
    elif robot_name == RobotTypes.Turtlebot:
        robot = TurtleBot(p, client_id, robot_role, physical_step_duration, robot_config,
                          sensor_config, start, yaw)
    elif robot_name == RobotTypes.ObjectRobot:
        robot = ObjectRobot(p, client_id, robot_role, physical_step_duration, robot_config,
                            sensor_config, start, yaw)
    else:
        raise NotImplementedError
    return robot
