from environment.sensors.lidar_sensor import LidarSensor
from environment.sensors.vision_sensor import VisionSensor
from environment.sensors.multi_vision_sensor import MultiVisionSensor


class SensorTypes:
    LidarSensor = "lidar"
    VisionSensor = "vision"
    MultiVisionSensor = "multi_vision"


def init_sensor(robot_id, sensor_name, sensor_config):
    if sensor_name == SensorTypes.LidarSensor:
        return LidarSensor(robot_id, sensor_config=sensor_config)
    elif sensor_name == SensorTypes.VisionSensor:
        return VisionSensor(robot_id=robot_id, sensor_config=sensor_config)
    elif sensor_name == SensorTypes.MultiVisionSensor:
        return MultiVisionSensor(robot_id=robot_id, sensor_config=sensor_config)
    else:
        raise NotImplementedError
