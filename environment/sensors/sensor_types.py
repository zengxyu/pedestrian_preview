from environment.sensors.lidar_sensor import LidarSensor
from environment.sensors.penetrate_ray_sensor import PenetrateRaySensor
from environment.sensors.vision_sensor import VisionSensor
from environment.sensors.multi_vision_sensor import MultiVisionSensor


class SensorTypes:
    LidarSensor = "lidar"
    VisionSensor = "vision"
    MultiVisionSensor = "multi_vision"
    PenetrateRaySensor = "penetrating_rays"


def init_sensors(robot_id, sensor_names, sensors_config):
    sensors = []
    for sensor_name in sensor_names:
        sensor_config = sensors_config[sensor_name]
        if sensor_name == SensorTypes.LidarSensor:
            sensors.append(LidarSensor(robot_id, sensor_config=sensor_config))

        elif sensor_name == SensorTypes.VisionSensor:
            sensors.append(VisionSensor(robot_id=robot_id, sensor_config=sensor_config))

        elif sensor_name == SensorTypes.MultiVisionSensor:
            sensors.append(MultiVisionSensor(robot_id=robot_id, sensor_config=sensor_config))

        elif sensor_name == SensorTypes.PenetrateRaySensor:
            sensors.append(PenetrateRaySensor(robot_id=robot_id, sensor_config=sensor_config))

        else:
            raise NotImplementedError

    return sensors
