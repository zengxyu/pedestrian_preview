import numpy as np

from environment.sensors.vision_sensor import VisionSensor


class HumanAgent:
    def __init__(self):
        self.sensor = VisionSensor(robot_id=self.body_id, sensor_config=self.sensor_config)
        self.sensor_config = self.sensors_config[self.robot_config["sensor"]]
        human = Man(self.client_id, partitioned=True, timestep=self.physical_step_duration,
                    translation_scaling=0.95 / 5)
        human.reset()
        human.resetGlobalTransformation(
            xyz=np.array([start[0], start[1], 0.94 * human.scaling]),
            rpy=np.array([0, 0, 0]),
            gait_phase_value=0
        )

    def get_position(self):
        return
