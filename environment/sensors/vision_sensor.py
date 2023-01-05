import math

import pybullet as p


class VisionSensor:
    def __int__(self, robot_id, sensor_config):
        self.robot_id = robot_id

        self.distance = sensor_config["distance"]
        self.image_width = sensor_config["image_width"]
        self.image_height = sensor_config["image_height"]
        self.fov = sensor_config["fov"]
        self.aspect = sensor_config["aspect"]
        self.nearVal = sensor_config["near_val"]
        self.farVal = sensor_config["far_val"]
        self.shadow = sensor_config["shadow"]

    def get_obs(self):
        # get robot observation
        agent_pos, agent_orn = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(agent_orn)[-1]
        x_eye, y_eye, z_eye = agent_pos
        z_eye += 0.3  # make the camera a little higher than the robot

        # compute focusing point of the camera
        distance = 1
        x_target = x_eye + math.cos(yaw) * distance
        y_target = y_eye + math.sin(yaw) * distance
        z_target = z_eye

        view_matrix = p.computeViewMatrix(cameraEyePosition=[x_eye, y_eye, z_eye],
                                          cameraTargetPosition=[x_target, y_target, z_target],
                                          cameraUpVector=[0, 0, 1.0])
        projection_matrix = p.computeProjectionMatrixFOV(fov=90, aspect=1.5, nearVal=0.02, farVal=3.5)
        images = p.getCameraImage(self.image_width, self.image_height, view_matrix, projection_matrix, self.shadow,
                                  renderer=p.ER_BULLET_HARDWARE_OPENGL)
        return images

    def get_rgb(self):
        return

    def get_depth(self):
        return
