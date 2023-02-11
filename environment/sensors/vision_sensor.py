import math

import pybullet as p


class ImageMode:
    DEPTH = "depth"
    RGB = "rgb"
    RGBD = "rgbd"
    ROW = "row"


class VisionSensor:
    def __init__(self, robot_id, sensor_config):
        self.robot_id = robot_id

        self.distance = sensor_config["distance"]
        self.image_width = sensor_config["image_width"]
        self.image_height = sensor_config["image_height"]
        self.fov = sensor_config["fov"]
        self.aspect = sensor_config["aspect"]
        self.nearVal = sensor_config["near_val"]
        self.farVal = sensor_config["far_val"]
        self.shadow = sensor_config["shadow"]
        self.placement_height = sensor_config["placement_height"]

    def get_obs(self):
        # get robot observation
        agent_pos, agent_orn = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(agent_orn)[-1]
        x_eye, y_eye, z_eye = agent_pos
        z_eye += self.placement_height  # make the camera a little higher than the robot

        # compute focusing point of the camera
        x_target = x_eye + math.cos(yaw) * self.distance
        y_target = y_eye + math.sin(yaw) * self.distance
        z_target = z_eye

        view_matrix = p.computeViewMatrix(cameraEyePosition=[x_eye, y_eye, z_eye],
                                          cameraTargetPosition=[x_target, y_target, z_target],
                                          cameraUpVector=[0, 0, 1.0])
        projection_matrix = p.computeProjectionMatrixFOV(fov=self.fov, aspect=self.aspect, nearVal=self.nearVal,
                                                         farVal=self.farVal)
        width, height, rgb_image, depth_image, seg_image = p.getCameraImage(self.image_width, self.image_height,
                                                                            view_matrix,
                                                                            projection_matrix, self.shadow,
                                                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)
        return width, height, rgb_image, depth_image, seg_image
