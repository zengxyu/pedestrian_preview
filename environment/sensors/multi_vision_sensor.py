import math

import numpy as np
import pybullet as p


class ImageMode:
    DEPTH = "depth"
    RGB = "rgb"
    RGBD = "rgbd"
    ROW = "row"
    MULTI_ROW = "multi_row"


class MultiVisionSensor:
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
        self.num_camera = sensor_config["num_camera"]

    def get_obs(self):
        # get robot observation
        agent_pos, agent_orn = p.getBasePositionAndOrientation(self.robot_id)

        delta_yaws = [2 * np.pi / self.num_camera * i for i in range(self.num_camera)]

        cur_yaw = p.getEulerFromQuaternion(agent_orn)[-1]
        rgb_images = []
        depth_images = []
        seg_images = []
        width = 0
        height = 0
        for delta_yaw in delta_yaws:
            yaw = cur_yaw + delta_yaw
            x_eye, y_eye, z_eye = agent_pos
            z_eye += self.placement_height  # make the camera a little higher than the robot

            # compute focusing point of the camera
            x_target = x_eye + math.cos(yaw) * self.distance
            y_target = y_eye + math.sin(yaw) * self.distance
            z_target = z_eye

            view_matrix = p.computeViewMatrix(cameraEyePosition=[x_eye, y_eye, z_eye],
                                              cameraTargetPosition=[x_target, y_target, -20],
                                              cameraUpVector=[0, 0, 1.0])
            projection_matrix = p.computeProjectionMatrixFOV(fov=self.fov, aspect=self.aspect, nearVal=self.nearVal,
                                                             farVal=self.farVal)
            width, height, rgb_image, depth_image, seg_image = p.getCameraImage(self.image_width, self.image_height,
                                                                                view_matrix,
                                                                                projection_matrix, self.shadow,
                                                                                renderer=p.ER_BULLET_HARDWARE_OPENGL)
            rgb_images.append(rgb_image)
            depth_images.append(depth_image)
            seg_images.append(seg_image)

        return width, height, np.array(rgb_images), np.array(depth_images), np.array(seg_images)
