import os
import numpy as np

import pybullet as p

import igibson
from igibson.objects.object_base import BaseObject
from igibson.objects.stateful_object import StatefulObject


class Pedestrian(StatefulObject):
    """
    """

    def __init__(self, style="standing", pos=[0, 0, 0], position_offset=[0, 0, 0], scale=0.86, dynamic=True, speed=1,
                 exclude_from_physics=False, **kwargs):
        super(Pedestrian, self).__init__(**kwargs)
        self.collision_filename = os.path.join(
            igibson.assets_path, "models", "person_meshes", "person_{}".format(style), "meshes", "person_vhacd.obj"
        )
        self.visual_filename = os.path.join(
            igibson.assets_path, "models", "person_meshes", "person_{}".format(style), "meshes", "person.obj"
        )
        self.cid = None
        self.id = None
        self.pos = pos
        self.position_offset = position_offset

        self.default_orn_euler = np.array([np.pi / 2.0, 0.0, np.pi / 2.0])
        self.scale = scale

        self.hidden = False

        # Parameters for dynamic simulation
        # ================================================
        self.dynamic = dynamic
        self.speed = speed if dynamic else None
        self.exclude_from_physics = exclude_from_physics

        self.view_offset = np.pi / 2

    def _load(self, simulator):
        """
        Load the object into pybullet
        """
        # Check whether the pedestrian should be included in physics sim object
        # ================================================
        mass = 0
        # flags = p.GEOM_FORCE_CONCAVE_TRIMESH if self.exclude_from_physics else None
        flags = p.GEOM_FORCE_CONCAVE_TRIMESH

        collision_id = p.createCollisionShape(p.GEOM_MESH, fileName=self.collision_filename, meshScale=[self.scale] * 3)
        visual_id = p.createVisualShape(p.GEOM_MESH, fileName=self.visual_filename, meshScale=[self.scale] * 3)
        body_id = p.createMultiBody(
            basePosition=[0, 0, 0],
            baseMass=mass,
            flags=flags,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id
        )
        self.id = body_id
        p.resetBasePositionAndOrientation(body_id, self.pos, [-0.5, -0.5, -0.5, 0.5])
        self.cid = p.createConstraint(
            body_id,
            -1,
            -1,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            self.pos,
            parentFrameOrientation=[-0.5, -0.5, -0.5, 0.5],
        )  # facing x axis

        simulator.load_object_in_renderer(self, body_id, self.class_id, **self._rendering_params)

        return [body_id]

    def reset_position_orientation(self, pos, orn):
        """
        Reset pedestrian position and orientation by changing constraint
        """
        self.pos = pos
        p.changeConstraint(self.cid, pos, orn)

    def set_position(self, pos):
        self.pos = pos
        pos = np.array(pos)
        super().set_position(pos)

    # def get_position(self):
    #     """Get object position in the format of Array[x, y, z]"""
    #     return self.get_position_orientation()[0] - np.array(self.position_offset)

    def get_position(self):
        # cur_position, cur_orientation_quat = p.getBasePositionAndOrientation(self.id)
        #  np.array([cur_position[0], cur_position[1]])
        return self.get_position_orientation()[0]

    def set_yaw(self, yaw):
        yaw = yaw + self.view_offset
        cur_orn = super().get_orientation()
        cur_orn_euler = p.getEulerFromQuaternion(cur_orn)

        euler_angle = [cur_orn_euler[0],
                       cur_orn_euler[1],
                       # cur_orn_euler[2] + yaw]
                       yaw]

        self.set_orientation(p.getQuaternionFromEuler(euler_angle))

    def get_yaw(self):
        quat_orientation = super().get_orientation()

        # Euler angles in radians ( roll, pitch, yaw )
        euler_orientation = p.getEulerFromQuaternion(quat_orientation)

        yaw = euler_orientation[2]  # - self.default_orn_euler[2]
        return yaw - self.view_offset

    def set_speed(self, speed):
        assert self.dynamic, "Cannot set speed for a statis pedestrian."
        self.speed = speed

    def get_speed(self, ):
        return self.speed

# import os
# import igibson
# from igibson.objects.object_base import BaseObject
# import pybullet as p
# import numpy as np

# class Pedestrian(BaseObject):
#     """
#     Pedestiran object
#     """

#     def __init__(self, style='standing', scale=1.0, visual_only=True):
#         super(Pedestrian, self).__init__()
#         self.collision_filename = os.path.join(
#             igibson.assets_path, 'models', 'person_meshes',
#             'person_{}'.format(style), 'meshes', 'person_vhacd.obj')
#         self.visual_filename = os.path.join(
#             igibson.assets_path, 'models', 'person_meshes',
#             'person_{}'.format(style), 'meshes', 'person.obj')
#         self.visual_only = visual_only
#         self.scale = scale
#         self.default_orn_euler = np.array([np.pi / 2.0, 0.0, np.pi / 2.0])

#     def _load(self):
#         """
#         Load the object into pybullet
#         """
#         collision_id = p.createCollisionShape(
#             p.GEOM_MESH,
#             fileName=self.collision_filename,
#             meshScale=[self.scale] * 3)
#         visual_id = p.createVisualShape(
#             p.GEOM_MESH,
#             fileName=self.visual_filename,
#             meshScale=[self.scale] * 3)
#         if self.visual_only:
#             body_id = p.createMultiBody(baseCollisionShapeIndex=-1,
#                                         baseVisualShapeIndex=visual_id)
#         else:
#             body_id = p.createMultiBody(baseMass=60,
#                                         baseCollisionShapeIndex=collision_id,
#                                         baseVisualShapeIndex=visual_id)
#         p.resetBasePositionAndOrientation(
#             body_id,
#             [0.0, 0.0, 0.0],
#             p.getQuaternionFromEuler(self.default_orn_euler)
#         )
#         return body_id

#     def set_yaw(self, yaw):
#         euler_angle = [self.default_orn_euler[0],
#                        self.default_orn_euler[1],
#                        self.default_orn_euler[2] + yaw]
#         pos, _ = p.getBasePositionAndOrientation(self.body_id)
#         p.resetBasePositionAndOrientation(
#             self.body_id, pos, p.getQuaternionFromEuler(euler_angle)
#         )

#     def get_yaw(self):
#         quat_orientation = super().get_orientation()

#         # Euler angles in radians ( roll, pitch, yaw )
#         euler_orientation = p.getEulerFromQuaternion(quat_orientation)

#         yaw = euler_orientation[2] - self.default_orn_euler[2]
#         return yaw
