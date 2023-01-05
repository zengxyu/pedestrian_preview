import random
import copy
import logging
import os

import igibson
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from igibson.external.pybullet_tools.utils import AABB
from igibson.object_states.utils import detect_closeness
from igibson.render.viewer import Viewer
from igibson.robots import REGISTERED_ROBOTS
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.scenes.stadium_scene import StadiumScene
from transforms3d.euler import euler2quat
from igibson.utils.utils import quatToXYZW
from igibson import object_states
from igibson.utils.utils import parse_config
from igibson.tasks.behavior_task import BehaviorTask
from igibson.tasks.dummy_task import DummyTask
from igibson.tasks.dynamic_nav_random_task import DynamicNavRandomTask
from igibson.tasks.interactive_nav_random_task import InteractiveNavRandomTask
from igibson.tasks.point_nav_fixed_task import PointNavFixedTask
from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.tasks.reaching_random_task import ReachingRandomTask
from igibson.tasks.room_rearrangement_task import RoomRearrangementTask
from igibson.envs.igibson_env import iGibsonEnv
from igibson.simulator_vr import SimulatorVR

from environment.gibson.scene.MyInteractiveIndoorScene import MyInteractiveIndoorScene
from environment.gibson.tasks.social_nav_random_task import SocialNavRandomTask
from PIL import Image

from utils.image_utility import dilate_image

log = logging.getLogger(__name__)


class GibsonBaseEnv(iGibsonEnv):
    def __init__(
            self,
            config_file,
            normalize_observation=True,
            *args,
            **kwargs,
    ):
        self.config_file_backup = copy.deepcopy(parse_config(config_file))
        self.log = None
        self._init_logger()

        self.normalize_observation = normalize_observation
        self.observation_space_mean = None
        self.observation_space_delta = None

        self.task_randomization_freq = self.config.get("task_randomization_freq", 1)
        self.scene_randomization_freq = self.config.get("scene_randomization_freq", None)

        self.scene_id_list = self.config.get("scene_id_list", None)

        self.evaluation = False
        self.demonstration_mode = False
        self.state = None

        self.action_timestep_robot = self.config.get("action_timestep_robot", self.action_timestep)
        self.physics_timestep = self.config.get("physics_timestep", self.physics_timestep)
        super().__init__(config_file, *args, **kwargs)

        self.occupancy_map = None
        self.dilated_occupancy_map = None

    def get_occupancy_map(self):
        """
        Add an igibson traversability map to the axes
        :param gray_tone_...:   Zero is for black, 1.0 is for white
        """

        trav_map = self.scene.floor_map[0]
        # plt.imshow(trav_map)
        # plt.show()

        # trav_map_obs = 255 - np.array(Image.open(
        #     os.path.join(igibson.ig_dataset_path, "scenes", self.scene.scene_id, "layout", "floor_trav_no_door_0.png")))
        # trav_map_no_obs = 255 - np.array(Image.open(
        #     os.path.join(igibson.ig_dataset_path, "scenes", self.scene.scene_id, "layout", "floor_trav_no_obj_0.png")))
        # # Color obstacles as gray_tone
        # trav_map_obs[trav_map_obs == 255] = 1
        #
        # # Set the Walls to black
        # trav_map_obs[trav_map_no_obs == 255] = 1
        # plt.imshow(trav_map_obs)
        # plt.show()
        trav_map_copy = trav_map.copy()
        trav_map_copy[trav_map == 255] = 0
        trav_map_copy[trav_map == 0] = 1

        trav_map_copy = trav_map_copy.astype(np.bool)
        # plt.imshow(trav_map_copy)
        # plt.show()
        return trav_map_copy.astype(np.bool)

    def load(self):
        """
        Load the scene and robot specified in the config file.
        """
        if self.config["scene"] == "empty":
            scene = EmptyScene()
        elif self.config["scene"] == "stadium":
            scene = StadiumScene()
        elif self.config["scene"] == "gibson":
            scene = StaticIndoorScene(
                self.config["scene_id"],
                waypoint_resolution=self.config.get("waypoint_resolution", 0.2),
                num_waypoints=self.config.get("num_waypoints", 10),
                build_graph=self.config.get("build_graph", False),
                trav_map_resolution=self.config.get("trav_map_resolution", 0.1),
                trav_map_erosion=self.config.get("trav_map_erosion", 2),
                pybullet_load_texture=self.config.get("pybullet_load_texture", False),
            )
        elif self.config["scene"] == "igibson":
            urdf_file = self.config.get("urdf_file", None)
            if urdf_file is None and not self.config.get("online_sampling", True):
                urdf_file = "{}_task_{}_{}_{}_fixed_furniture".format(
                    self.config["scene_id"],
                    self.config["task"],
                    self.config["task_id"],
                    self.config["instance_id"],
                )
            include_robots = self.config.get("include_robots", True)
            scene = InteractiveIndoorScene(
                self.config["scene_id"],
                urdf_file=urdf_file,
                waypoint_resolution=self.config.get("waypoint_resolution", 0.2),
                num_waypoints=self.config.get("num_waypoints", 10),
                build_graph=self.config.get("build_graph", False),
                trav_map_resolution=self.config.get("trav_map_resolution", 0.1),
                trav_map_erosion=self.config.get("trav_map_erosion", 2),
                trav_map_type=self.config.get("trav_map_type", "with_obj"),
                texture_randomization=self.texture_randomization_freq is not None,
                object_randomization=self.object_randomization_freq is not None,
                object_randomization_idx=self.object_randomization_idx,
                should_open_all_doors=self.config.get("should_open_all_doors", False),
                load_object_categories=self.config.get("load_object_categories", None),
                not_load_object_categories=self.config.get("not_load_object_categories", None),
                load_room_types=self.config.get("load_room_types", None),
                load_room_instances=self.config.get("load_room_instances", None),
                merge_fixed_links=self.config.get("merge_fixed_links", True)
                                  and not self.config.get("online_sampling", False),
                include_robots=include_robots,
            )
            # TODO: Unify the function import_scene and take out of the if-else clauses.
            first_n = self.config.get("_set_first_n_objects", -1)
            if first_n != -1:
                scene._set_first_n_objects(first_n)

        self.simulator.import_scene(scene)

        # Get robot config
        robot_config = self.config["robot"]

        # If no robot has been imported from the scene
        if len(scene.robots) == 0:
            # Get corresponding robot class
            robot_name = robot_config.pop("name")
            assert robot_name in REGISTERED_ROBOTS, "Got invalid robot to instantiate: {}".format(robot_name)
            robot = REGISTERED_ROBOTS[robot_name](**robot_config)

            self.simulator.import_object(robot)

            # The scene might contain cached agent pose
            # By default, we load the agent pose that matches the robot name (e.g. Fetch, BehaviorRobot)
            # The user can also specify "agent_pose" in the config file to use the cached agent pose for any robot
            # For example, the user can load a BehaviorRobot and place it at Fetch's agent pose
            agent_pose_name = self.config.get("agent_pose", robot_name)
            if isinstance(scene, InteractiveIndoorScene) and agent_pose_name in scene.agent_poses:
                pos, orn = scene.agent_poses[agent_pose_name]

                if agent_pose_name != robot_name:
                    # Need to change the z-pos - assume we always want to place the robot bottom at z = 0
                    lower, _ = robot.states[AABB].get_value()
                    pos[2] = -lower[2]

                robot.set_position_orientation(pos, orn)

                if any(
                        detect_closeness(
                            bid, exclude_bodyB=scene.objects_by_category["floors"][0].get_body_ids(), distance=0.01
                        )
                        for bid in robot.get_body_ids()
                ):
                    log.warning("Robot's cached initial pose has collisions.")

        self.scene = scene
        self.robots = scene.robots

    def set_action_timestep(self, timestep):
        """
        :return:  The previous action timestep
        """
        prev = self.simulator.render_timestep

        self.action_timestep = timestep
        self.simulator.render_timestep = timestep

        self.simulator.set_timestep(self.simulator.physics_timestep, timestep)
        self.simulator.physics_timestep_num = self.simulator.render_timestep / self.simulator.physics_timestep
        assert self.simulator.physics_timestep_num.is_integer(), "render_timestep must be a multiple of physics_timestep"
        self.simulator.physics_timestep_num = int(self.simulator.physics_timestep_num)

        return prev

    def _init_logger(self, debug=False):
        if debug:
            loglevel = logging.DEBUG
        else:
            loglevel = logging.INFO

        stream_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(stream_formatter)
        stream_handler.setLevel(loglevel)

        self.log = logging.Logger(name="POC")
        self.log.addHandler(stream_handler)

    def load_task_setup(self):
        """
        Load task setup.
        """
        self.initial_pos_z_offset = self.config.get("initial_pos_z_offset", 0.1)
        # s = 0.5 * G * (t ** 2)
        drop_distance = 0.5 * 9.8 * (self.action_timestep ** 2)
        assert drop_distance < self.initial_pos_z_offset, "initial_pos_z_offset is too small for collision checking"

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(self.config.get("collision_ignore_body_b_ids", []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(self.config.get("collision_ignore_link_a_ids", []))

        # discount factor
        self.discount_factor = self.config.get("discount_factor", 0.99)

        # domain randomization frequency
        self.texture_randomization_freq = self.config.get("texture_randomization_freq", None)
        self.object_randomization_freq = self.config.get("object_randomization_freq", None)

        # task
        if "task" not in self.config:
            self.task = DummyTask(self)
        # ==============================================================================
        # Our own tasks go here:
        # ==============================================================================
        elif self.config["task"] == "social_nav_random":
            self.task = SocialNavRandomTask(self)
        else:
            try:
                import bddl

                with open(os.path.join(os.path.dirname(bddl.__file__), "activity_manifest.txt")) as f:
                    all_activities = [line.strip() for line in f.readlines()]

                if self.config["task"] in all_activities:
                    self.task = BehaviorTask(self)
                else:
                    raise Exception("Invalid task: {}".format(self.config["task"]))
            except ImportError:
                raise Exception("bddl is not available.")

    def load(self):
        """
        Load environment.
        """
        super(iGibsonEnv, self).load()
        self.occupancy_map = self.get_occupancy_map()
        # plt.imshow(self.occupancy_map)
        # plt.show()
        self.dilated_occupancy_map = dilate_image(self.occupancy_map, dilation_size=2)

        # self.load_action_space()
        self.load_task_setup()
        self.load_observation_space()
        self.load_miscellaneous_variables()

    def change_scene(self, scene_id, scene_type=None):
        if self.config["scene_id"] == scene_id:
            self.log.info("Scene {} already loaded.".format(scene_id))
            return False
        self.log.info("Loading Scene {}...".format(scene_id))

        config_new = copy.deepcopy(self.config_file_backup)
        config_new["scene_id"] = scene_id
        config_new["scene"] = scene_type if scene_type is not None else config_new["scene"]

        # Special case fix: Pomaria_2_int has only a bedroom
        if config_new["scene_id"] == "Pomaria_2_int":
            config_new["load_room_types"].append("bedroom")

        self.reload(config_new)

        if isinstance(self.simulator, SimulatorVR):
            self.simulator.main_vr_robot = None

        self.task.save_id = None

        self.update_observation_space()
        self.log.info("Scene {} loaded into environment.".format(scene_id))
        return True

    def sample_scene(self):
        assert isinstance(self.scene_id_list, list), "No scene id list provided for scene randomization!"
        return random.choice(self.scene_id_list)

    def step(self, action):
        """
        Apply robot's action and return the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        self.current_step += 1
        self.task.step(self)
        if action is not None:
            self.robots[0].apply_action(action)
        collision_links = self.run_simulation()
        self.collision_links = collision_links
        self.collision_step += int(len(collision_links) > 0)

        self.state = self.get_state()
        info = {}
        reward, info = self.task.get_reward(self, collision_links, action, info)
        done, info = self.task.get_termination(self, collision_links, action, info)

        self.populate_info(info)
        if done and self.automatic_reset:
            info["last_observation"] = self.state
            self.state = self.reset()

        if self.current_step % 10 == 0:
            print()
        return self.state, reward, done, info

    def get_state_full(self):
        return super().get_state()

    def compute_observation(self, action):
        return self.get_state()

    def get_robot_pose_and_yaw(self):
        pos = self.robots[0].get_position()
        orn = self.robots[0].get_orientation()
        return pos[0], pos[1], orn[2]

    def update_observation_space(self, task_range=3):
        if hasattr(self, "latent_dim"):
            low = (
                np.array([-task_range] * self.latent_dim),
            )
            high = (
                np.array([+task_range] * self.latent_dim),
            )
        else:
            low, high = np.array([]), np.array([])

        if hasattr(self.task, "task_observation_range"):
            low = low + self.task.task_observation_range[0]
            high = high + self.task.task_observation_range[1]
            self.log.info("Task specific observation space found and added.")

        low, high = np.concatenate(low), np.concatenate(high)

        self.observation_space = Box(low, high)
        self.log.info("Observation space shape is now {}".format(self.observation_space.shape))
        self.observation_space_delta = self.observation_space.high - self.observation_space.low
        self.observation_space_mean = self.observation_space.low + self.observation_space_delta / 2

    def test_valid_position(self, obj, pos, orn=None, ignore_self_collision=False):
        """
        Test if the robot or the object can be placed with no collision.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param ignore_self_collision: whether the object's self-collisions should be ignored.
        :return: whether the position is valid
        """
        floor_ind = self.scene.world_to_map(np.array(pos[:2]))
        trav_map = self.scene.floor_map[0]

        if trav_map[floor_ind[0], floor_ind[1]] == 0:
            # Non traversible!
            return False

        return super().test_valid_position(obj, pos, orn, ignore_self_collision)

    def set_pos_orn_with_z_offset(self, obj, pos, orn=None, offset=None):
        """
        Reset position and orientation for the robot or the object.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param offset: z offset
        """
        if orn is None:
            orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

        if offset is None:
            offset = self.initial_pos_z_offset

        # first set the correct orientation
        obj.set_position(pos)
        obj.set_orientation(quatToXYZW(euler2quat(*orn), "wxyz"))
        # get the AABB in this orientation
        lower, _ = obj.states[object_states.AABB].get_value()
        # Get the stable Z
        stable_z = pos[2] + (pos[2] - lower[2])
        # change the z-value of position with stable_z + additional offset
        # in case the surface is not perfect smooth (has bumps)
        obj.set_position([pos[0], pos[1], stable_z + offset])
