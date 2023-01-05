from typing import List

import numpy as np
import rvo2
from pybullet_utils.bullet_client import BulletClient

from environment.robots.pedestrian_walker import Pedestrian
from environment.nav_utilities.coordinates_converter import cvt_to_bu
from environment.nav_utilities.debug_plot_helper import plot_line
from environment.gen_env.gen_office_map import distant_point_sampler
import threading


class PedestrianGroup:
    def __init__(self, p: BulletClient, client_id: int, grid_res: float, occ_map: List, pedestrian_num: int):
        """orca also use bullet/world coordinates"""
        self.p = p
        self.client_id = client_id
        self.grid_res = grid_res
        self.occ_map = occ_map
        self.pedestrian_num = pedestrian_num

        self.pedestrians = []
        self.orca_agents = []
        self.threads = []

        self.orca_sim = None

        self.generate_pedestrians()
        self.initialize_orca_sim()

        self.global_step_num = 0

    def move(self):
        print("=========================================")
        for i, pedestrian in enumerate(self.pedestrians):
            t = threading.Thread(target=self.step, args=(i, pedestrian,))
            self.threads.append(t)
            t.start()

    def step(self, i, pedestrian: Pedestrian):
        while True:
            alpha = 0.1
            if pedestrian.reached():
                pedestrian.turn()
            # constant speed
            velocity = alpha * (
                    np.array(pedestrian.e_pose) - np.array(pedestrian.s_pose))

            self.orca_sim.setAgentPrefVelocity(i, tuple(velocity))
            speed = np.linalg.norm(velocity)
            pedestrian.set_velocity(speed)
            pedestrian.step()
            self.orca_sim.doStep()

            # print('step=%2i  t=%.3f  %s' % (self.global_step_num, self.orca_sim.getGlobalTime(), '  '.join(str(positions))))

    def initialize_orca_sim(self):
        # max speed : The default maximum speed of a new agent.
        # radius : The default radius of a new agent.
        # params : neighbor_dist, max_neighbors, time_horizon, time_horizon_obst
        orca_param = [0.01, 5, 1.5, 2]
        radius = 1
        max_speed = 10
        w, h = np.shape(self.occ_map)
        self.orca_sim = rvo2.PyRVOSimulator(1. / 240., *orca_param, radius, max_speed)

        # add pedestrians
        for pedestrian in self.pedestrians:
            # self.sim.setAgentPosition(i + 1, human_state.position)
            agent = self.orca_sim.addAgent(tuple(pedestrian.s_bu_pose), *orca_param, radius, max_speed, (0, 0))
            self.orca_agents.append(agent)
        print('Simulation has %i agents and %i obstacle vertices in it.' %
              (self.orca_sim.getNumAgents(), self.orca_sim.getNumObstacleVertices()))
        print('Running simulation')

    def generate_pedestrians(self):
        """

        :return:
        """
        w, h = np.shape(self.occ_map)
        distance = int(0.5 * min(w, h))

        # new pedestrians
        for i in range(self.pedestrian_num):
            s_p = distant_point_sampler(self.occ_map)
            e_p = distant_point_sampler(self.occ_map, from_point=s_p, distance=distance)

            s_pose = cvt_to_bu(s_p, self.grid_res)
            g_pose = cvt_to_bu(e_p, self.grid_res)
            # gid, g_pose = place_goal(self.p, cvt_2_bullet_coord(e_p, self.grid_res), height=3)

            pedestrian = Pedestrian(self.p, s_p, e_p, s_pose, g_pose)

            plot_line(self.p, [*s_pose, 0.5], [*g_pose, 0.5])

            self.pedestrians.append(pedestrian)

# if __name__ == '__main__':
#     from pybullet_utils import bullet_client
#     import pybullet as p
#     import pybullet_data
#     import os
#     import pickle
#
#     p = bullet_client.BulletClient(connection_mode=p.GUI)
#     client_id = p._client
#
#     p.setAdditionalSearchPath(pybullet_data.getDataPath())
#     plane_id = p.loadURDF("plane.urdf", physicsClientId=client_id)
#
#     path = os.path.join(get_project_path(), "environment/env_test/output/scene/office_world.obj")
#     _, obstacle_ids, full_occupancy_map, grid_resolution = pickle.load(open(path, 'rb'))
#
#     pedestrian_group = PedestrianGroup(p, client_id, grid_resolution, full_occupancy_map, pedestrian_num=5)
#     pedestrian_group.move()
