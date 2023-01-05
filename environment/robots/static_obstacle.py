import logging as logger
from environment.gen_scene.build_office_world import create_cylinder

from environment.robots.base_obstacle import BaseObstacle, BaseObstacleGroup


class StaticObstacle(BaseObstacle):
    def __init__(self, p, occ_map, grid_res, type="static"):
        # bullet_client, occupancy map, grid resolution
        super().__init__()
        self.p = p
        self.occ_map = occ_map
        self.grid_res = grid_res

        # create obstacle entity
        self.bu_start_position = None  # start position on bullet
        self.bu_cur_position = None

        self.obstacle_id = -1
        self.verbose = True
        self.type = type

    def create(self, position):
        self.bu_start_position = position
        self.bu_cur_position = self.bu_start_position
        self.obstacle_id, _ = create_cylinder(self.p, self.bu_start_position, height=1.0, radius=0.1)

    def get_cur_position(self):
        return self.bu_cur_position


class StaticObstacleGroup(BaseObstacleGroup):
    def __init__(self, p, occ_map, grid_res):
        super().__init__()
        self.p = p
        self.occ_map = occ_map
        self.grid_res = grid_res

    def create(self, positions, type):
        for i, position in enumerate(positions):
            obstacle = StaticObstacle(self.p, self.occ_map, self.grid_res, type)
            obstacle.create(position)
            self.obstacles.append(obstacle)
            self.obstacle_ids.append(obstacle.obstacle_id)
            logger.debug("The {}-th obstacle with obstacle id : {} created!".format(i, obstacle.obstacle_id))

        return self
