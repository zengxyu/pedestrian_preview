import numpy as np


def separate_inner_outer_walls(occ_map):
    outer_occ_map = np.zeros_like(occ_map)
    h, w = occ_map.shape
    outer_occ_map[0, :] = occ_map[0, :]
    outer_occ_map[h - 1, :] = occ_map[h - 1, :]
    outer_occ_map[:, 0] = occ_map[:, 0]
    outer_occ_map[:, w - 1] = occ_map[:, w - 1]
    inner_occ_map = np.zeros_like(occ_map)
    inner_occ_map[1:h - 2, 1:w - 2] = occ_map[1:h - 2, 1:w - 2]
    return outer_occ_map, inner_occ_map


def drop_world_walls(_bullet_client, occ_map, resolution, configs):
    # separate inner walls and outer walls
    outer_occ_map, inner_occ_map = separate_inner_outer_walls(occ_map)
    obstacles = []
    # place outer walls
    outer_obstacles = drop_walls(_bullet_client, outer_occ_map, resolution, configs["thickness"],
                                 configs["outer_height"])
    # place inner walls
    inner_obstacles = drop_walls(_bullet_client, inner_occ_map, resolution, configs["thickness"],
                                 configs["inner_height"])
    obstacles.extend(outer_obstacles)
    obstacles.extend(inner_obstacles)
    return obstacles


def drop_walls(_bullet_client, occ_map, resolution, thickness, height):
    obstacles = []

    MAXX = occ_map.shape[0]
    MAXY = occ_map.shape[1]

    while np.any(occ_map):
        indsx, indsy = np.where(occ_map)
        minx = maxx = indsx[0]
        miny = maxy = indsy[0]

        while np.logical_and.reduce(occ_map[maxx + 1: maxx + 1 + 1, miny: maxy + 1]) and maxx + 1 < MAXX:
            maxx += 1

        while np.logical_and.reduce(occ_map[minx - 1: minx, miny: maxy + 1]) and minx - 1 >= 0:
            minx -= 1

        while np.logical_and.reduce(occ_map[minx: maxx + 1, maxy + 1: maxy + 1 + 1]) and maxy + 1 < MAXY:
            maxy += 1

        while np.logical_and.reduce(occ_map[minx: maxx + 1, miny - 1: miny]) and miny - 1 >= 0:
            miny -= 1

        obstacles.append(
            place_wall_from_cells(
                _bullet_client, [minx, miny], [maxx, maxy], resolution, thickness, height
            )
        )
        occ_map[minx: maxx + 1, miny: maxy + 1] = False

    return obstacles


def place_wall_from_cells(_bullet_client, start, end, resolution, thickness, height):
    # x direction
    x = (start[0] + end[0]) * 0.5 * resolution
    dx = end[0] - start[0]
    dx = thickness + (0.0 if dx == 0 else (dx + 1) * resolution * 0.5)

    # y direction
    y = (start[1] + end[1]) * 0.5 * resolution
    dy = end[1] - start[1]
    dy = thickness + (0.0 if dy == 0 else (dy + 1) * resolution * 0.5)

    # place wall and return id
    return _bullet_client.createMultiBody(
        0,
        _bullet_client.createCollisionShape(
            shapeType=_bullet_client.GEOM_BOX,
            halfExtents=[dx, dy, height * 0.5],
            collisionFramePosition=[x, y, height * 0.5],
        ),
        _bullet_client.createVisualShape(
            shapeType=_bullet_client.GEOM_BOX,
            halfExtents=[dx, dy, height * 0.5],
            visualFramePosition=[x, y, height * 0.5],
            rgbaColor=[0.8, 0.8, 0.8, 1]
        ),
    )



def clear_world(_bullet_client, obstacles, base_id, goal_id=None):
    if goal_id:
        _bullet_client.removeBody(goal_id)
    for obs in obstacles:
        if obs:
            _bullet_client.removeBody(obs)
    _bullet_client.restoreState(base_id)
    return [], None  # obstacle_ids, goal_id


def create_cylinder(_bullet_client, pose, with_collision, goal_id=None, height=None, radius=None, color=None):
    x, y = pose
    if goal_id:
        _bullet_client.removeBody(goal_id)

    visual_shape_id = _bullet_client.createVisualShape(
        _bullet_client.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=color

    )
    if with_collision:
        collision_shape_id = _bullet_client.createCollisionShape(
            _bullet_client.GEOM_CYLINDER,
            radius=radius,
            height=height,
        )

        return _bullet_client.createMultiBody(
            baseVisualShapeIndex=visual_shape_id,
            baseCollisionShapeIndex=collision_shape_id,
            basePosition=[x, y, height / 2],
        )
    else:
        return _bullet_client.createMultiBody(
            baseVisualShapeIndex=visual_shape_id,
            baseCollisionShapeIndex=-1,
            basePosition=[x, y, height / 2],
        )
