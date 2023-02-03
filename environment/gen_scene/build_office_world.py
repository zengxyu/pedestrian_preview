import numpy as np


def drop_walls(_bullet_client, occ_map, resolution, configs):
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
                _bullet_client, [minx, miny], [maxx, maxy], resolution, configs
            )
        )
        occ_map[minx: maxx + 1, miny: maxy + 1] = False

    return obstacles


def place_wall_from_cells(_bullet_client, start, end, resolution, configs):
    # x direction
    x = (start[0] + end[0]) * 0.5 * resolution
    dx = end[0] - start[0]
    dx = configs["thickness"] + (0.0 if dx == 0 else (dx + 1) * resolution * 0.5)

    # y direction
    y = (start[1] + end[1]) * 0.5 * resolution
    dy = end[1] - start[1]
    dy = configs["thickness"] + (0.0 if dy == 0 else (dy + 1) * resolution * 0.5)

    # place wall and return id
    return _bullet_client.createMultiBody(
        0,
        _bullet_client.createCollisionShape(
            shapeType=_bullet_client.GEOM_BOX,
            halfExtents=[dx, dy, configs["height"] * 0.5],
            collisionFramePosition=[x, y, configs["height"] * 0.5],
        ),
        _bullet_client.createVisualShape(
            shapeType=_bullet_client.GEOM_BOX,
            halfExtents=[dx, dy, configs["height"] * 0.5],
            visualFramePosition=[x, y, configs["height"] * 0.5],
            rgbaColor=[0.8, 0.8, 0.8, 1]
        ),
    )


def add_entity(_bullet_client, pose, configs, size="medium"):
    if size == "large":
        radius = 1.5 * int(configs["thickness"])
    elif size == "medium":
        radius = 1.0 * configs["thickness"]
    else:
        radius = 0.5 * configs["thickness"]

    # indx, indy = point
    collision_shape_id = _bullet_client.createCollisionShape(
        _bullet_client.GEOM_CYLINDER,
        radius=radius,
        height=configs["height"] * 5,
    )

    visual_shape_id = _bullet_client.createVisualShape(
        _bullet_client.GEOM_CYLINDER,
        radius=radius,
        length=configs["height"] * 5,
        rgbaColor=[0, 0, 0, 1]

    )

    oid = _bullet_client.createMultiBody(
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[
            pose[0],
            pose[1],
            0.5,
        ]

    )
    return oid


def clear_world(_bullet_client, obstacles, base_id, goal_id=None):
    if goal_id:
        _bullet_client.removeBody(goal_id)
    for obs in obstacles:
        if obs:
            _bullet_client.removeBody(obs)
    _bullet_client.restoreState(base_id)
    return [], None  # obstacle_ids, goal_id


def create_cylinder(_bullet_client, pose, goal_id=None, height=None, radius=None):
    x, y = pose
    if goal_id:
        _bullet_client.removeBody(goal_id)
    return (
        _bullet_client.createMultiBody(
            0,
            _bullet_client.createCollisionShape(
                _bullet_client.GEOM_CYLINDER,
                radius=radius,
                height=height,
            ),
            basePosition=[x, y, 0.5],
        ),
        (x, y),
    )
