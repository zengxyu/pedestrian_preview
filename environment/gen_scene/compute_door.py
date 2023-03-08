import numpy as np


def compute_door(occupancy_map):
    h, w = occupancy_map.shape

    door_mask = np.ones_like(occupancy_map)
    door_mask[0, :] = occupancy_map[0, :]
    door_mask[h - 1, :] = occupancy_map[h - 1, :]
    door_mask[:, 0] = occupancy_map[:, 0]
    door_mask[:, w - 1] = occupancy_map[:, w - 1]

    indxs, indys = np.where(np.invert(door_mask))

    door_cx = np.mean(indxs)
    door_cy = np.mean(indys)
    return np.array([door_cx, door_cy])


def compute_doors(occupancy_map):
    h, w = occupancy_map.shape

    door_mask = np.ones_like(occupancy_map)
    door_mask[0, :] = occupancy_map[0, :]
    door_mask[h - 1, :] = occupancy_map[h - 1, :]
    door_mask[:, 0] = occupancy_map[:, 0]
    door_mask[:, w - 1] = occupancy_map[:, w - 1]

    indxs, indys = np.where(np.invert(door_mask))
    ind_x = indxs[0]
    max_x = ind_x
    ind_y = indys[0]
    max_y = ind_y
    MAXX, MAXY = np.shape(occupancy_map)[0], np.shape(occupancy_map)[1]
    while np.logical_and.reduce(occupancy_map[ind_x + 1: ind_x + 1 + 1, ind_y: max_x + 1]) and max_x + 1 < MAXX:
        max_x += 1
    while np.logical_and.reduce(occupancy_map[minx - 1: minx, miny: maxy + 1]) and minx - 1 >= 0:
        minx -= 1
    while np.logical_and.reduce(occupancy_map[ind_x + 1: ind_x + 1 + 1, ind_y: max_y + 1]) and max_y + 1 < MAXY:
        max_y += 1
    door_cx = np.mean(indxs)
    door_cy = np.mean(indys)
    return np.array([door_cx, door_cy])
