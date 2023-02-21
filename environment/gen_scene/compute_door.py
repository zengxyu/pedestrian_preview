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
