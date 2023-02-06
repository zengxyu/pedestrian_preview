#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : nav-learning 
    @Author  : Xiangyu Zeng
    @Date    : 5/10/22 9:03 PM 
    @Description    :
        
===========================================
"""

import numpy as np


def convolve_map(input_map, window=5):
    output_map = np.zeros(input_map.shape, dtype=bool)
    for i in range(input_map.shape[0]):
        for j in range(input_map.shape[1]):
            output_map[i, j] = np.any(
                input_map[
                max(0, i - window + 1): min(i + window, input_map.shape[0]),
                max(0, j - window + 1): min(j + window, input_map.shape[1]),
                ]
            )
    return output_map


def fill_gaps(occ_map, window=5):
    occ_map = convolve_map(occ_map, window=window)
    out_map = np.zeros(occ_map.shape, dtype=bool)
    # in the window range, all of the cells are true, then out_map[i, j] is true
    for i in range(occ_map.shape[0]):
        for j in range(occ_map.shape[1]):
            out_map[i, j] = np.logical_and.reduce(
                occ_map[
                max(0, i - window + 1): min(i + window, occ_map.shape[0]),
                max(0, j - window + 1): min(j + window, occ_map.shape[1]),
                ],
                None,
            )

    return out_map


def get_borders(occ_map):
    out_map = occ_map.copy()
    for i in range(1, occ_map.shape[0] - 1):
        for j in range(1, occ_map.shape[1] - 1):
            # in the window range, all of the cells are true, then out_map[i, j] is false;
            # which means if a cell is in the internal area, then, the cell would be false(free)
            out_map[i, j] = (
                False
                if np.logical_and.reduce(
                    occ_map[i - 1: i + 2, j - 1: j + 2],
                    None,
                )
                else occ_map[i, j]
            )
    return out_map


def compute_neighborhood(occ_map, count=4):
    out_map = occ_map.copy()
    for i in range(occ_map.shape[0]):
        for j in range(occ_map.shape[1]):
            value = np.sum(
                occ_map[
                max(0, i - 1): min(i + 2, occ_map.shape[0]),
                max(0, j - 1): min(j + 2, occ_map.shape[1]),
                ],
            )
            out_map[i, j] = (value == count) or (value == (count + 1))
    return out_map


def evolve_neighbors(occ_map, x, y):
    if occ_map[x, y]:
        occ_map[x, y] = False
        neighbors = [
            [x, max(y - 1, 0)],
            [x, min(y + 1, occ_map.shape[1] - 1)],
            [max(x - 1, 0), y],
            [min(x + 1, occ_map.shape[0] - 1), y],
        ]
        for neighbor in neighbors:
            occ_map = evolve_neighbors(occ_map, *neighbor)
    return occ_map


def make_door(occ_map, out_map, door_list, x, y, conf, res):
    width = int(0.5 * conf["doors"] / res)
    narrow_width = int(conf["distance"] / res)
    if x > 0 and x < (occ_map.shape[0] - 1):
        if occ_map[x - 1, y] and occ_map[x + 1, y]:
            if y > 0:
                if occ_map[x, y - 1]:
                    y_ = y - 1
                    while (
                            occ_map[x, y_]
                            and (not occ_map[x - 1, y_])
                            and (not occ_map[x + 1, y_])
                            and y_ > 0
                    ):
                        y_ -= 1
                    if occ_map[x + 1, y_] or occ_map[x - 1, y_] or occ_map[x, y_ - 1]:
                        y_ = int((y + y_) * 0.5)
                        out_map[x, y_ - width: y_ + width + 1] = False
                        door_list.append(
                            [[x, y_], [x + narrow_width, y_], [x - narrow_width, y_]]
                        )
            if y < (occ_map.shape[1] - 1):
                if occ_map[x, y + 1]:
                    y_ = y + 1
                    while (
                            occ_map[x, y_]
                            and (not occ_map[x - 1, y_])
                            and (not occ_map[x + 1, y_])
                            and y_ < (occ_map.shape[1] - 1)
                    ):
                        y_ += 1
                    if occ_map[x + 1, y_] or occ_map[x - 1, y_] or occ_map[x, y_ + 1]:
                        y_ = int((y + y_) * 0.5)
                        out_map[x, y_ - width: y_ + width + 1] = False
                        door_list.append(
                            [[x, y_], [x + narrow_width, y_], [x - narrow_width, y_]]
                        )
    if y > 0 and y < (occ_map.shape[1] - 1):
        if occ_map[x, y - 1] and occ_map[x, y + 1]:
            if x > 0:
                if occ_map[x - 1, y]:
                    x_ = x - 1
                    while (
                            occ_map[x_, y]
                            and (not occ_map[x_, y - 1])
                            and (not occ_map[x_, y + 1])
                            and x_ > 0
                    ):
                        x_ -= 1
                    if occ_map[x_, y + 1] or occ_map[x_, y - 1] or occ_map[x_ - 1, y]:
                        x_ = int((x + x_) * 0.5)
                        out_map[x_ - width: x_ + width + 1, y] = False
                        door_list.append(
                            [[x_, y], [x_, y + narrow_width], [x_, y - narrow_width]]
                        )
            if x < (occ_map.shape[0] - 1):
                if occ_map[x + 1, y]:
                    x_ = x + 1
                    while (
                            occ_map[x_, y]
                            and (not occ_map[x_, y - 1])
                            and (not occ_map[x_, y + 1])
                            and x_ < (occ_map.shape[0] - 1)
                    ):
                        x_ += 1
                    if occ_map[x_, y + 1] or occ_map[x_, y - 1] or occ_map[x_ + 1, y]:
                        x_ = int((x + x_) * 0.5)
                        out_map[x_ - width: x_ + width + 1, y] = False
                        door_list.append(
                            [[x_, y], [x_, y + narrow_width], [x_, y - narrow_width]]
                        )
    return out_map, door_list


def get_narrow_map(occ_map, conf, res):
    out_map = np.zeros(shape=occ_map.shape, dtype=bool)
    narrow_width = int(conf["distance"] / res)
    blocked_width = conf["blocked_width"]
    for i in range(occ_map.shape[0]):
        for j in range(occ_map.shape[1]):
            if not occ_map[i, j]:
                if np.any(occ_map[max(i - narrow_width, 0): i, j]) and np.any(
                        occ_map[
                        min(i + 1, occ_map.shape[0]): min(
                            i + 1 + narrow_width, occ_map.shape[0]
                        ),
                        j,
                        ]
                ):
                    out_map[
                    i - blocked_width: i + 1 + blocked_width,
                    j - narrow_width: j + narrow_width + 1,
                    ] = True
                if np.any(occ_map[i, max(j - narrow_width, 0): j]) and np.any(
                        occ_map[
                        i,
                        min(j + 1, occ_map.shape[1]): min(
                            j + 1 + narrow_width, occ_map.shape[1]
                        ),
                        ]
                ):
                    out_map[
                    i - narrow_width: i + narrow_width + 1,
                    j - blocked_width: j + 1 + blocked_width,
                    ] = True
    # show_image(plt, out_map)
    return out_map


def get_corridor_map(occ_map, door_list, conf):
    out_map = np.zeros(shape=occ_map.shape, dtype=bool)
    blocked_width = 2  # conf["blocked_width"]
    for tripel in door_list:
        for other_tripel in door_list:
            if tripel is other_tripel:
                continue

            initial_index = np.argmin(
                np.linalg.norm(
                    np.broadcast_to(np.array(tripel), (3, 3, 2)).reshape(9, 2)
                    - np.repeat(np.array(other_tripel), 3, axis=0),
                    axis=1,
                )
            )
            for index_shift in range(9):
                x, y = tripel[((initial_index + index_shift) % 9) % 3]
                other_x, other_y = other_tripel[
                    int(((initial_index + index_shift) % 9) / 3)
                ]

                dx, dy = other_x - x, other_y - y
                step_x = 1 if x < other_x else -1
                step_y = 1 if y < other_y else -1
                direct_list = []
                for current_x in range(x, other_x, step_x):
                    direct_list.append(
                        [
                            current_x,
                            y + int((current_x - x) * dy / dx),
                            y + int((current_x + step_x - x) * dy / dx) + step_y,
                        ]
                    )

                x, X = min(x, other_x), max(x, other_x)
                y, Y = min(y, other_y), max(y, other_y)
                width_x, width_y = (
                    (blocked_width, 0) if abs(dx) < abs(dy) else (0, blocked_width)
                )
                collision_list = []
                for x_, y_, y__ in direct_list:
                    collision_list.append(
                        np.any(
                            occ_map[
                            x_ - width_x: x_ + 1 + width_x,
                            y_ - width_y: y__ + width_y,
                            ]
                        )
                    )
                    if collision_list[-1]:
                        break
                if direct_list and not np.any(collision_list):
                    for x_, y_, y__ in direct_list:
                        out_map[
                        x_ - width_x: x_ + 1 + width_x,
                        y_ - width_y: y__ + width_y,
                        ] = True
                    break
                elif not (
                        np.any(
                            occ_map[x - blocked_width: x + 1 + blocked_width, y: Y + 1]
                        )
                        or np.any(
                    occ_map[x: X + 1, Y - blocked_width: Y + 1 + blocked_width]
                )
                ):
                    out_map[x - blocked_width: x + 1 + blocked_width, y: Y + 1] = True
                    out_map[x: X + 1, Y - blocked_width: Y + 1 + blocked_width] = True
                    break
                elif not (
                        np.any(
                            occ_map[X - blocked_width: X + 1 + blocked_width, y: Y + 1]
                        )
                        or np.any(
                    occ_map[x: X + 1, y - blocked_width: y + 1 + blocked_width]
                )
                ):
                    out_map[X - blocked_width: X + 1 + blocked_width, y: Y + 1] = True
                    out_map[x: X + 1, y - blocked_width: y + 1 + blocked_width] = True
                    break
    return out_map


def is_door_neighbor(door_map: np.array, point: np.array, surr_radius: float) -> bool:
    """
    # for dilated occupancy map, if over than 0.7 * total_pixels around the one pixel,
    # this pixel is a door neighbor
    :return:
    """
    # 检查周围是否存在门
    om_i, om_j = point[0], point[1]
    surr_min_i = int(max(om_i - surr_radius, 0))
    surr_max_i = int(min(om_i + surr_radius, door_map.shape[0]))
    surr_min_j = int(max(om_j - surr_radius, 0))
    surr_max_j = int(min(om_j + surr_radius, door_map.shape[1]))

    # # compute occupied pixels
    total_occupied_pixels = np.sum(door_map[surr_min_i:surr_max_i, surr_min_j:surr_max_j])

    # for dilated occupancy map, if over than 0.7 * total_pixels around the one pixel,
    # this pixel is a door neighbor
    is_door_neighbor_flag = total_occupied_pixels > 0

    return is_door_neighbor_flag


def make_exit_door(occ_map, conf, res):
    exit_door_width = int(conf["exit_door_width"] / res)
    half_width = int(exit_door_width / 2)
    max_num_doors = int(conf["max_num_doors"])
    if max_num_doors <= 0:
        return
    num_doors = np.random.randint(1, max_num_doors)
    edge_corner_map = compute_edge_neighbor(occ_map)
    evolve_corner_map = evolve_edge_corner_map(edge_corner_map, exit_door_width=exit_door_width)
    up_available_points = []
    down_available_points = []
    left_available_points = []
    right_available_points = []
    for i in range(occ_map.shape[0]):
        if not evolve_corner_map[i, 0]:
            up_available_points.append([i, 0])
        if not evolve_corner_map[i, occ_map.shape[1] - 1]:
            down_available_points.append([i, occ_map.shape[1] - 1])
    for j in range(occ_map.shape[1]):
        if not evolve_corner_map[0, j]:
            left_available_points.append([0, j])
        if not evolve_corner_map[occ_map.shape[0] - 1, j]:
            right_available_points.append([occ_map.shape[0] - 1, j])

    # sample one point from each edge
    edges = [0, 1, 2, 3]
    sampled_edges = np.random.choice(edges, num_doors, replace=False)
    door_points = []
    for edge_index in sampled_edges:
        if edge_index == 0 and len(up_available_points) != 0:
            i, j = up_available_points[np.random.randint(0, len(up_available_points))]
        elif edge_index == 1 and len(down_available_points) != 0:
            i, j = down_available_points[np.random.randint(0, len(down_available_points))]
        elif edge_index == 2 and len(left_available_points) != 0:
            i, j = left_available_points[np.random.randint(0, len(left_available_points))]
        elif edge_index == 3 and len(right_available_points) != 0:
            i, j = right_available_points[np.random.randint(0, len(right_available_points))]
        else:
            raise NotImplementedError
        door_points.append([i, j])

    # update exit door on occupancy map
    for i, j in door_points:
        if i == 0 or i == occ_map.shape[0] - 1:
            occ_map[i, j - half_width:j + half_width + 1] = 0

        elif j == 0 or j == occ_map.shape[1] - 1:
            occ_map[i - half_width:i + half_width + 1, j] = 0


def compute_edge_neighbor(occ_map, count=4):
    """
    compute the corner on four edges
    """
    edge_corner_map = occ_map.copy()
    for i in range(occ_map.shape[0]):
        for j in [0, occ_map.shape[1] - 1]:
            value = np.sum(
                occ_map[
                max(0, i - 1): min(i + 2, occ_map.shape[0]),
                max(0, j - 1): min(j + 2, occ_map.shape[1]),
                ],
            )
            edge_corner_map[i, j] = (value == count) or (value == (count + 1))

    for i in [0, occ_map.shape[0] - 1]:
        for j in range(occ_map.shape[1]):
            value = np.sum(occ_map[
                           max(0, i - 1): min(i + 2, occ_map.shape[0]),
                           max(0, j - 1): min(j + 2, occ_map.shape[1]),
                           ])
            edge_corner_map[i, j] = (value == count) or (value == (count + 1))

    return edge_corner_map


def evolve_edge_corner_map(edge_corner_map, exit_door_width):
    """
    extend corner to both sides
    """
    half_width = int(exit_door_width / 2)
    edge_corner_map_copy = edge_corner_map.copy()
    for i in range(edge_corner_map.shape[0]):
        for j in [0, edge_corner_map.shape[1] - 1]:
            if edge_corner_map[i, j]:
                edge_corner_map_copy[max(0, i - half_width):min(i + half_width + 1, edge_corner_map.shape[0]), j] = 1

    for i in [0, edge_corner_map.shape[0] - 1]:
        for j in range(edge_corner_map.shape[1]):
            if edge_corner_map[i, j]:
                edge_corner_map_copy[i, max(0, j - half_width):min(j + half_width + 1, edge_corner_map.shape[1])] = 1
    return edge_corner_map_copy
