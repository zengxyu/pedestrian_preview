import itertools
import sys

from matplotlib import pyplot as plt

from environment.gen_scene.gen_map_util import *
from environment.gen_scene.common_sampler import *


def create_u_tunnel_in_office_map():
    grid_resolution = 0.1

    # random_index = np.random.randint(0, len(configs["outer_limit"]))
    # outer_limit = configs["outer_limit"][random_index]
    outer_limit = 7
    min_wall_length = 30
    max_wall_length = 50

    min_hall_width = 16
    max_hall_width = 28

    building_size = 7
    map_size = [70, 70]

    min_thick = 1
    max_thick = 5
    thick = np.random.randint(min_thick, max_thick)
    wall_length = np.random.randint(min_wall_length, max_wall_length)
    occupancy_map = np.zeros(
        (
            int(building_size / grid_resolution),
            int(building_size / grid_resolution),
        ),
        dtype=bool,
    )

    direction = np.random.choice([0, 1, 2, 3])

    ind_xy1 = np.random.randint(thick, map_size[0] - thick)
    ind_xy2 = np.random.randint(thick, map_size[0] - thick)

    while not min_hall_width < abs(ind_xy1 - ind_xy2) < max_hall_width:
        ind_xy1 = np.random.randint(thick, map_size[0] - thick)
        ind_xy2 = np.random.randint(thick, map_size[0] - thick)

    half_thick = int(thick / 2)

    if direction == 0:
        occupancy_map[ind_xy1 - half_thick:ind_xy1 + half_thick + 1, :wall_length] = True
        occupancy_map[ind_xy2 - half_thick:ind_xy2 + half_thick + 1, :wall_length] = True
    elif direction == 2:
        occupancy_map[ind_xy1 - half_thick:ind_xy1 + half_thick + 1, -wall_length:] = True
        occupancy_map[ind_xy2 - half_thick:ind_xy2 + half_thick + 1, -wall_length:] = True
    elif direction == 1:
        occupancy_map[:wall_length, ind_xy1 - half_thick:ind_xy1 + half_thick + 1] = True
        occupancy_map[:wall_length, ind_xy2 - half_thick:ind_xy2 + half_thick + 1] = True
    else:
        occupancy_map[-wall_length:, ind_xy1 - half_thick:ind_xy1 + half_thick + 1] = True
        occupancy_map[-wall_length:, ind_xy2 - half_thick:ind_xy2 + half_thick + 1] = True
    occupancy_map[:, [0, -1]] = True
    occupancy_map[[0, -1], :] = True
    return occupancy_map


def create_office_map(configs):
    logging.info("creating office map")
    has_no_closed_rooms_in_free_space = True
    loop_count = 0
    random_index = np.random.randint(0, len(configs["outer_limit"]))
    outer_limit = configs["outer_limit"][random_index]
    wall_ratio = configs["wall_ratio"][random_index]

    while has_no_closed_rooms_in_free_space:
        # Prevent from falling into an infinite loop
        loop_count += 1
        if loop_count > 100:
            return create_office_map(configs)

        building_size = outer_limit[0] + np.random.random_sample(
            size=(2)
        ) * (outer_limit[1] - outer_limit[0])
        grid_resolution = 2.0 * configs["thickness"]
        max_wall_length = int(max(building_size) / grid_resolution)

        occupancy_map = np.zeros(
            (
                int(building_size[0] / grid_resolution),
                int(building_size[1] / grid_resolution),
            ),
            dtype=bool,
        )
        occupancy_map[:, [0, -1]] = True
        occupancy_map[[0, -1], :] = True

        par_walls, total_walls = 2, 4
        while (
                (np.sum(occupancy_map) / np.prod(occupancy_map.shape))
                < wall_ratio
        ) and total_walls < 100:  # prevent ever loop
            # Sample index
            conv_map = convolve_map(occupancy_map, 12)
            indx, indy = np.where(np.invert(conv_map))  # empty cell index
            if len(indx) == 0:
                break
            cell = np.random.choice(len(indx))
            indx, indy = indx[cell], indy[cell]

            # get number if cells
            # wall_length = configs["wall_length_range"][0] + np.random.random() * (
            #         configs["wall_length_range"][1] - configs["wall_length_range"][0])
            # wall_length = int(wall_length / grid_resolution)
            wall_length = max_wall_length
            # Sample direction and adjust probs
            prob = 0.5 * par_walls / total_walls
            direction = np.random.choice(
                range(4), p=[prob, 0.5 - prob, prob, 0.5 - prob]
            )
            par_walls += direction % 2
            total_walls += 1

            # fill occupancy map
            if direction % 2 == 0:
                # generate wall vertically,
                # if direction == 0, generate wall rightwards;
                # if direction == 2, generate wall leftwards;
                range_ = (
                    range(indx, indx + wall_length, 1)
                    if direction == 0
                    else range(indx, indx - wall_length, -1)
                )
                for i in range_:
                    if conv_map[i, indy]:
                        break
                    occupancy_map[i, indy] = True
            else:
                # generate wall vertically,
                # if direction == 1, generate wall downwards;
                # if direction == 3, generate wall upwards;
                range_ = (
                    range(indy, indy + wall_length, 1)
                    if direction == 1
                    else range(indy, indy - wall_length, -1)
                )
                for i in range_:
                    if conv_map[indx, i]:
                        break
                    occupancy_map[indx, i] = True
        # show_image(plt, occupancy_map)
        room_map = fill_gaps(occupancy_map, 12)
        # show_image(plt, room_map)

        border_map = get_borders(room_map)
        # show_image(plt, occupancy_map)

        occupancy_map = fill_gaps(border_map, 5)
        # show_image(plt, occupancy_map)

        neighbor_map = compute_neighborhood(occupancy_map, count=4)
        inds = [(x, y) for x, y in zip(*np.where(neighbor_map))]
        corner_inds = []
        for (x, y) in inds:
            if ((x - 1, y) in inds) and ((x + 1, y) in inds):
                corner_inds.append((x, y))
            if ((x, y - 1) in inds) and ((x, y + 1) in inds):
                corner_inds.append((x, y))

        # corner_inds = filter_corners_in_same_room(border_map, occupancy_map, corner_inds)
        corner_map = np.zeros_like(occupancy_map)
        for x, y in corner_inds:
            corner_map[x, y] = True
        # show_image(plt, corner_map)

        sys.setrecursionlimit(int(1e6))
        disconnected_map = occupancy_map.copy()

        for (x, y) in corner_inds:
            disconnected_map = evolve_neighbors(disconnected_map, x, y)

        # disconnected_map used for checking if has_no_closed_rooms_in_free_space
        indx_disc, indy_disc = np.where(disconnected_map)
        if not ((len(list(set(indx_disc))) > 1) and (len(list(set(indy_disc))) > 1)):
            has_no_closed_rooms_in_free_space = False

    door_map = occupancy_map.copy()
    door_list = []
    for (x, y) in corner_inds:
        door_map, door_list = make_door(
            occupancy_map, door_map, door_list, x, y, configs, grid_resolution
        )
    occupancy_map = door_map.copy()
    # show_image(plt, occupancy_map)
    make_exit_door(occupancy_map, configs, grid_resolution)
    return occupancy_map


def filter_corners_in_same_room(room_map, occ_map, corners):
    # 去掉边缘
    room_map[0, :] = False
    room_map[room_map.shape[0] - 1, :] = False
    room_map[:, 0] = False
    room_map[:, room_map.shape[1] - 1] = False
    for corner in corners:
        room_map[corner[0], corner[1]] = True

    corners_res = []
    # 是同一个房间的角落
    while len(corners) > 0:
        corner = corners[0]
        corners_res.append(corner)
        corners.remove(corner)
        if len(corners) > 0:
            for temp in corners[::-1]:
                is_neighbor_res = [False]
                is_evolved_neighbor(room_map.copy(), corner, temp, is_neighbor_res)

                if is_neighbor_res[0] and not is_direct_neighbor(corner, temp):
                    corners.remove(temp)

    return corners_res


def show_image(plt, image):
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    for i in range(100):
        create_u_tunnel_in_office_map()
