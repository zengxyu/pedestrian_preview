import numpy as np



def place_wall(_bullet_client, pose_x, pose_y, len_x, len_y, height):
    len_x = (len_x + len_y) if len_x > len_y else len_x
    len_y = (len_x + len_y) if len_y > len_x else len_y
    return _bullet_client.createMultiBody(
        0,
        _bullet_client.createCollisionShape(
            shapeType=_bullet_client.GEOM_BOX,
            halfExtents=[len_x, len_y, height * 0.5],
            collisionFramePosition=[pose_x, pose_y, height * 0.5],
        ),
    )


def create_square(_bullet_client, configs, sampler_type=None):
    sidelength = (
            np.array(
                [
                    configs["limit"][0]
                    + np.random.random_sample()
                    * (configs["limit"][1] - configs["limit"][0])
                    for _ in range(2)
                ]
            )
            * 0.5
    )

    obstacles = []
    obstacles.append(
        place_wall(
            _bullet_client,
            sidelength[0],
            0,
            configs["thickness"],
            sidelength[1],
            configs["height"],
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client,
            -sidelength[0],
            0,
            configs["thickness"],
            sidelength[1],
            configs["height"],
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client,
            0,
            sidelength[1],
            sidelength[0],
            configs["thickness"],
            configs["height"],
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client,
            0,
            -sidelength[1],
            sidelength[0],
            configs["thickness"],
            configs["height"],
        )
    )

    def _sample_helper():
        x = (
                np.random.choice([-1.0, 1.0])
                * np.random.random_sample()
                * (sidelength[0] - configs["distance"])
        )
        y = (
                np.random.choice([-1.0, 1.0])
                * np.random.random_sample()
                * (sidelength[1] - configs["distance"])
        )
        return x, y

    def sampler(x_target=None, y_target=None):
        x, y = x_target, y_target
        while (
                np.sqrt(np.square(x - x_target) + np.square(y - y_target))
                < configs["distance"]
        ):
            x, y = _sample_helper()
        return x, y

    def major_sampler(x=None, y=None, range_=1.0, noise=0.1):
        assert (x is not None) and (y is not None)
        # Computing the overlength in each of the four directions
        # This is the distance the goal would land outside the world,
        # if completely going in the specific direction
        overlength = [
            (
                    range_
                    + configs["distance"]
                    + (1.0 - (2 * int(i / 2))) * [x, y][i % 2]
                    - sidelength[i % 2]
            )
            / range_
            for i in range(4)
        ]

        # Excluding some angle segments,
        # depending on the closeness of the walls
        # as the rooms are big and the overlength is computed
        # counterclockwise only one or two consecutive overlength
        # can exceed the limit
        theta = None
        for i in range(4):
            if overlength[i] > 0:  # current OL exceeds
                if overlength[(i + 1) % 4] > 0:  # next also exceeds
                    # All the following cases consider robot beeing close to corners
                    if overlength[i] > 1.0:  # currently in safety zone of OL
                        if overlength[(i + 1) % 4] > 1.0:  # saftey of next OL
                            # excluding the neighboring 3/2 pi, as would
                            # lead to out of world
                            theta = [((i + 2) / 2) * np.pi, ((i + 3) / 2) * np.pi]
                        else:
                            # Do not exclude the region of the next direction completly
                            theta = [
                                ((i + 1) / 2) * np.pi
                                + np.arccos(1.0 - overlength[(i + 1) % 4]),
                                ((i + 3) / 2) * np.pi,
                            ]
                    else:
                        if overlength[(i + 1) % 4] > 1.0:  # safety of next OL
                            # exclude next but not current direction completely
                            theta = [
                                ((i + 2) / 2) * np.pi,
                                ((i + 4) / 2) * np.pi - np.arccos(1.0 - overlength[i]),
                            ]
                        else:
                            # consider both partially
                            theta_i = np.arccos(1.0 - overlength[i])
                            theta_ip1 = np.arccos(1.0 - overlength[(i + 1) % 4])
                            if (theta_i + theta_ip1) > (0.5 * np.pi):
                                # robot is to close to corner,
                                # no possible poses between robot and corner
                                theta = [
                                    ((i + 1) / 2) * np.pi + theta_ip1,
                                    ((i + 4) / 2) * np.pi - theta_i,
                                ]
                            else:
                                # possible poses between corner and robot
                                theta = [
                                    (i / 2) * np.pi + theta_i,
                                    ((i + 1) / 2) * np.pi - theta_ip1,
                                    ((i + 1) / 2) * np.pi + theta_ip1,
                                    ((i + 4) / 2) * np.pi - theta_i,
                                ]

                else:
                    # Robot is only close to one wall
                    if overlength[i] > 1.0:
                        # Too close to the current one, exlcude pi
                        theta = [((i + 1) / 2) * np.pi, ((i + 3) / 2) * np.pi]
                    else:
                        # consider partially the region towards wall, as enough space
                        theta = np.arccos(1.0 - overlength[i])
                        theta = [(i / 2) * np.pi + theta, ((i + 4) / 2) * np.pi - theta]

                break
        theta = theta or [0, 2 * np.pi]

        if len(theta) == 2:
            theta = theta[0] + np.random.random_sample() * (theta[1] - theta[0])
        else:
            props = [theta[1] - theta[0], theta[3] - theta[2]]
            props = [p / sum(props) for p in props]
            choice = np.random.choice([0, 2], p=props)
            theta = theta[choice] + np.random.random_sample() * (
                    theta[choice + 1] - theta[choice]
            )
        return (
            x + np.cos(theta) * range_ + np.random.normal(0.0, noise),
            y + np.sin(theta) * range_ + np.random.normal(0.0, noise),
        )

    return obstacles, _sample_helper, sampler, major_sampler


def create_tunel(_bullet_client, configs, sampler_type=None):
    width = 0.5 * np.array(
        configs["limit"][0][0]
        + np.random.random_sample() * (configs["limit"][0][1] - configs["limit"][0][0])
    )
    length = 0.5 * np.array(
        configs["limit"][1][0]
        + np.random.random_sample() * (configs["limit"][1][1] - configs["limit"][1][0])
    )

    obstacles = []
    obstacles.append(
        place_wall(
            _bullet_client, width, 0, configs["thickness"], length, configs["height"]
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client, -width, 0, configs["thickness"], length, configs["height"]
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client, 0, length, width, configs["thickness"], configs["height"]
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client, 0, -length, width, configs["thickness"], configs["height"]
        )
    )

    if sampler_type == "data_generator":

        def _sample_helper():
            x = (
                    np.random.choice([-1.0, 1.0])
                    * np.random.random_sample()
                    * (width - configs["distance"])
            )
            y = np.random.random_sample() * (length - configs["distance"])
            return x, y

    else:

        def _sample_helper():
            x = (
                    np.random.choice([-1.0, 1.0])
                    * np.random.random_sample()
                    * (width - configs["distance"])
            )
            offset = configs["offset"] * length
            y = offset + np.random.random_sample() * (
                    length - offset - configs["distance"]
            )
            return x, y

    if sampler_type == "mid_level":

        def sampler_robot(unused1=None, unused2=None):
            x = (
                    np.random.choice([-1.0, 1.0])
                    * np.random.random_sample()
                    * (width - configs["distance"])
            )
            y = (
                    np.random.choice([-1.0, 1.0])
                    * np.random.random_sample()
                    * (length - configs["distance"])
            )
            return x, y

    else:

        def sampler_robot(unused1=None, unused2=None):
            x, y = _sample_helper()
            return x, -y

    def major_sampler(x=None, y=None, range_=1.0, noise=0.1):
        assert (x is not None) and (y is not None)
        # compute start/end angle of greenzone
        # greenzone is defined as the tunel, where the width is shrinken
        # by the safety distance
        dx = [
            ((width - configs["distance"]) + ((2 * i) - 1.0) * x) / range_
            for i in range(2)
        ]
        # Check if greenzone is reachable
        # this only occurs when range is smaller than safety distance
        for d in dx:
            if d <= -1.0:
                raise ValueError("Robot to far from greenzone - rework major_sampler")
        # compute actual angle limits
        # The seqution of the circle around the robot,
        # that is within the green zone
        # this only computes above the robot [0, pi]
        theta = [
            i * np.pi if dx[i] > 1.0 else np.arccos((1.0 - (2 * i)) * dx[i])
            for i in range(2)
        ]

        # Check if collision in y / cut of the end/beginning of tunel
        # done by removing the positive/negative angle region
        choices = [
            (1.0 - (2 * i))
            for i in range(2)
            if ((1.0 - (2 * i)) * y) < (length - configs["distance"] - range_)
        ]

        # Sample theta
        theta = np.random.choice(choices) * (
                theta[0] + np.random.random_sample() * (theta[1] - theta[0])
        )

        return (
            x + np.cos(theta) * range_ + np.random.normal(0.0, noise),
            y + np.sin(theta) * range_ + np.random.normal(0.0, noise),
        )

    return obstacles, _sample_helper, sampler_robot, major_sampler


def create_L(_bullet_client, configs, sampler_type=None):
    hallway_width = np.array(
        configs["limit"][0][0]
        + np.random.random_sample() * (configs["limit"][0][1] - configs["limit"][0][0])
    )
    short_L = np.array(
        configs["limit"][1][0]
        + np.random.random_sample() * (configs["limit"][1][1] - configs["limit"][1][0])
    )
    long_L = np.array(
        configs["limit"][2][0]
        + np.random.random_sample() * (configs["limit"][2][1] - configs["limit"][2][0])
    )
    sign = np.random.choice([-1.0, 1.0])

    obstacles = []
    obstacles.append(
        place_wall(
            _bullet_client,
            0.5 * (hallway_width + short_L),
            0,
            0.5 * (hallway_width + short_L),
            configs["thickness"],
            configs["height"],
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client,
            hallway_width + short_L,
            sign * 0.5 * hallway_width,
            configs["thickness"],
            0.5 * hallway_width,
            configs["height"],
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client,
            hallway_width + 0.5 * short_L,
            sign * hallway_width,
            0.5 * short_L,
            configs["thickness"],
            configs["height"],
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client,
            hallway_width,
            sign * (hallway_width + 0.5 * long_L),
            configs["thickness"],
            0.5 * long_L,
            configs["height"],
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client,
            0.5 * hallway_width,
            sign * (hallway_width + long_L),
            0.5 * hallway_width,
            configs["thickness"],
            configs["height"],
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client,
            0,
            sign * 0.5 * (hallway_width + long_L),
            configs["thickness"],
            0.5 * (hallway_width + long_L),
            configs["height"],
        )
    )

    def sampler_target():
        x = configs["distance"] + np.random.random_sample() * (
                hallway_width - (2 * configs["distance"])
        )
        y = sign * (
                hallway_width
                + configs["distance"]
                + 0.5 * long_L
                + np.random.random_sample() * ((0.5 * long_L) - (2 * configs["distance"]))
        )
        return x, y

    if sampler_type == "data_generator" or sampler_type == "mid_level":
        factor = 0.5 if sampler_type == "data_generator" else 1.0
        prob = (hallway_width + short_L) / (hallway_width + short_L + factor * long_L)

        def sampler_robot(unused1=None, unused2=None):
            index = np.random.choice(range(2), p=(prob, 1.0 - prob))
            x = [
                (
                        configs["distance"]
                        + np.random.random_sample()
                        * (hallway_width + short_L - (2 * configs["distance"]))
                ),
                configs["distance"]
                + np.random.random_sample()
                * (hallway_width - (2 * configs["distance"])),
            ][index]
            y = [
                sign
                * (
                        configs["distance"]
                        + np.random.random_sample()
                        * (hallway_width - (2 * configs["distance"]))
                ),
                sign
                * (
                        hallway_width
                        - configs["distance"]
                        + np.random.random_sample() * factor * long_L
                ),
            ][index]
            return x, y

    else:

        def sampler_robot(unused1=None, unused2=None):
            x = (
                    hallway_width
                    + configs["distance"]
                    + np.random.random_sample() * (short_L - (2 * configs["distance"]))
            )
            y = sign * (
                    configs["distance"]
                    + np.random.random_sample()
                    * (hallway_width - (2 * configs["distance"]))
            )
            return x, y

    def major_sampler(x=None, y=None, range_=1.0, noise=0.1):
        assert (x is not None) and (y is not None)
        # check if in long L or short L
        # if in square crossing area randomly chose one
        region = (
            0
            if ((sign * y) > hallway_width)
            else 1
            if (x > hallway_width)
            else np.random.choice([0, 1])
        )

        # compute start/end angle of greenzone
        # depending on region, the coordinates are rotated or not
        # However both versions can be computed as a tunel
        delta = [
            (
                    [hallway_width - configs["distance"], -configs["distance"]][i]
                    + ((2.0 * i) - 1.0) * [x, sign * y][region]
            )
            / range_
            for i in range(2)
        ]
        # Check if greenzone is reachable
        # only if safety zone exceeds range
        for d in delta:
            if d <= -1.0:
                raise ValueError("Robot to far from greenzone - rework major_sampler")
        # compute actual angle limits
        # this is only the upper half
        theta = [
            i * np.pi if delta[i] > 1.0 else np.arccos((1.0 - (2.0 * i)) * delta[i])
            for i in range(2)
        ]

        # Check if considering positive/negative angles
        # leads to a collision
        # so: exclude end or start of "tunel"
        choices = [
            (1.0 - (2 * i)) * sign
            for i in range(2)
            if [
                [
                    (sign * y)
                    < (hallway_width + long_L - configs["distance"] - range_),
                    (sign * y) > (configs["distance"] + range_),
                ][i],
                [
                    x > (configs["distance"] + range_),
                    x < (hallway_width + short_L - configs["distance"] - range_),
                ][i],
            ][region]
        ]

        # Sample theta
        # Important: rotate the system back if in short L region
        theta = np.random.choice(choices) * (
                theta[0] + np.random.random_sample() * (theta[1] - theta[0])
        ) + (
                        region * sign * 0.5 * np.pi
                )  # rotation here

        return (
            x + np.cos(theta) * range_ + np.random.normal(0.0, noise),
            y + np.sin(theta) * range_ + np.random.normal(0.0, noise),
        )

    return obstacles, sampler_target, sampler_robot, major_sampler


def create_CZ(_bullet_client, configs, sampler_type=None):
    hallway_width = np.array(
        configs["limit"][0][0]
        + np.random.random_sample() * (configs["limit"][0][1] - configs["limit"][0][0])
    )
    short_center = np.array(
        configs["limit"][1][0]
        + np.random.random_sample() * (configs["limit"][1][1] - configs["limit"][1][0])
    )
    short_outer = np.array(
        configs["limit"][2][0]
        + np.random.random_sample() * (configs["limit"][2][1] - configs["limit"][2][0])
    )
    long_mid = np.array(
        configs["limit"][3][0]
        + np.random.random_sample() * (configs["limit"][3][1] - configs["limit"][3][0])
    )
    sign_mid = np.random.choice([-1.0, 1.0])
    sign_outer = np.random.choice([-1.0, 1.0])

    obstacles = []
    obstacles.append(
        place_wall(
            _bullet_client,
            0.5 * (hallway_width + short_center),
            0,
            0.5 * (hallway_width + short_center),
            configs["thickness"],
            configs["height"],
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client,
            hallway_width + short_center,
            sign_mid * 0.5 * hallway_width,
            configs["thickness"],
            0.5 * hallway_width,
            configs["height"],
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client,
            hallway_width + 0.5 * short_center,
            sign_mid * hallway_width,
            0.5 * short_center,
            configs["thickness"],
            configs["height"],
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client,
            hallway_width,
            sign_mid * (hallway_width + 0.5 * long_mid),
            configs["thickness"],
            0.5 * long_mid,
            configs["height"],
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client,
            float(sign_outer > 0.0) * hallway_width + sign_outer * 0.5 * short_outer,
            sign_mid * (hallway_width + long_mid),
            0.5 * short_outer,
            configs["thickness"],
            configs["height"],
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client,
            float(sign_outer > 0.0) * hallway_width + sign_outer * short_outer,
            sign_mid * (1.5 * hallway_width + long_mid),
            configs["thickness"],
            0.5 * hallway_width,
            configs["height"],
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client,
            0.5 * (hallway_width + sign_outer * short_outer),
            sign_mid * (2.0 * hallway_width + long_mid),
            0.5 * (hallway_width + short_outer),
            configs["thickness"],
            configs["height"],
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client,
            float(sign_outer < 0.0) * hallway_width,
            sign_mid * (1.5 * hallway_width + long_mid),
            configs["thickness"],
            0.5 * hallway_width,
            configs["height"],
        )
    )
    obstacles.append(
        place_wall(
            _bullet_client,
            0,
            sign_mid * 0.5 * (hallway_width + long_mid),
            configs["thickness"],
            0.5 * (hallway_width + long_mid),
            configs["height"],
        )
    )

    def sampler_target():
        x = float(sign_outer > 0.0) * hallway_width + sign_outer * (
                configs["distance"]
                + np.random.random_sample() * (short_outer - (2 * configs["distance"]))
        )
        y = sign_mid * (
                hallway_width
                + long_mid
                + configs["distance"]
                + np.random.random_sample() * (hallway_width - (2 * configs["distance"]))
        )
        return x, y

    if sampler_type == "data_generator" or sampler_type == "mid_level":
        prob = (hallway_width + short_center) / (
                short_center + 2 * hallway_width + long_mid
        )

        def sampler_robot(unused1=None, unused2=None):
            index = np.random.choice(range(2), p=(prob, 1.0 - prob))
            x = [
                (
                        configs["distance"]
                        + np.random.random_sample()
                        * (hallway_width + short_center - (2 * configs["distance"]))
                ),
                configs["distance"]
                + np.random.random_sample()
                * (hallway_width - (2 * configs["distance"])),
            ][index]
            y = [
                sign_mid
                * (
                        configs["distance"]
                        + np.random.random_sample()
                        * (hallway_width - (2 * configs["distance"]))
                ),
                sign_mid
                * (
                        hallway_width
                        - configs["distance"]
                        + np.random.random_sample() * (long_mid + hallway_width)
                ),
            ][index]
            return x, y

    else:

        def sampler_robot(unused1=None, unused2=None):
            x = (
                    hallway_width
                    + configs["distance"]
                    + np.random.random_sample() * (short_center - (2 * configs["distance"]))
            )
            y = sign_mid * (
                    configs["distance"]
                    + np.random.random_sample()
                    * (hallway_width - (2 * configs["distance"]))
            )
            return x, y

    def major_sampler(x=None, y=None, range_=1.0, noise=0.1):
        assert (x is not None) and (y is not None)
        # check if in long L or short L
        # if in square crossing area randomly chose one
        region = (
            2
            if ((sign_mid * y) > (hallway_width + long_mid))
            else 0
            if (x < hallway_width)
            else 1
        )

        if region == 2:
            # compute start/end angle of greenzone
            # depending on region, the coordinates are rotated or not
            # However both versions can be computed as a tunel
            delta = [
                (
                        [
                            long_mid + 2 * hallway_width - configs["distance"],
                            -(long_mid + hallway_width + configs["distance"]),
                        ][i]
                        + ((2.0 * i) - 1.0) * sign_mid * y
                )
                / range_
                for i in range(2)
            ]
            # Check if greenzone is reachable
            # only if safety zone exceeds range
            for d in delta:
                if d <= -1.0:
                    raise ValueError(
                        "Robot to far from greenzone - rework major_sampler"
                    )
            # compute actual angle limits
            # this is only the upper half
            theta = [
                i * np.pi if delta[i] > 1.0 else np.arccos((1.0 - (2.0 * i)) * delta[i])
                for i in range(2)
            ]

            # Check if considering positive/negative angles
            # leads to a collision
            # so: exclude end or start of "tunel"
            [start, end] = [
                [
                    configs["distance"] - short_outer,
                    hallway_width - configs["distance"],
                ],
                [
                    configs["distance"],
                    hallway_width + short_outer - configs["distance"],
                ],
            ][int(sign_outer > 0.0)]
            choices = [
                (1.0 - (2 * i)) * sign_mid
                for i in range(2)
                if [x > (start + range_), x < (end - range_)][i]
            ]

            # Sample theta
            # Important: rotate the system back if in short L region
            theta = np.random.choice(choices) * (
                    theta[0] + np.random.random_sample() * (theta[1] - theta[0])
            ) + (
                            sign_mid * 0.5 * np.pi
                    )  # rotation here

            return (
                x + np.cos(theta) * range_ + np.random.normal(0.0, noise),
                y + np.sin(theta) * range_ + np.random.normal(0.0, noise),
            )

        # compute start/end angle of greenzone
        # depending on region, the coordinates are rotated or not
        # However both versions can be computed as a tunel
        delta = [
            (
                    [hallway_width - configs["distance"], -configs["distance"]][i]
                    + ((2.0 * i) - 1.0) * [x, sign_mid * y][region]
            )
            / range_
            for i in range(2)
        ]
        # Check if greenzone is reachable
        # only if safety zone exceeds range
        for d in delta:
            if d <= -1.0:
                raise ValueError("Robot to far from greenzone - rework major_sampler")
        # compute actual angle limits
        # this is only the upper half
        theta = [
            i * np.pi if delta[i] > 1.0 else np.arccos((1.0 - (2.0 * i)) * delta[i])
            for i in range(2)
        ]

        # Check if considering positive/negative angles
        # leads to a collision
        # so: exclude end or start of "tunel"
        choices = [
            (1.0 - (2 * i)) * sign_mid
            for i in range(2)
            if [
                [
                    (sign_mid * y) < (hallway_width + long_mid - range_),
                    (sign_mid * y) > (configs["distance"] + range_),
                ][i],
                [
                    x > (configs["distance"] + range_),
                    x < (hallway_width + short_center - configs["distance"] - range_),
                ][i],
            ][region]
        ]

        # Sample theta
        # Important: rotate the system back if in short L region
        theta = np.random.choice(choices) * (
                theta[0] + np.random.random_sample() * (theta[1] - theta[0])
        ) + (
                        region * sign_mid * 0.5 * np.pi
                )  # rotation here

        return (
            x + np.cos(theta) * range_ + np.random.normal(0.0, noise),
            y + np.sin(theta) * range_ + np.random.normal(0.0, noise),
        )

    return obstacles, sampler_target, sampler_robot, major_sampler


# Environment parameter
configs_all = {"height": 1.0, "thickness": 0.05}
configs_2021Jan05 = [
    # square
    {"limit": [4.0, 10.0], "distance": 0.5, **configs_all},
    {  # tunel
        "limit": [[2.0, 2.5], [6.0, 10.0]],
        "distance": 0.5,
        "offset": 0.5,
        **configs_all,
    },
    {  # L
        "limit": [[2.0, 2.5], [3.0, 4.0], [5.0, 8.0]],
        "distance": 0.5,
        **configs_all,
    },
    {  # CZ
        "limit": [[2.0, 2.5], [3.0, 4.0], [3.0, 4.0], [5.0, 8.0]],
        "distance": 0.5,
        **configs_all,
    },
]
funcs_2021Jan05 = [create_square, create_tunel, create_L, create_CZ]


def create_world(
        _bullet_client,
        func_list,
        configs_list,
        exact=False,
        sampler_type=None,
):
    if exact:
        index = -1
    else:
        index = np.random.choice(range(len(func_list)))
    return func_list[index](_bullet_client, configs_list[index], sampler_type)
