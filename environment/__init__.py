# from gym.envs.registration import register
#
# from gym_hrl.envs.utils.gen_turtlebot_world import (
#     configs_2021Jan05 as world_configs_square,
#     funcs_2021Jan05 as world_create_funcs_square,
# )
#
# HRL_ENVIRONMENTS = [
#     # Empty turtlebot environment
#     {
#         "id": "turtlebotFlat-v0",
#         "entry_point": "environment.turtlebot_flat_env:TurtlebotFlatEnv",
#         "kwargs": {
#             "render": False,
#             "schedule": False,
#             "evaluation": False,
#             "world_configs": world_configs_square,
#             "world_create_funcs": world_create_funcs_square,
#             "exact_complexity_level": False,
#         },
#     },
#     {
#         "id": "turtlebotFlatGUI-v0",
#         "entry_point": "environment.turtlebot_flat_env:TurtlebotFlatEnv",
#         "kwargs": {
#             "render": True,
#             "schedule": False,
#             "evaluation": True,
#             "world_configs": world_configs_square,
#             "world_create_funcs": world_create_funcs_square,
#             "exact_complexity_level": False,
#         },
#     },
#     {
#         "id": "circleCar-v0",
#         "entry_point": "environment.circle_car:CircleCar",
#         "kwargs": {
#             "render": True
#         },
#     },
#     {
#         "id": "raceCar-v0",
#         "entry_point": "environment.race_car:RaceCar",
#         "kwargs": {
#             "render": True,
#             "human_control": True
#         },
#     },
#
# ]
#
#
# def _register():
#     for env in HRL_ENVIRONMENTS:
#         register(**env)
