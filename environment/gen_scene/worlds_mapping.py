from environment.gen_scene.gen_corridor_map import create_corridor_map
from environment.gen_scene.gen_cross_map import create_cross_map
from environment.gen_scene.gen_inclosure_map import create_inclosure_map
from environment.gen_scene.gen_office_map import create_office_map
from environment.gen_scene.gen_open_map import create_open_map

WorldMapClassMapping = {"office": create_office_map,
                        "corridor": create_corridor_map,
                        "cross": create_cross_map,
                        "inclosure": create_inclosure_map,
                        "open": create_open_map

                        }


def get_world_creator_func(scene_name):
    return WorldMapClassMapping[scene_name]
