from environment.gen_scene.common_sampler import sg_in_distance_sampler, sg_corner_sampler, sg_opposite_baffle_sampler, \
    point_sampler, sg_opposite_baffle_sampler2

SamplerClassMapping = {
    "sg_in_distance_sampler": sg_in_distance_sampler,
    "sg_corner_sampler": sg_corner_sampler,
    "sg_opposite_baffle_sampler": sg_opposite_baffle_sampler,
    "sg_opposite_baffle_sampler2": sg_opposite_baffle_sampler2,
    "point_sampler": point_sampler
}


def get_sampler_class(sampler_name):
    return SamplerClassMapping[sampler_name]
