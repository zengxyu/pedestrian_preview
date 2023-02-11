from environment.gen_scene.common_sampler import sg_in_distance_sampler, sg_corner_sampler, point_sampler

SamplerClassMapping = {
    "sg_in_distance_sampler": sg_in_distance_sampler,
    "sg_corner_sampler": sg_corner_sampler,
    "point_sampler": point_sampler
}


def get_sampler_class(sampler_name):
    return SamplerClassMapping[sampler_name]
