import numpy as np

from collision import *

array = np.array


def profileStepLogic(city):
    num_s = 3
    s_loc1 = [
        array([0.5, 0.5, 2]),
        array([0.5, 2.5, 2]),
        array([1.5, 0.5, 2])
    ]
    s_speed1 = [
        2 * array([0, 1, 0]),
        2 * array([0, -1, 0]),
        2 * array([0, 1, 0])
    ]
    # surfaces = compute_building_surfaces(self.city)
    # bs_info, bs_aabb = compute_surfaces_from_buildings(surfaces)
    bs_info, bs_aabb = compute_city_info(5, city)
    s_radius = 0.5
    s_loc2, s_speed2, recorders = step_spheres(
        num_spheres=num_s,
        spheres_location1=s_loc1,
        spheres_speeds=s_speed1,
        buildings_info=bs_info,
        buildings_aabb=bs_aabb,
        spheres_radius=s_radius
    )
    # print(f"next location: \n{array(s_loc2)}\n")
    # print(f"next speed: \n{array(s_speed2)}\n")
    # loc_recorder, s_recorder, e_recorder = recorders
    # print(f"loc recorder length: {len(loc_recorder)}")
    # print(f"event recorder: \n{array(e_recorder)}\n")


def main():

    city = array([[0, 0, 0],
                  [0, 3, 1],
                  [0, 0, 0]], dtype=np.int32)

    profileStepLogic(city)


if __name__ == "__main__":
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.strip_dirs()
    stats.print_stats()
