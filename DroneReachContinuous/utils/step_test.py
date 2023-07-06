import numpy as np

from collision import *
import unittest
import cProfile
import pstats
import subprocess
from mayavi import mlab
import os

array = np.array

"""
Testing flow and logic for the step_spheres function
"""


class TestSphereStepFlow(unittest.TestCase):

    def setUp(self):
        self.city = array([[0, 0, 0],
                           [0, 3, 1],
                           [0, 0, 0]], dtype=np.int32)

        self.city_bg = np.zeros((3, 3))

    @unittest.skip
    def testComputeCity(self):
        bs_info, bs_aabb = compute_city_info(5, self.city)
        for b in bs_info.keys():
            b_info = bs_info[b]
            print("===============")
            print(f"building centroid: {b}")
            for surface in b_info.keys():
                print(f"--surface facing: {surface}")
                surface_info = b_info[surface]
                for info_key in surface_info.keys():
                    print(f"----{info_key}")
                    _info = surface_info[info_key]
                    if isinstance(_info, np.ndarray):
                        _info = array(_info)
                    print(f"------{_info}")
            print("===============")

    @staticmethod
    def printBuildingAABB(bs_aabb):
        for b in bs_aabb.keys():
            print(f"--building: {b}")
            info = bs_aabb[b]
            centroid, dim = info
            print(f"----centroid: \n ----{centroid}")
            print(f"----dim: \n ----{dim}")

    @unittest.skip
    def testImpactBorder(self):
        # todo: impact is still buggy
        # todo: try it with small spheres and varying speeed conditions
        num_s = 1
        s_loc1 = [
            array([1.5, 0.5, 2])
        ]
        s_speed1 = [
            -2 * array([0, 1, 0])
        ]
        bs_info, bs_aabb = compute_city_info(5, self.city_bg)
        # self.printBuildingAABB(bs_aabb)
        s_radius = 0.5
        s_loc2, s_speed2, recorders = step_spheres(
            num_spheres=num_s,
            spheres_location1=s_loc1,
            spheres_speeds=s_speed1,
            buildings_info=bs_info,
            buildings_aabb=bs_aabb,
            spheres_radius=s_radius
        )
        loc_recorder, s_recorder, e_recorder = recorders
        temp_e_recorder = np.swapaxes(e_recorder, 0, 1)
        print(f"event recorder: \n{array(temp_e_recorder)}\n")
        print(f"loc recorder: ")
        num_frame = 1
        ag_id = 0
        # swap the axes 0 and 1 of the recorder
        loc_temp = np.swapaxes(loc_recorder, 0, 1)
        for ag_trac in loc_temp:
            print(f"---agents: {ag_id}\n")
            print(f"{array(ag_trac)}")
            ag_id += 1

    @unittest.skip
    def testImpact(self):
        # todo: impact is still buggy
        # todo: try it with small spheres and varying speeed conditions
        num_s = 1
        s_loc1 = [
            array([1.5, 0.5, 2])
        ]
        s_speed1 = [
            2 * array([0, 1, 0])
        ]
        bs_info, bs_aabb = compute_city_info(5, self.city)
        # self.printBuildingAABB(bs_aabb)
        s_radius = 0.5
        s_loc2, s_speed2, recorders = step_spheres(
            num_spheres=num_s,
            spheres_location1=s_loc1,
            spheres_speeds=s_speed1,
            buildings_info=bs_info,
            buildings_aabb=bs_aabb,
            spheres_radius=s_radius
        )
        loc_recorder, s_recorder, e_recorder = recorders
        temp_e_recorder = np.swapaxes(e_recorder, 0, 1)
        print(f"event recorder: \n{array(temp_e_recorder)}\n")
        print(f"loc recorder: ")
        num_frame = 1
        ag_id = 0
        # swap the axes 0 and 1 of the recorder
        loc_temp = np.swapaxes(loc_recorder, 0, 1)
        for ag_trac in loc_temp:
            print(f"---agents: {ag_id}\n")
            print(f"{array(ag_trac)}")
            ag_id += 1

    # @unittest.skip
    def testSphereAABB(self):
        s_loc2 = [
            array([4.75, 2.75, 2.5])
        ]
        s_loc1 = [
            array([4.8369565, 2.8369565, 2.5])
        ]
        s_radius = 1
        # todo: the current AABB finder fails at small spheres...
        s_aabb_centroids, s_aabb_dim = find_aabb_for_sphere(
            array(s_loc1), array(s_loc2), s_radius
        )
        print(f"\ns_aabb_centroids: {s_aabb_centroids}\n"
              f"s_aabb_dimension: {s_aabb_dim}")

    @unittest.skip
    def testImpact2(self):
        case_name = "border_impact1"
        _fps = 24
        num_s = 1
        city_size = 5
        city_height = 5
        city = np.zeros((city_size, city_size))
        agents_type = {0: 0}
        s_loc1 = [
            array([4.75, 2.75, 2.5])
        ]
        s_speed1 = [
            4 * array([0.5, 0.5, 0])
        ]
        bs_info, bs_aabb = compute_city_info(5, city)
        s_radius = 0.1
        total_time_step = 2
        total_loc = []
        new_loc, new_speed, recorders = step_spheres(
            city_size=city_size,
            city_height=city_height,
            num_spheres=num_s,
            spheres_location1=s_loc1,
            spheres_speeds=s_speed1,
            buildings_info=bs_info,
            buildings_aabb=bs_aabb,
            spheres_radius=s_radius
        )
        # update loc for next step

        # add constraints for borders ?????
        # extract x loc, y loc, and z loc
        new_loc = array(new_loc)
        # print(f"--new loc: {new_loc}")
        # print(f"--new loc x: {new_loc[:, 0], new_loc[:, 1]}")
        loc_x = new_loc[:, 0]
        loc_y = new_loc[:, 1]
        loc_z = new_loc[:, 2]

        if isinstance(new_loc, np.ndarray):
            new_loc = list(new_loc)
        s_loc1 = new_loc
        s_speed1 = new_speed

        loc_recorder, s_recorder, e_recorder = recorders
        loc_recorder = array(loc_recorder)
        print(f"----loc recorder: \n{loc_recorder}\n")
        total_loc.append(loc_recorder)
        e_recorder = array(e_recorder)
        temp_e_recorder = np.swapaxes(e_recorder, 1, 0)  # shape: (3, 24)

        total_loc = array(total_loc)
        print(f"---total loc: \n{total_loc}\n")
        # temp_e_recorder = np.swapaxes(e_recorder, 0, 1)
        # loc_recorder = array(loc_recorder)
        # print(f"event recorder: \n{array(temp_e_recorder)}\n")
        # print(f"loc recorder: \n{loc_recorder}\n")

    @unittest.skip
    def testStepLogic(self):
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
        bs_info, bs_aabb = compute_city_info(5, self.city)
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
        loc_recorder, s_recorder, e_recorder = recorders
        # todo: define a log-printing function and call it instead
        # print(f"loc recorder length: {len(loc_recorder)}")
        temp_e_recorder = np.swapaxes(e_recorder, 0, 1)
        print(f"event recorder: \n{array(temp_e_recorder)}\n")
        print(f"loc recorder: ")
        num_frame = 1
        ag_id = 0
        # swap the axes 0 and 1 of the recorder
        loc_temp = np.swapaxes(loc_recorder, 0, 1)
        for ag_trac in loc_temp:
            print(f"---agents: {ag_id}\n")
            print(f"{array(ag_trac)}")
            ag_id += 1

    @unittest.skip
    def testBroadPhaseCheck(self):
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
        surfaces = compute_building_surfaces(self.city)
        bs_info, bs_aabb = compute_surfaces_from_buildings(surfaces)
        s_radius = 0.5
        delta_t = 1.0 / 5
        start_loc = array(s_loc1)
        start_speed = array(s_speed1)
        for interval_id in range(5):
            end_loc = start_loc + start_speed * delta_t
            # print(f"==step: {interval_id}, start loc: \n{start_loc}")
            # print(f"==step: {interval_id}, end loc: \n{end_loc}\n")
            s_aabb_centroids, s_aabb_dim = \
                find_aabb_for_sphere(
                    start_loc, end_loc, s_radius
                )

            i_check, c_check = broad_phase_check(
                num_s,
                s_aabb_centroids,
                s_aabb_dim,
                bs_info,
                bs_aabb
            )
            # print(f"--step: {interval_id}, i-check: {i_check}")
            # print(f"++step: {interval_id}, c-check: {c_check}")
            start_loc = end_loc


if __name__ == "__main__":

    # profiler = cProfile.Profile()
    # profiler.enable()
    unittest.main()
    # profiler.disable()
    #
    # # Create a pstats.Stats object
    # stats = pstats.Stats(profiler, stream=open('profile_stats.txt', 'w'))
    #
    # # Print the profiling results
    # stats.print_stats()

    # cProfile.run('unittest.main()', sort='cumtime')


