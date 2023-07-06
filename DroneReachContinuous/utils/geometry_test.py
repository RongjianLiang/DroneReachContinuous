import unittest

import numpy as np

from geometry import *
from miscellaneous import *
array = np.array


class TestGeometry(unittest.TestCase):

    def setUp(self):
        self.buildings_test = [np.array([[0, 0, 0],
                                         [0, 3, 1],
                                         [0, 0, 0]], dtype=np.int32),
                               np.array([[0, 0, 0],
                                         [0, 3, 0],
                                         [0, 0, 0]], dtype=np.int32)
                               ]

        self.lines_test = [
            [np.array([0.5, 1.5, 2]), np.array([1.5, 1.5, 2])],
            [np.array([1.5, 0.5, 2]), np.array([1.5, 1.5, 2])],
            [np.array([1.5, 0.5, 4]), np.array([1.5, 1.5, 2])],
            [np.array([1.5, 0.5, 2]), np.array([1.5, 1.5, 4])],
            [np.array([0.5, 1.5, 2]), np.array([1.5, 1.5, 4])],
            [np.array([0.5, 1.5, 4]), np.array([1.5, 1.5, 2])],
            [np.array([2.5, 1.5, 2]), np.array([0.5, 1.5, 4])],
            [np.array([1.5, 2.5, 2]), np.array([1.5, 1.5, 4])],
            [np.array([1.5, 2.5, 4]), np.array([1.5, 1.5, 4])]
        ]

        self.lines_intersection_test = [
            {
                "seg1": [np.array([0, 0, 0]), np.array([2, 2, 0])],
                "seg2": [np.array([2, 0, 0]), np.array([0, 2, 0])],
                "result": np.array([1, 1, 0])
            },
            {
                "seg1": [np.array([1, 0, 0]), np.array([1, 2, 0])],
                "seg2": [np.array([0, 1, 0]), np.array([2, 1, 0])],
                "result": np.array([1, 1, 0])
            },
            {
                "seg1": [np.array([1, 0, 0]), np.array([1, 2, 0])],
                "seg2": [np.array([1, 0, 0]), np.array([1, 2, 0])],
                "result": None
            },
            {
                "seg1": [np.array([1, 0, 0]), np.array([1, 2, 0])],
                "seg2": [np.array([2, 0, 0]), np.array([2, 2, 0])],
                "result": None
            },
            {
                "seg1": [np.array([0, 1, 0]), np.array([2, 1, 0])],
                "seg2": [np.array([0, 2, 0]), np.array([2, 2, 0])],
                "result": None
            },
            {
                "seg1": [np.array([0, 1, 0]), np.array([2, 1, 0])],
                "seg2": [np.array([0, 1, 0]), np.array([2, 1, 0])],
                "result": None
            },
            {
                "seg1": [np.array([1, 1, 0]), np.array([2, 1, 0])],
                "seg2": [np.array([1, 2, 0]), np.array([1, 3, 0])],
                "result": None
            },
            {
                "seg1": [np.array([1, 2, 0]), np.array([2, 2, 0])],
                "seg2": [np.array([1, 0, 0]), np.array([1, 1, 0])],
                "result": None
            },
            {
                "seg1": [np.array([3, 2, 0]), np.array([4, 3, 0])],
                "seg2": [np.array([1, 2, 0]), np.array([0, 3, 0])],
                "result": None
            },
            {
                "seg1": [np.array([0, 0, 0]), np.array([1, 1, 0])],
                "seg2": [np.array([4, 1, 0]), np.array([3, 1, 0])],
                "result": None
            },
            {
                "seg1": [np.array([0, 1, 0]), np.array([2, 1, 0])],
                "seg2": [np.array([0, 0, 0]), np.array([0, 2, 0])],
                "result": np.array([0, 1, 0])
            },
            {
                "seg1": [np.array([1, 1, 0]), np.array([2, 2, 0])],
                "seg2": [np.array([1, 1, 0]), np.array([0, 2, 0])],
                "result": np.array([1, 1, 0])
            }
        ]

        self.test_sphere_radius = 0.3

        self.sphere_test_cases = [
            {"start1": np.array([0, 0, 0]), "end1": np.array([0, 2, 0]),
             "start2": np.array([0, 2, 0]), "end2": np.array([0, 0, 0]),
             "speed1": np.array([0, 1, 0]),
             "speed2": np.array([0, -1, 0]),
             "radius1": 0.5,
             "radius2": 0.5},
            {"start1": np.array([0, 2, 0]), "end1": np.array([2, 0, 0]),
             "start2": np.array([2, 2, 0]), "end2": np.array([0, 0, 0]),
             "speed1": 1 * (np.array([2, 0, 0]) - np.array([0, 2, 0])),
             "speed2": 1 * (np.array([0, 0, 0]) - np.array([2, 2, 0])),
             "radius1": 0.5,
             "radius2": 0.5}
        ]


    @unittest.skip
    def test_compute_buildings_surface(self):
        for buildings in self.buildings_test:
            with self.subTest(buildings=buildings):
                surfaces = compute_building_surfaces(buildings)
                buildings_dict = get_building_dict(buildings)
                num_buildings = len(buildings_dict.keys())
                max_number_of_walls = 5 * num_buildings
                self.assertIsNotNone(surfaces,
                                     "Incorrect results: surfaces are not None")
                self.assertTrue(len(surfaces) <= max_number_of_walls,
                                "Incorrect number of walls: exceeding maximum")

    @unittest.skip
    def test_compute_surfaces_from_buildings(self):
        buildings = self.buildings_test[1]
        with self.subTest(buildings=buildings):
            surfaces = compute_building_surfaces(buildings)
            print("\n")
            print(surfaces)
            print("\n")
            surface_rep_dict = compute_surfaces_from_buildings(surfaces)
            self.assertIsNotNone(surface_rep_dict,
                                 "Incorrect result: surfaces are not None")
            # use -u option to enable display of print result
            for b in surface_rep_dict.keys():
                rep = surface_rep_dict[b]
                for s_type in rep.keys():
                    s_info = rep[s_type]
                    print(f"{s_type} : [{s_info[0]}, {s_info[1]}, {s_info[2]}])")

    @unittest.skip
    def test_line_surface_contact(self):
        for buildings in self.buildings_test:
            with self.subTest(buildings=buildings):
                surfaces = compute_building_surfaces(buildings)
                surface_rep = compute_surfaces_from_buildings(surfaces, option="simple")
                check_res = []
                for rep in surface_rep:
                    for line in self.lines_test:
                        line_start = line[0]
                        line_end = line[1]
                        centroid = rep[1]
                        normal = rep[0]
                        intersection_point = find_line_surface_intersection(
                            line_start, line_end, normal, centroid
                        )
                        check_res.append(intersection_point)
                        # to see the print result
                        # run with option "-u"
                        # if intersection_point is not None:
                        #     print(f"surface normal: {normal}, centroid: {centroid}")
                        #     print(f"line start: {line_start}, line_end: {line_end}")
                        #     print(f"intersection: {intersection_point}\n")
                self.assertTrue(len(check_res) >= 1,
                                "Incorrect check result: no intersection")

    @unittest.skip
    def test_sphere_surface_contact(self):
        for buildings in self.buildings_test:
            with self.subTest(buildings=buildings):
                surfaces = compute_building_surfaces(buildings)
                surface_rep = compute_surfaces_from_buildings(surfaces, option="simple")
                check_res = []
                for rep in surface_rep:
                    for line in self.lines_test:
                        line_start = line[0]
                        line_end = line[1]
                        centroid = rep[1]
                        normal = rep[0]
                        sphere_radius = self.test_sphere_radius
                        res = find_sphere_surface_contact(
                            sphere_radius, line_start, line_end, normal, centroid
                        )
                        check_res.append(res)
                self.assertTrue(len(check_res) >= 1,
                                "Incorrect check result: no intersection")
        # print(check_res)

    @unittest.skip
    def test_sphere_collision(self):
        for test_case in self.sphere_test_cases:
            # print(test_case)
            with self.subTest(test_case=test_case):
                start1 = test_case["start1"]
                start2 = test_case["start2"]
                end1 = test_case["end1"]
                end2 = test_case["end2"]
                speed1 = test_case["speed1"]
                speed2 = test_case["speed2"]
                radius1 = test_case["radius1"]
                radius2 = test_case["radius2"]
                collision_result = solve_sphere_collision(
                    start1, end1, start2, end2, speed1, speed2, radius1, radius2
                )

                self.assertIsNotNone(collision_result,
                                     "Incorrect results: no collision")
                # print(collision_result)

    @unittest.skip
    def test_lines_closest_distance(self):
        cases = [
            {
                "line_1": [np.array([1.5, 0.5, 4]), np.array([1.5, 1.5, 2])],
                "line_2": [np.array([1, 1, 3]), np.array([2, 1, 3])]
            },
            {
                "line_1": [np.array([1, 0, 4]), np.array([1, 3, 4])],
                "line_2": [np.array([1, 1, 3]), np.array([2, 1, 3])]
            },
            {
                "line_1": [np.array([1, 0, 4]), np.array([1, 3, 3])],
                "line_2": [np.array([1, 1, 3]), np.array([2, 1, 3])]
            },
            {
                "line_1": [np.array([1, 0, 4]), np.array([2, 3, 3])],
                "line_2": [np.array([1, 1, 3]), np.array([2, 1, 3])]
            },
            {
                "line_1": [np.array([1, 0, 4]), np.array([2, 3, 3])],
                "line_2": [np.array([1, 1, 3]), np.array([2, 2, 3])]
            },
            {
                "line_1": [np.array([1, 0, 4]), np.array([2, 3, 3])],
                "line_2": [np.array([1, 1, 3]), np.array([2, 2, 4])]
            }
        ]
        for case in cases:
            with self.subTest(case=case):
                line_1 = case["line_1"]
                line_2 = case["line_2"]
                res = find_closest_positions(line_1[0], line_1[1], line_2[0], line_2[1])
                print(res)

    # todo: test this one after testing and passing the last one
    @unittest.skip
    def test_find_line_intersection(self):
        cases = self.lines_intersection_test
        for case in cases:
            with self.subTest(case=case):
                seg1_start = case["seg1"][0]
                seg1_end = case["seg1"][1]
                seg2_start = case["seg2"][0]
                seg2_end = case["seg2"][1]
                res = find_line_intersection(
                    seg1_start, seg1_end, seg2_start, seg2_end
                )
                self.assertTrue(np.all(np.equal(res, case["result"])))

    @unittest.skip
    def test_interpolate_line_segment1(self):
        cases = [
            {
                "seg": [np.array([0, 0, 0]), np.array([2, 2, 0])],
                "const_coords": {"x": 1, "y": 1}
            },
            {
                "seg": [np.array([0, 0, 0]), np.array([2, 0, 0])],
                "const_coords": {"x": 1, "y": 1}
            }
        ]
        for case in cases:
            with self.subTest(case=case):
                seg_start = case["seg"][0]
                seg_end = case["seg"][1]
                const_coords = case["const_coords"]
                res = helper_interpolate_line_segment_special(seg_start, seg_end, const_coords)
                print(res)

    @unittest.skip
    def test_find_line_intersection_axis(self):
        cases = [
            {
                # no intersection
                "seg1": [np.array([0, 0, 0]), np.array([2, 2, 0])],
                "seg1_axis_check": [False, False, False],
                "seg2": [np.array([1, 0, 0]), np.array([2, 0, 0])],
                "seg2_axis_check": [True, False, False]
            },
            {
                # one intersection at [1, 1, 0]
                "seg1": [np.array([1, 1, 0]), np.array([1, 3, 0])],
                "seg1_axis_check": [False, True, False],
                "seg2": [np.array([2, 0, 0]), np.array([0, 2, 0])],
                "seg2_axis_check": [False, False, False]
            },
            {
                "seg1": [np.array([1, 1, 0]), np.array([1, 3, 0])],
                "seg1_axis_check": [False, True, False],
                "seg2": [np.array([0, 2, 0]), np.array([2, 2, 0])],
                "seg2_axis_check": [True, False, False]
            }
        ]

        for case in cases:
            with self.subTest(case=case):
                seg1_start = case["seg1"][0]
                seg1_end = case["seg1"][1]
                seg1_axis_check = case["seg1_axis_check"]
                seg2_start = case["seg2"][0]
                seg2_end = case["seg2"][1]
                seg2_axis_check = case["seg2_axis_check"]
                res = helper_line_intersection_axis_special(seg1_start, seg1_end, seg1_axis_check, seg2_start, seg2_end,
                                                            seg2_axis_check)
                print(res)

    @unittest.skip
    def test_find_line_intersection_axis1(self):
        cases = [
            {
                # one intersection at [1, 1, 0]
                "seg1": [np.array([1, 1, 0]), np.array([1, 3, 0])],
                "seg1_axis_check": [False, True, False],
                "seg2": [np.array([2, 0, 0]), np.array([0, 2, 0])],
                "seg2_axis_check": [False, False, False]
            }
        ]

        for case in cases:
            with self.subTest(case=case):
                seg1_start = case["seg1"][0]
                seg1_end = case["seg1"][1]
                seg1_axis_check = case["seg1_axis_check"]
                seg2_start = case["seg2"][0]
                seg2_end = case["seg2"][1]
                seg2_axis_check = case["seg2_axis_check"]
                res = helper_line_intersection_axis_special(seg1_start, seg1_end, seg1_axis_check, seg2_start, seg2_end,
                                                            seg2_axis_check)
                print(res)

    @unittest.skip
    def test_find_line_intersection_axis2(self):
        cases = [
            {
                # one intersection at [1, 1, 0]
                "seg1": [np.array([1, 1, 0]), np.array([1, 3, 0])],
                "seg1_axis_check": [False, True, False],
                "seg2": [np.array([2, 1, 0]), np.array([0, 3, 0])],
                "seg2_axis_check": [False, False, False]
            }
        ]

        for case in cases:
            with self.subTest(case=case):
                seg1_start = case["seg1"][0]
                seg1_end = case["seg1"][1]
                seg1_axis_check = case["seg1_axis_check"]
                seg2_start = case["seg2"][0]
                seg2_end = case["seg2"][1]
                seg2_axis_check = case["seg2_axis_check"]
                res = helper_line_intersection_axis_special(seg1_start, seg1_end, seg1_axis_check, seg2_start, seg2_end,
                                                            seg2_axis_check)
                print(res)

    @unittest.skip
    def test_find_line_intersection_axis3(self):
        cases = [
            {
                "seg1": [np.array([1, 0, 0]), np.array([1, 1, 0])],
                "seg1_axis_check": [False, True, False],
                "seg2": [np.array([1, 2, 0]), np.array([2, 2, 0])],
                "seg2_axis_check": [True, False, False]
            }
        ]

        for case in cases:
            with self.subTest(case=case):
                seg1_start = case["seg1"][0]
                seg1_end = case["seg1"][1]
                seg1_axis_check = case["seg1_axis_check"]
                seg2_start = case["seg2"][0]
                seg2_end = case["seg2"][1]
                seg2_axis_check = case["seg2_axis_check"]
                res = helper_line_intersection_axis_special(seg1_start, seg1_end, seg1_axis_check, seg2_start, seg2_end,
                                                            seg2_axis_check)
                # print(res)

    @unittest.skip
    def test_distance_point_to_line_segment(self):
        cases = [
            {
                "point": np.array([0, 0, 0]),
                "segment": [np.array([1, 0, 0]), np.array([1, 1, 0])],
                "result": 1
            },
            {
                "point": np.array([0, 0, 0]),
                "segment": [np.array([1, 0, 0]), np.array([0, 1, 0])],
                "result": np.sqrt(1 / 2)
            },
            {
                "point": np.array([0, 0, 0]),
                "segment": [np.array([0, 0, 0]), np.array([1, 0, 0])],
                "result": 0
            },
            {
                "point": np.array([0, 0, 0]),
                "segment": [np.array([1, 0, 0]), np.array([1, 0, 0])],
                "result": 1
            }
        ]
        for case in cases:
            with self.subTest(case=case):
                point = case["point"]
                segment_start = case["segment"][0]
                segment_end = case["segment"][1]
                segment_norm = np.linalg.norm(segment_end - segment_start)
                segment_unit_dir = segment_end - segment_start / segment_norm
                res = helper_distance_point_to_line_segment(point, segment_start, segment_end,
                                                            segment_norm, segment_unit_dir)
                self.assertEqual(res, case["result"])

    @unittest.skip
    def test_aabb_intersection(self):
        cases = [
            {"aabb1": {"center": np.array([1, 1, 0]), "dimensions": np.array([1, 1, 1])},
             "aabb2": {"center": np.array([2, 1, 0]), "dimensions": np.array([1, 1, 1])},
             "comments": "case for surface contact",
             "result": False},
            {"aabb1": {"center": np.array([2, 1, 0]), "dimensions": np.array([3, 1, 1])},
             "aabb2": {"center": np.array([2, 1, 0]), "dimensions": np.array([1, 1, 1])},
             "result": True},
            {"aabb1": {"center": np.array([2, 1, 0]), "dimensions": np.array([1, 1, 1])},
             "aabb2": {"center": np.array([2, 1, 0]), "dimensions": np.array([1, 1, 1])},
             "result": True},
            {"aabb1": {"center": np.array([2, 1, 0]), "dimensions": np.array([3, 1, 1])},
             "aabb2": {"center": np.array([2, 1, 0]), "dimensions": np.array([1, 3, 1])},
             "result": True},
            {"aabb1": {"center": np.array([0, 1, 0]), "dimensions": np.array([1, 1, 1])},
             "aabb2": {"center": np.array([2, 1, 0]), "dimensions": np.array([1, 3, 1])},
             "result": False},
            {"aabb1": {"center": np.array([0, 1, 0]), "dimensions": np.array([1, 1, 1])},
             "aabb2": {"center": np.array([2, 1, 0]), "dimensions": np.array([1, 3, 1])},
             "comments": "case for surface contact",
             "result": False},
            {"aabb1": {"center": np.array([2, 1, 0]), "dimensions": np.array([1, 1, 1])},
             "aabb2": {"center": np.array([2, 3, 0]), "dimensions": np.array([1, 1, 1])},
             "result": False},
            {"aabb1": {"center": np.array([2, 1, 0]), "dimensions": np.array([2, 1, 1])},
             "aabb2": {"center": np.array([2, 3, 0]), "dimensions": np.array([1, 3, 1])},
             "comments": "case for surface contact",
             "result": False}
        ]
        for case in cases:
            with self.subTest(case=case):
                aabb1 = [case["aabb1"]["center"], case["aabb1"]["dimensions"]]
                aabb2 = [case["aabb2"]["center"], case["aabb2"]["dimensions"]]
                result = check_aabb_intersection(aabb1, aabb2)
                self.assertEqual(result, case["result"])

    # def test_new_closest_sphere_edge(self):
    #     sphere_edge_case = [
    #         {
    #             "s_1": array([0.5, 0.5, 2]),
    #             "s_2": array([0.5, 1.5, 2]),
    #             "e_1": array([0, 1, 1.5]),
    #             "e_2": array([1, 1, 1.5]),
    #             "radius": 0.5,
    #             "result": [array([0.5, 1, 2]), 0.5]
    #         },
    #         {
    #             "s_1": array([0.5, 0.5, 2]),
    #             "s_2": array([1.5, 0.5, 2]),
    #             "e_1": array([1, 0, 1.5]),
    #             "e_2": array([1, 1, 1.5]),
    #             "radius": 0.5,
    #             "result": [array([1, 0.5, 2]), 0.5]
    #         }
    #     ]
    #     for case in sphere_edge_case:
    #         with self.subTest(case=case):
    #             result = new_find_closest_position_sphere_and_edge(
    #                 sphere_start=case["s_1"],
    #                 sphere_end=case["s_2"],
    #                 edge_start=case["e_1"],
    #                 edge_end=case["e_2"],
    #                 radius=case["radius"]
    #             )
    #             if result is not None:
    #                 coords, dis = result
    #                 r_coords, r_dis = case["result"]
    #                 is_coords_equal = np.all(np.equal(coords, r_coords))
    #                 is_dis_equal = np.all(np.equal(dis, r_dis))
    #                 self.assertTrue(is_coords_equal,
    #                                 f"Incorrect closest position: \n"
    #                                 f"get {coords}, expected: {r_coords}")
    #                 self.assertTrue(is_dis_equal,
    #                                 f"Incorrect closest distance: \n"
    #                                 f"get {dis}, expected: {r_dis}")
    #             else:
    #                 self.assertIsNotNone(result,
    #                                      "Incorrect result: should not be None!")


if __name__ == "__main__":
    unittest.main()
