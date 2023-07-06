import unittest
from collision import *
from geometry import *


class TestImpactAndCollision(unittest.TestCase):

    def setUp(self):
        self.buildings_test = [np.array([[0, 0, 0],
                                         [0, 3, 1],
                                         [0, 0, 0]], dtype=np.int32),
                               np.array([[0, 0, 0],
                                         [0, 3, 0],
                                         [0, 0, 0]], dtype=np.int32)
                               ]

        self.lines_test = [
            [np.array([1.5, 0.5, 4]), np.array([1.5, 1.5, 2])],     # 0: [1.5, 1, 3]
            [np.array([1.5, 0.5, 2]), np.array([1.5, 1.5, 4])],     # 1: [1.5, 1, 3]
            [np.array([0.5, 1.5, 2]), np.array([1.5, 1.5, 4])],     # 2: [1, 1.5, 3]
            [np.array([0.5, 1.5, 4]), np.array([1.5, 1.5, 2])],     # 3: [1, 1.5, 3]
            [np.array([2.5, 1.5, 2]), np.array([0.5, 1.5, 4])],     # 4: [2, 1.5, 3] todo: None??
            [np.array([1.5, 2.5, 2]), np.array([1.5, 1.5, 3])],     # 5: [1.5, 2, 2.5]
            [np.array([1.5, 2.5, 2]), np.array([1.5, 1.5, 2])],     # 6: [1.5, 2, 2]
            [np.array([1.5, 8, 4]), np.array([1.5, 1.5, 3])]        # 7:
        ]

        self.sphere_collision_test_cases = [
            {
                "start1": np.array([0, 0, 0]), "end1": np.array([0, 2, 0]),
                "start2": np.array([0, 2, 0]), "end2": np.array([0, 0, 0]),
                "speed1": 1 * (np.array([0, 2, 0]) - np.array([0, 0, 0])),
                "speed2": 1 * (np.array([0, 0, 0]) - np.array([0, 2, 0])),
                "radius1": 0.5, "radius2": 0.5
            },
            {
                "start1": np.array([0, 2, 0]), "end1": np.array([2, 0, 0]),
                "start2": np.array([2, 2, 0]), "end2": np.array([0, 0, 0]),
                "speed1": 1 * (np.array([2, 0, 0]) - np.array([0, 2, 0])),
                "speed2": 1 * (np.array([0, 0, 0]) - np.array([2, 2, 0])),
                "radius1": 0.5, "radius2": 0.5
            }
        ]
        self.surfaces = compute_building_surfaces(self.buildings_test[1])
        self.surfaces_rep = compute_surfaces_from_buildings(self.surfaces)
        self.buildings_dict, _ = \
            compute_surfaces_from_buildings(self.surfaces)

        self.sphere_impact_testcases_with_building_1 = {
            "building": (1.5, 1.5, 1.5),
            "building_info": self.buildings_dict,
            "sphere_radius": 0.5,
            "cases": {
                "lines": self.lines_test,
                "result": [
                            array([1.5, 1, 3]),
                            array([1.5, 1, 3]),
                            array([1, 1.5, 3]),
                            array([1, 1.5, 3]),
                            array([2, 1.5, 3]),
                            array([1.5, 2, 2.5]),
                            array([1.5, 2, 2])
                ]
            }
        }

    @unittest.skip
    def test_sphere_collision(self):
        test_result = []
        for test_case in self.sphere_collision_test_cases:
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
                if collision_result is not None:
                    test_result.append(collision_result)
        self.assertIsNotNone(test_result,
                             "Incorrect results: no collision")

    @unittest.skip
    def test_sphere_impact(self):
        pass
        # for test_cases in self.sphere_impact_testcases:
        #     with self.subTest(test_cases=test_cases):
        #         line_start = test_cases["line_start"]
        #         line_end = test_cases["line_end"]
        #         sphere_radius = test_cases["sphere_radius"]
        #         normal = test_cases["surface_normal"]
        #         centroid = test_cases["surface_centroid"]
        #         height = test_cases["surface_height"]
        #         # print("test case: ", test_cases)
        #         # print("surface rep:", surfaces_rep)
        #         impact_result = find_sphere_surface_impact(sphere_radius, line_start, line_end, normal, centroid, )
        #         # self.assertIsNotNone(impact_result,
        #         #                      "Incorrect impact result, impact should present")
        #         # print("result:", impact_result)
        #         # print("\n")

    def test_sphere_impact_with_buildings(self):
        result = []
        specify_id = 7
        # for test_cases in self.sphere_impact_testcases_with_buildings:
        test_case = self.sphere_impact_testcases_with_building_1
        b = test_case["building"]
        bs_info = test_case["building_info"]
        sphere_radius = test_case["sphere_radius"]
        cases = test_case["cases"]
        if specify_id == -1:
            for case_id in range(len(cases["lines"])):
                with self.subTest(case=cases["lines"][case_id]):

                    line_start = cases["lines"][case_id][0]
                    line_end = cases["lines"][case_id][1]
                    impact_results = sphere_building_impact(
                        sphere_radius, line_start, line_end, b, bs_info
                    )
                    result.append(impact_results)
                    print(f"result: {impact_results}")
                    # res = np.all(np.equal(impact_results[1]["coords"], cases["result"][case_id]))
                    # self.assertTrue(res)
        else:
            case = cases["lines"][specify_id]
            line_start = case[0]
            line_end = case[1]
            s_aabb_centroid, s_aabb_dim = find_aabb_for_sphere(
                line_start, line_end, 0.5
            )
            bs_aabb = helper_find_AABB_for_buildings(self.buildings_test[1])
            # for bpc, we need the b_info and b_aabb with keys being building centroid
            i_check, _ = broad_phase_check(
                1, s_aabb_centroid, s_aabb_dim, bs_info, bs_aabb
            )
            # for detailed check, we single out the building.
            for i_res in i_check:
                _, b_centroid = i_res
                b_info = bs_info[b_centroid]
                impact_results = sphere_building_impact(
                    sphere_radius, line_start, line_end, b_centroid, b_info
                )
                result.append(impact_results)
                print(f"result: {impact_results}")

    @unittest.skip
    def test_closest_geometry(self):
        line1_id = 7
        line1 = self.lines_test[line1_id]
        line2 = [array([2, 2, 3]), array([1, 2, 3])]
        closest_res = find_closest_positions(
            line1[0], line1[1], line2[0], line2[1]
        )
        print(closest_res)


if __name__ == "__main__":
    unittest.main()





