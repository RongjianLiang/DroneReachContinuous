import unittest

import numpy as np

from physics import *
array = np.array


class TestPhysics(unittest.TestCase):

    def setUp(self):

        self.sphere_impact_test_cases = [
            {
                "s_speed": array([0, 2, 0]),
                "s_coords": array([2, 2, 0]),
                "t": 0.5,
                "i_type": "x-z",
                "answer":{
                    "s_new_speed": array([0, -2, 0]),
                    "s_new_loc": array([2, 2, 0])
                }
            }
        ]
        self.collision_testcases = [
            {
                "s1_coords": array([2.5, 2, 0]),
                "s1_speed": array([2, 0, 0]),
                "s1_t": 0.5,
                "s2_coords": array([3.5, 2, 0]),
                "s2_speed": array([-2, 0, 0]),
                "s2_t": 0.5,
                "answer": {
                    "s1_new_speed": array([-2, 0, 0]),
                    "s1_new_coords": array([1.5, 2, 0]),
                    "s2_new_speed": array([2, 0, 0]),
                    "s2_new_coords": array([4.5, 2, 0])
                }
            },
            {
                "comments": "off-center, opposite direction",
                "s1_coords": array([1, 3, 0]),
                "s1_speed": array([1, 0, 0]),
                "s1_t": 0.5,
                "s2_coords": array([2, 2, 0]),
                "s2_speed": array([-1, 0, 0]),
                "s2_t": 0.5
            },
            {
                "comments": "off center, non-axial speeds",
                "s1_coords": array([1, 3, 0]),
                "s1_speed": array([1, -0.5, 0]),
                "s1_t": 0.5,
                "s2_coords": array([2, 2, 0]),
                "s2_speed": array([-1, 0.5, 0]),
                "s2_t": 0.5
            }
        ]

    def testResolveImpact(self):
        for case in self.sphere_impact_test_cases:
            with self.subTest(case=case):
                s_new_speed, s_new_loc = resolve_sphere_impact(
                    case["s_speed"], case["s_coords"],
                    case["t"], case["i_type"]
                )
                if "answer" in case.keys():
                    res1 = np.equal(s_new_speed, case["answer"]["s_new_speed"])
                    res2 = np.equal(s_new_loc, case["answer"]["s_new_loc"])
                    self.assertTrue(np.all(res1))
                    self.assertTrue(np.all(res2))
                else:
                    print(f"s_new_speed: {s_new_speed}, s_new_loc: {s_new_loc}")

    def testResolveCollision(self):
        for case in self.collision_testcases:
            with self.subTest(case=case):
                s1_new_speed, s2_new_speed, s1_new_coords, s2_new_coords = resolve_sphere_collision(
                    case["s1_coords"], case["s1_speed"], case["s1_t"],
                    case["s2_coords"], case["s2_speed"], case["s2_t"]
                )
                if "answer" in case.keys():
                    res1 = np.equal(s1_new_speed, case["answer"]["s1_new_speed"])
                    res2 = np.equal(s2_new_speed, case["answer"]["s2_new_speed"])
                    res3 = np.equal(s1_new_coords, case["answer"]["s1_new_coords"])
                    res4 = np.equal(s2_new_coords, case["answer"]["s2_new_coords"])
                    self.assertTrue(np.all(res1))
                    self.assertTrue(np.all(res2))
                    self.assertTrue(np.all(res3))
                    self.assertTrue(np.all(res4))
                else:
                    comments = case["comments"]
                    print(f"test type: {comments}\n "
                          f"s1_new_speed: {s1_new_speed}, s2_new_speed: {s2_new_speed}\n"
                          f"s1_new_coords: {s1_new_coords}, s2_new_coords: {s2_new_coords}\n")


if __name__ == "__main__":
    unittest.main()
