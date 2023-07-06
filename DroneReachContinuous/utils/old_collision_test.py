import unittest
import numpy as np
from old_collision import *


class TestImpact(unittest.TestCase):
    def setUp(self):
        # init some simple parameters for the test
        self.test_num_agents = 1
        self.test_env_size = 5
        self.test_env_height = 8
        self.spacing_for_drones = 0.4

        self.test_city_map = np.array(
            [[0, 0, 0],
             [0, 3, 0],
             [0, 0, 0]
             ], dtype=int_dtype
        )
        self.interpolation = 20

    def test_x_positive_impact(self):
        test_start_coords = np.array(
            [[0, 1, 2]], dtype=float_dtype
        )
        test_end_coords = np.array(
            [[1.2, 1, 2]], dtype=float_dtype
        )
        spacing = float_dtype(
            (test_end_coords[0][0] - test_start_coords[0][0]) / self.interpolation
        )
        x_paths = np.array(
            [np.linspace(test_start_coords[0][0], test_end_coords[0][0], self.interpolation)]
        )
        y_paths = np.array(
            [np.linspace(test_start_coords[0][1], test_end_coords[0][1], self.interpolation)]
        )
        z_paths = np.array(
            [np.linspace(test_start_coords[0][2], test_end_coords[0][2], self.interpolation)]
        )

        impact_info = impact_check_first(
            num_agents_=self.test_num_agents,
            city_map_=self.test_city_map,
            x_paths=x_paths,
            y_paths=y_paths,
            z_paths=z_paths,
        )

        impact_coords = impact_info[0][0]
        pre_impact_coords = impact_info[0][1]
        full_path_len = impact_info[0][3]
        # print(impact_info)
        self.assertEqual(full_path_len, len(x_paths[0]),
                         "Incorrect full length path")

        self.assertAlmostEqual(impact_coords[0], 1., delta=spacing,
                               msg="Incorrect impact coordinates: not within bounds")
        self.assertAlmostEqual(pre_impact_coords[0], 1., delta=spacing,
                               msg="Incorrect pre-impact coordinates: not within bounds")
        self.assertTrue(impact_coords[0] >= 1,
                        "Incorrect impact coordinates: should be bigger")
        self.assertTrue(pre_impact_coords[0] <= 1,
                        "Incorrect pre-impact coordinates: should be smaller")

    def test_y_positive_impact(self):
        test_start_coords = np.array(
            [[1, 0, 2]], dtype=float_dtype
        )
        test_end_coords = np.array(
            [[1, 1.2, 2]], dtype=float_dtype
        )
        spacing = float_dtype(
            (test_end_coords[0][1] - test_start_coords[0][1]) / self.interpolation
        )
        x_paths = np.array(
            [np.linspace(test_start_coords[0][0], test_end_coords[0][0], self.interpolation)]
        )
        y_paths = np.array(
            [np.linspace(test_start_coords[0][1], test_end_coords[0][1], self.interpolation)]
        )
        z_paths = np.array(
            [np.linspace(test_start_coords[0][2], test_end_coords[0][2], self.interpolation)]
        )

        impact_info = impact_check_first(
            num_agents_=self.test_num_agents,
            city_map_=self.test_city_map,
            x_paths=x_paths,
            y_paths=y_paths,
            z_paths=z_paths,
        )

        impact_coords = impact_info[0][0]
        pre_impact_coords = impact_info[0][1]
        full_path_len = impact_info[0][3]
        # print(impact_info)
        self.assertEqual(full_path_len, len(x_paths[0]),
                         "Incorrect full length path")

        self.assertAlmostEqual(impact_coords[1], 1., delta=spacing,
                               msg="Incorrect impact coordinates: not within bounds")
        self.assertAlmostEqual(pre_impact_coords[1], 1., delta=spacing,
                               msg="Incorrect pre-impact coordinates: not within bounds")
        self.assertTrue(impact_coords[1] >= 1,
                        "Incorrect impact coordinates: should be bigger")
        self.assertTrue(pre_impact_coords[1] <= 1,
                        "Incorrect pre-impact coordinates: should be smaller")

    def test_z_negative_impact(self):
        test_start_coords = np.array(
            [[1, 1, 4]], dtype=float_dtype
        )
        test_end_coords = np.array(
            [[1, 1, 2.9]], dtype=float_dtype
        )
        spacing = float_dtype(
            np.abs(test_start_coords[0][2] - test_end_coords[0][2]) / self.interpolation
        )
        x_paths = np.array(
            [np.linspace(test_start_coords[0][0], test_end_coords[0][0], self.interpolation)]
        )
        y_paths = np.array(
            [np.linspace(test_start_coords[0][1], test_end_coords[0][1], self.interpolation)]
        )
        z_paths = np.array(
            [np.linspace(test_start_coords[0][2], test_end_coords[0][2], self.interpolation)]
        )

        impact_info = impact_check_first(
            num_agents_=self.test_num_agents,
            city_map_=self.test_city_map,
            x_paths=x_paths,
            y_paths=y_paths,
            z_paths=z_paths,
        )

        impact_coords = impact_info[0][0]
        pre_impact_coords = impact_info[0][1]
        full_path_len = impact_info[0][3]
        # print(impact_info)
        self.assertEqual(full_path_len, len(x_paths[0]),
                         "Incorrect full length path")

        self.assertAlmostEqual(impact_coords[2], 3., delta=spacing,
                               msg="Incorrect impact coordinates: not within bounds")
        self.assertAlmostEqual(pre_impact_coords[2], 3., delta=spacing,
                               msg="Incorrect pre-impact coordinates: not within bounds")
        self.assertTrue(impact_coords[2] <= 3,
                        "Incorrect impact coordinates: should be smaller")
        self.assertTrue(pre_impact_coords[2] >= 3,
                        "Incorrect pre-impact coordinates: should be bigger")


if __name__ == "__main__":
    unittest.main()
