import unittest
from math_operations import add_numbers, subtract_numbers, multiply_numbers, divide_numbers


class TestMathOperations(unittest.TestCase):
    def test_add_numbers(self):
        result = add_numbers(3, 4)
        self.assertEqual(result, 7)

    def test_subtract_numbers(self):
        result = subtract_numbers(10, 5)
        self.assertEqual(result, 5)

    def test_multiply_numbers(self):
        result = multiply_numbers(2, 3)
        self.assertEqual(result, 6)

    def test_divide_numbers(self):
        result = divide_numbers(10, 2)
        self.assertEqual(result, 5)

    def test_divide_by_zero(self):
        with self.assertRaises(ValueError):
            divide_numbers(10, 0)


if __name__ == '__main__':
    unittest.main()