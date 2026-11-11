from fractions import Fraction
import unittest

from gemmlowp_model import (
    INT32_MIN, INT32_MAX, dot_output, high_mul, requantize, rounding_right_shift,
)


class GemmlowpTests(unittest.TestCase):
    def test_high_multiply_ties_and_saturation(self):
        self.assertEqual(high_mul(1, 2**30), 1)
        self.assertEqual(high_mul(-1, 2**30), 0)
        self.assertEqual(high_mul(INT32_MIN, INT32_MIN), INT32_MAX)

    def test_right_shift_has_a_different_tie_rule(self):
        self.assertEqual(rounding_right_shift(1, 1), 1)
        self.assertEqual(rounding_right_shift(-1, 1), -1)
        self.assertEqual(rounding_right_shift(INT32_MIN, 0), INT32_MIN)

    def test_backend_agreement_is_different_from_ideal_agreement(self):
        # The effective exact multiplier is 1/4. Two rounding stages give 1.
        self.assertEqual(dot_output([1], [1], 2**30, 1), 1)
        self.assertEqual(round(Fraction(1, 4)), 0)

    def test_bias_zero_points_and_saturation(self):
        self.assertEqual(dot_output([3], [5], 2**30, 0, zx=1, zw=2, bias=2, zy=5), 9)
        self.assertEqual(requantize(10000, 2**30, 0), 127)
        self.assertEqual(requantize(-10000, 2**30, 0), -128)

    def test_unsupported_inputs_are_rejected(self):
        with self.assertRaises(OverflowError):
            high_mul(2**31, 1)
        with self.assertRaises(OverflowError):
            requantize(INT32_MAX, INT32_MAX, 0, 127)
        for shift in (-1, 32):
            with self.assertRaises(ValueError):
                rounding_right_shift(1, shift)
        with self.assertRaises(ValueError):
            requantize(1, -1, 0)


if __name__ == '__main__':
    unittest.main()
