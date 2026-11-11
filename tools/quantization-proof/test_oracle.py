"""Boundary, enclosure and fault-detection tests for the executable oracle."""

from dataclasses import replace
from fractions import Fraction as F
import math
import random
import struct
import unittest

from oracle import (
    GemmCase, Interval, Quantizer, centered_dot_bound, check_gemm,
    check_interval, checked_centered_dot, exact_gemm, gemm_intervals,
)


def scalar_case(x=1, w=1, bias=0, sx=1, sw=1, sb=1, sy=1, zx=0, zy=0):
    return GemmCase(((x,),), ((w,),), (bias,), Quantizer(sx, zx),
                    (Quantizer(sw),), (Quantizer(sb, qmin=-2**31, qmax=2**31-1),),
                    (Quantizer(sy, zy),))


class QuantizerTests(unittest.TestCase):
    def test_round_trip_every_int8_code_and_parameter_grid(self):
        for scale in (F(1, 4), F(1, 3), 0.1):
            for zero in (-128, -1, 0, 1, 127, 200):
                quant = Quantizer(scale, zero)
                for code in range(-128, 128):
                    self.assertEqual(quant.encode(quant.decode(code)), code)

    def test_halfway_sign_parity_and_zero_point_placement(self):
        quant = Quantizer(1)
        for value, expected in ((F(-5, 2), -2), (F(-3, 2), -2), (F(-1, 2), 0),
                                (F(1, 2), 0), (F(3, 2), 2), (F(5, 2), 2)):
            with self.subTest(value=value):
                self.assertEqual(quant.encode(value), expected)
        self.assertEqual(Quantizer(1, 1).encode(F(1, 2)), 2)
        self.assertEqual(round(F(1, 2)) + 1, 1)

    def test_scale_preserves_fp32_storage_value(self):
        stored = struct.unpack('<f', struct.pack('<f', 0.1))[0]
        decimal, binary = Quantizer(F('0.1')), Quantizer(stored)
        self.assertEqual(Quantizer(1).encode(decimal.decode(5)), 0)
        self.assertEqual(Quantizer(1).encode(binary.decode(5)), 1)
        self.assertEqual(binary.scale, F(13421773, 134217728))

    def test_clipping_before_fixed_width_conversion(self):
        self.assertEqual(Quantizer(1).encode(130), 127)
        self.assertEqual(Quantizer(1).encode(-130), -128)
        self.assertEqual(Quantizer(1, qmin=7, qmax=7).encode(1000), 7)

    def test_reject_invalid_parameters_and_input_codes(self):
        for scale in (0, -1, float('inf'), float('nan')):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                Quantizer(scale)
        with self.assertRaises(ValueError):
            Quantizer(1, qmin=2, qmax=1)
        for code in (1.2, True):
            with self.assertRaises(TypeError):
                Quantizer(1).decode(code)
        with self.assertRaises(ValueError):
            Quantizer(1).decode(128)

    def test_relu_exhausts_all_codes_for_multiple_zero_points(self):
        for zero in (-128, -7, 0, 5, 127):
            quant = Quantizer(F(1, 3), zero)
            for code in range(-128, 128):
                expected = quant.encode(max(quant.decode(code), 0))
                self.assertEqual(max(code, zero), expected)
        # The tempting max(q, 0) mutation is observable with nonzero z.
        self.assertNotEqual(max(2, 0), Quantizer(F(1, 4), 5).encode(0))


class IntervalTests(unittest.TestCase):
    def test_outward_enclosure_including_negative_values(self):
        for value in (F(-7, 3), F(-1, 2**26), F(0), F(1, 3), F(10**20, 7)):
            for bits in (0, 4, 24):
                interval = Interval.enclose(value, bits)
                self.assertLessEqual(interval.lower, value)
                self.assertGreaterEqual(interval.upper, value)
                self.assertLessEqual(interval.upper - interval.lower, F(1, 2**bits))
                negative = interval.times_integer(-3)
                self.assertLessEqual(negative.lower, -3 * value)
                self.assertGreaterEqual(negative.upper, -3 * value)

    def test_three_verdicts_and_exact_tie(self):
        quant = Quantizer(1)
        ambiguous = Interval(F(1, 2), F(1, 2) + F(1, 2**25))
        for candidate in (0, 1):
            self.assertEqual(check_interval(ambiguous, quant, candidate).verdict, 'inconclusive')
        self.assertEqual(check_interval(ambiguous, quant, 2).verdict, 'fail')
        tie = Interval(F(1, 2), F(1, 2))
        self.assertEqual(check_interval(tie, quant, 0).verdict, 'pass')
        self.assertEqual(check_interval(tie, quant, 1).verdict, 'fail')

    def test_saturation_certificate_crosses_rounding_boundaries(self):
        for bounds, expected in (((128, 130), 127), ((-140, -129), -128)):
            result = check_interval(Interval(*bounds), Quantizer(1), expected)
            self.assertEqual(result.verdict, 'pass')
            self.assertEqual(result.expected, expected)

    def test_invalid_enclosures(self):
        with self.assertRaises(ValueError):
            Interval(1, 0)
        with self.assertRaises(ValueError):
            Interval.enclose(1, -1)


class GemmTests(unittest.TestCase):
    def test_boundary_uses_exact_fallback_without_accepting_wrong_code(self):
        case = scalar_case(bias=1, sx=F(1, 2), sb=F(1, 2**26))
        self.assertEqual(exact_gemm(case), ((1,),))
        for actual, expected in ((0, 'fail'), (1, 'pass')):
            self.assertEqual(check_gemm(case, ((actual,),)).verdict, 'inconclusive')
            result = check_gemm(case, ((actual,),), exact_fallback=True)
            self.assertEqual(result.verdict, expected)
            self.assertEqual(result.elements[0][0].method, 'exact')
        refined = check_gemm(case, ((1,),), precision_bits=27)
        self.assertEqual(refined.verdict, 'pass')
        self.assertEqual(refined.elements[0][0].method, 'interval')

    def test_exact_tie_fallback_terminates(self):
        case = scalar_case(x=3, sx=F(1, 6))
        result = check_gemm(case, ((0,),), precision_bits=4, exact_fallback=True)
        self.assertEqual(result.verdict, 'pass')
        self.assertEqual(result.elements[0][0].expected, 0)

    def test_tensor_verdict_preserves_failure_and_uncertainty(self):
        case = GemmCase(((1,),), ((1, 1),), (1, 1), Quantizer(F(1, 2)),
                        (Quantizer(1),)*2, (Quantizer(F(1, 2**26)),)*2, (Quantizer(1),)*2)
        self.assertEqual(check_gemm(case, ((0, 2),)).verdict, 'fail')
        self.assertEqual(check_gemm(case, ((0, 1),)).verdict, 'inconclusive')
        self.assertEqual(check_gemm(case, ((1, 1),), exact_fallback=True).verdict, 'pass')
        self.assertEqual(check_gemm(case, ((0, 1),), exact_fallback=True).verdict, 'fail')

    def test_distinct_channel_parameters_and_axis_mutation(self):
        case = GemmCase(((2, 3),), ((1, 1), (1, 1)), (0, 0), Quantizer(1),
                        (Quantizer(1), Quantizer(2)), (Quantizer(1), Quantizer(1)),
                        (Quantizer(1), Quantizer(1)))
        self.assertEqual(exact_gemm(case), ((5, 10),))
        wrong_axis = exact_gemm(replace(case, w_quant=case.w_quant[::-1]))
        self.assertEqual(check_gemm(case, wrong_axis, exact_fallback=True).verdict, 'fail')

    def test_seeded_enclosures_against_independent_decoded_reference(self):
        rng = random.Random(20260905)
        scales = (F(1, 3), F(1, 4), F(5, 7), F(3), F.from_float(0.1))
        for trial in range(80):
            m, p, n = rng.randint(1, 2), rng.randint(1, 5), rng.randint(1, 3)
            def quant():
                return Quantizer(rng.choice(scales), rng.randint(-2, 2))
            case = GemmCase(
                tuple(tuple(rng.randint(-8, 8) for _ in range(p)) for _ in range(m)),
                tuple(tuple(rng.randint(-8, 8) for _ in range(n)) for _ in range(p)),
                tuple(rng.randint(-20, 20) for _ in range(n)), quant(),
                tuple(quant() for _ in range(n)), tuple(quant() for _ in range(n)),
                tuple(quant() for _ in range(n)))
            expected = exact_gemm(case)
            for bits in (0, 8, 24):
                with self.subTest(trial=trial, bits=bits):
                    intervals = gemm_intervals(case, bits)
                    for i in range(m):
                        for j in range(n):
                            real = sum(case.x_quant.decode(case.x[i][k])
                                       * case.w_quant[j].decode(case.w[k][j]) for k in range(p))
                            real += case.bias_quant[j].decode(case.bias[j])
                            self.assertLessEqual(intervals[i][j].lower, real)
                            self.assertGreaterEqual(intervals[i][j].upper, real)
                    self.assertNotEqual(check_gemm(case, expected, bits).verdict, 'fail')
                    self.assertEqual(check_gemm(case, expected, bits, True).verdict, 'pass')
                    faulty = [list(row) for row in expected]
                    faulty[0][0] += -1 if faulty[0][0] == 127 else 1
                    self.assertEqual(check_gemm(case, faulty, bits, True).verdict, 'fail')

    def test_targeted_faults_are_rejected(self):
        zero_case = scalar_case(x=2, w=2, sx=F(1, 4), zx=5)
        tie_case = scalar_case(x=-3, sx=F(1, 2))
        early_case = scalar_case(x=3, bias=-1, sx=F(1, 2), sb=F(1, 2))
        clipping_case = scalar_case(x=65, w=2)
        faults = (
            ('omit zero point', zero_case, exact_gemm(replace(zero_case, x_quant=Quantizer(F(1, 4))))),
            ('ties toward positive infinity', tie_case, ((math.floor(F(-3, 2) + F(1, 2)),),)),
            ('round before adding bias', early_case, ((round(round(F(3, 2)) - F(1, 2)),),)),
            ('wrap before clipping', clipping_case, (((130 + 128) % 256 - 128,),)),
        )
        for name, case, wrong in faults:
            with self.subTest(fault=name):
                self.assertNotEqual(wrong, exact_gemm(case))
                self.assertEqual(check_gemm(case, wrong, exact_fallback=True).verdict, 'fail')

    def test_reject_shapes_and_illegal_input_codes(self):
        case = scalar_case()
        for changes in ({'x': ()}, {'x': ((1, 2),)}, {'w': ((1, 2),)},
                        {'bias': ()}, {'w_quant': ()}, {'x': ((128,),)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(case, **changes)
        with self.assertRaises(ValueError):
            check_gemm(case, ((1, 2),))
        self.assertEqual(check_gemm(case, ((128,),)).verdict, 'fail')


class AccumulationTests(unittest.TestCase):
    def test_int32_worst_case_boundary(self):
        quant = Quantizer(1, -128)
        self.assertLessEqual(centered_dot_bound(33025, quant, quant), 2**31 - 1)
        self.assertGreater(centered_dot_bound(33026, quant, quant), 2**31 - 1)
        self.assertEqual(checked_centered_dot([127]*33025, [127]*33025, -128, -128), 2147450625)
        with self.assertRaises(OverflowError):
            checked_centered_dot([127]*33026, [127]*33026, -128, -128)

    def test_later_cancellation_cannot_hide_prefix_overflow(self):
        # The mathematical result is 127, but the prefix 127+127 exceeds INT8.
        with self.assertRaises(OverflowError):
            checked_centered_dot([127, 127], [1, 1], bias=-127, bits=8)

    def test_subtraction_product_and_bias_overflow(self):
        for x, w, zx, bias in (([127], [1], -128, 0), ([127], [2], 0, 0),
                               ([127], [1], 0, 1)):
            with self.subTest(x=x, w=w, zx=zx, bias=bias), self.assertRaises(OverflowError):
                checked_centered_dot(x, w, zx=zx, bias=bias, bits=8)
        self.assertEqual(checked_centered_dot([-128], [1], bits=8), -128)

    def test_expanded_formula_needs_separate_bounds(self):
        # Direct centered accumulation is zero; the expanded raw dot exceeds INT16.
        self.assertEqual(checked_centered_dot([127]*3, [127]*3, 127, 127, bits=16), 0)
        with self.assertRaises(OverflowError):
            checked_centered_dot([127]*3, [127]*3, bits=16)


if __name__ == '__main__':
    unittest.main()
