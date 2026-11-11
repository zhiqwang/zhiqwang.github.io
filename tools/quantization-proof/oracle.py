"""Exact and interval oracles for the article's affine GEMM contract.

Only the standard library is needed. Arithmetic uses unbounded integers and
Fractions; this certifies the ideal specification, not a floating-point backend.
Weights, biases and outputs may have distinct parameters per output column.
"""

from dataclasses import dataclass
from fractions import Fraction
import math
import operator
from typing import Literal


def integer(value):
    """Accept integer codes, without silently truncating floats or booleans."""
    if isinstance(value, bool):
        raise TypeError("a code must be an integer, not a boolean")
    return int(operator.index(value))


def exact_value(value):
    """Preserve a finite Python float's stored value, not its printed decimal.

    Widen FP32 metadata with float(np.float32_value) before calling this function.
    Use Fraction explicitly when the specification instead means a rational.
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("the numerical contract requires finite values")
        return Fraction.from_float(value)
    return Fraction(integer(value))


@dataclass(frozen=True)
class Quantizer:
    scale: Fraction
    zero_point: int = 0
    qmin: int = -128
    qmax: int = 127

    def __post_init__(self):
        object.__setattr__(self, "scale", exact_value(self.scale))
        for name in ("zero_point", "qmin", "qmax"):
            object.__setattr__(self, name, integer(getattr(self, name)))
        if self.scale <= 0 or self.qmin > self.qmax:
            raise ValueError("scale must be positive and the code range nonempty")

    def decode(self, code):
        code = integer(code)
        if not self.qmin <= code <= self.qmax:
            raise ValueError("input code outside its quantizer's legal range")
        return self.scale * (code - self.zero_point)

    def encode(self, value):
        # Fraction.__round__ implements nearest integer, ties to even.
        code = round(exact_value(value) / self.scale + self.zero_point)
        return min(self.qmax, max(self.qmin, code))


@dataclass(frozen=True)
class Interval:
    lower: Fraction
    upper: Fraction

    def __post_init__(self):
        object.__setattr__(self, "lower", exact_value(self.lower))
        object.__setattr__(self, "upper", exact_value(self.upper))
        if self.lower > self.upper:
            raise ValueError("reversed interval")

    @classmethod
    def enclose(cls, value, precision_bits):
        """Round outward onto multiples of 2**(-precision_bits), exactly."""
        precision_bits = integer(precision_bits)
        if precision_bits < 0:
            raise ValueError("precision_bits must be nonnegative")
        denominator = 1 << precision_bits
        scaled = exact_value(value) * denominator
        return cls(Fraction(math.floor(scaled), denominator),
                   Fraction(math.ceil(scaled), denominator))

    def times_integer(self, value):
        value = integer(value)
        endpoints = (self.lower * value, self.upper * value)
        return Interval(min(endpoints), max(endpoints))

    def __add__(self, other):
        return Interval(self.lower + other.lower, self.upper + other.upper)


@dataclass(frozen=True)
class GemmCase:
    x: tuple[tuple[int, ...], ...]
    w: tuple[tuple[int, ...], ...]
    bias: tuple[int, ...]
    x_quant: Quantizer
    w_quant: tuple[Quantizer, ...]
    bias_quant: tuple[Quantizer, ...]
    y_quant: tuple[Quantizer, ...]

    def __post_init__(self):
        for name in ("x", "w"):
            object.__setattr__(self, name, tuple(
                tuple(integer(q) for q in row) for row in getattr(self, name)))
        object.__setattr__(self, "bias", tuple(integer(q) for q in self.bias))
        for name in ("w_quant", "bias_quant", "y_quant"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        n = len(self.y_quant)
        if not self.x or not self.w or not n:
            raise ValueError("this example requires nonempty matrices")
        if any(len(row) != len(self.w) for row in self.x):
            raise ValueError("X must have shape m by p")
        if any(len(row) != n for row in self.w):
            raise ValueError("W must have shape p by n")
        if any(len(values) != n for values in
               (self.bias, self.w_quant, self.bias_quant)):
            raise ValueError("bias and channel parameters must have length n")
        for row in self.x:
            for code in row:
                self.x_quant.decode(code)
        for row in self.w:
            for j, code in enumerate(row):
                self.w_quant[j].decode(code)
        for j, code in enumerate(self.bias):
            self.bias_quant[j].decode(code)


def exact_gemm(case):
    """Independent DQ -> products -> sum+bias -> Q path, using Fractions."""
    return tuple(tuple(case.y_quant[j].encode(
        sum((case.x_quant.decode(code) * case.w_quant[j].decode(case.w[k][j])
             for k, code in enumerate(row)), Fraction(0))
        + case.bias_quant[j].decode(case.bias[j]))
        for j in range(len(case.y_quant))) for row in case.x)


def gemm_intervals(case, precision_bits=24):
    """Enclose decoded outputs independently of the candidate implementation.

    Centered dot products are exact Python integers. Enclose sx*sw and sb on a
    dyadic grid, then multiply intervals by the integer dot/bias and add them.
    The construction also works for negative dots, cancellation and saturation.
    Endpoint output quantization is performed exactly by Quantizer.encode.
    """
    products = tuple(Interval.enclose(case.x_quant.scale * q.scale, precision_bits)
                     for q in case.w_quant)
    biases = tuple(Interval.enclose(q.scale, precision_bits) for q in case.bias_quant)
    results = []
    for row in case.x:
        output = []
        for j in range(len(case.y_quant)):
            dot = sum((code - case.x_quant.zero_point)
                      * (case.w[k][j] - case.w_quant[j].zero_point)
                      for k, code in enumerate(row))
            bias = case.bias[j] - case.bias_quant[j].zero_point
            output.append(products[j].times_integer(dot) + biases[j].times_integer(bias))
        results.append(tuple(output))
    return tuple(results)


Verdict = Literal["pass", "fail", "inconclusive"]


@dataclass(frozen=True)
class ElementCheck:
    verdict: Verdict
    actual: int
    code_bounds: tuple[int, int]
    expected: int | None
    method: Literal["interval", "exact"] = "interval"


def check_interval(interval, quantizer, actual):
    actual = integer(actual)
    lower, upper = quantizer.encode(interval.lower), quantizer.encode(interval.upper)
    expected = lower if lower == upper else None
    if actual < lower or actual > upper:
        verdict = "fail"
    elif expected is not None:
        verdict = "pass"
    else:
        verdict = "inconclusive"
    return ElementCheck(verdict, actual, (lower, upper), expected)


@dataclass(frozen=True)
class GemmCheck:
    verdict: Verdict
    elements: tuple[tuple[ElementCheck, ...], ...]
    intervals: tuple[tuple[Interval, ...], ...]


def check_gemm(case, actual, precision_bits=24, exact_fallback=False):
    actual = tuple(tuple(integer(q) for q in row) for row in actual)
    if len(actual) != len(case.x) or any(len(row) != len(case.y_quant) for row in actual):
        raise ValueError("candidate output shape must be m by n")
    intervals = gemm_intervals(case, precision_bits)
    exact = None
    results = []
    for i, row in enumerate(intervals):
        checks = []
        for j, interval in enumerate(row):
            result = check_interval(interval, case.y_quant[j], actual[i][j])
            if result.verdict == "inconclusive" and exact_fallback:
                if exact is None:
                    exact = exact_gemm(case)
                expected = exact[i][j]
                result = ElementCheck("pass" if result.actual == expected else "fail",
                                      result.actual, result.code_bounds, expected, "exact")
            checks.append(result)
        results.append(tuple(checks))
    verdicts = {result.verdict for row in results for result in row}
    verdict = "fail" if "fail" in verdicts else (
        "inconclusive" if "inconclusive" in verdicts else "pass")
    return GemmCheck(verdict, tuple(results), intervals)


def centered_dot_bound(length, x_quant, w_quant, bias=0):
    """Absolute bound on every direct accumulation prefix, including bias.

    bias must already be in accumulator units. This is NOT a bound on the
    separate intermediate terms of the expanded zero-point-correction formula.
    """
    length, bias = integer(length), integer(bias)
    if length < 0:
        raise ValueError("length must be nonnegative")
    mx = max(abs(x_quant.qmin - x_quant.zero_point), abs(x_quant.qmax - x_quant.zero_point))
    mw = max(abs(w_quant.qmin - w_quant.zero_point), abs(w_quant.qmax - w_quant.zero_point))
    return length * mx * mw + abs(bias)


def checked_centered_dot(x, w, zx=0, zw=0, bias=0, bits=32):
    """Scalar signed-integer model: widen, subtract, multiply, sum, then bias.

    Reject overflow at each step, even if later cancellation would hide it.
    This is a checked model, not an emulator of hardware wraparound/saturation.
    """
    bits = integer(bits)
    if bits < 2:
        raise ValueError("signed arithmetic requires at least two bits")
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1

    def checked(value):
        value = integer(value)
        if not lo <= value <= hi:
            raise OverflowError(f"{value} is outside signed {bits}-bit range")
        return value

    x, w = tuple(x), tuple(w)
    if len(x) != len(w):
        raise ValueError("dot-product lengths differ")
    zx, zw, bias = checked(zx), checked(zw), checked(bias)
    acc = 0
    for a, b in zip(x, w):
        a, b = checked(checked(a) - zx), checked(checked(b) - zw)
        acc = checked(acc + checked(a * b))
    return checked(acc + bias)
