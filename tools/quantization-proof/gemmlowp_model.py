"""A specified scalar requantization path, separate from the ideal oracle.

Models gemmlowp's INT32 high multiply followed by a rounding right shift,
then zero-point addition and INT8 saturation. Multiplier selection and left
shifts are outside this small contract. See check_gemmlowp.py for a comparison
with the pinned upstream C++ primitives, rather than another Python copy.
"""

from oracle import checked_centered_dot, integer

INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


def int32(value):
    value = integer(value)
    if not INT32_MIN <= value <= INT32_MAX:
        raise OverflowError('value outside INT32')
    return value


def high_mul(a, b):
    """Nearest high multiply; halfway cases go toward positive infinity."""
    a, b = int32(a), int32(b)
    return min(INT32_MAX, (a * b + 2**30) // 2**31)


def rounding_right_shift(value, shift):
    """Nearest division by 2**shift; halfway cases go away from zero."""
    value, shift = int32(value), integer(shift)
    if not 0 <= shift <= 31:
        raise ValueError('right shift must be between 0 and 31')
    if shift == 0:
        return value
    magnitude = (abs(value) + 2**(shift - 1)) // 2**shift
    return -magnitude if value < 0 else magnitude


def requantize(acc, multiplier, right_shift, zero_point=0):
    multiplier, zero_point = int32(multiplier), integer(zero_point)
    if multiplier < 0 or not -128 <= zero_point <= 127:
        raise ValueError('requires nonnegative multiplier and an INT8 zero point')
    shifted = rounding_right_shift(high_mul(acc, multiplier), right_shift)
    offset = int32(shifted + zero_point)
    return min(127, max(-128, offset))


def dot_output(x, w, multiplier, right_shift, zx=0, zw=0, bias=0, zy=0):
    """Bias is in accumulator units; every accumulation step must fit INT32."""
    return requantize(checked_centered_dot(x, w, zx, zw, bias), multiplier, right_shift, zy)
