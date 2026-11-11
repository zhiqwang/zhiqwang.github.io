"""Reproduce two zero-point-zero GEMM boundary cases.

Source of the computation paths:
https://leimao.github.io/article/Neural-Networks-Quantization/

This reduced reproduction was checked against the functions posted in that
article, using Python 3.10 and NumPy 1.26.4. Every zero point is zero, so the
zero-point correction terms vanish. It does not download or execute web code.

Run from the repository root:
    uv run --no-project --python 3.10 --with numpy==1.26.4 \
        python tools/quantization-proof/leimao_gemm_counterexample.py
"""

from fractions import Fraction

import numpy as np

from oracle import GemmCase, Quantizer, exact_gemm, check_gemm


def decode_fp32(codes, scale):
    return (codes.astype(np.int32) * scale).astype(np.float32)


def output_codes(values, scale):
    scaled = values * (1 / scale)
    return np.clip(np.rint(scaled), -128, 127).astype(np.int8)


def reproduce(name, x_code, w_code, b_code, sx, sw, sb, sy):
    x = decode_fp32(x_code, sx)
    w = decode_fp32(w_code, sw)
    b = decode_fp32(b_code, sb)
    fp_result = np.matmul(x, w) + b
    fp_code = output_codes(fp_result, sy)

    accumulator = np.matmul(x_code.astype(np.int32), w_code.astype(np.int32))
    scaled_bias = b_code.astype(np.int32) * (sb / sy)
    scaled_result = scaled_bias + accumulator * (sx * sw / sy)
    integer_path_code = output_codes(scaled_result, 1.0)

    case = GemmCase(x_code.tolist(), w_code.tolist(), b_code[0].tolist(),
                    Quantizer(sx), (Quantizer(sw),),
                    (Quantizer(sb, qmin=-2**31, qmax=2**31-1),), (Quantizer(sy),))
    exact_result = sum(Fraction(int(a)) * Fraction.from_float(sx)
                       * Fraction(int(b)) * Fraction.from_float(sw)
                       for a, b in zip(x_code[0], w_code[:, 0]))
    exact_result += Fraction(int(b_code.item())) * Fraction.from_float(sb)
    exact_code = exact_gemm(case)[0][0]
    codes_equal = np.array_equal(integer_path_code, fp_code)
    decoded_equal = np.array_equal(
        decode_fp32(integer_path_code, sy), decode_fp32(fp_code, sy)
    )

    assert fp_result.dtype == np.float32
    assert scaled_result.dtype == np.float64
    assert exact_code == integer_path_code.item() == 1
    assert fp_code.item() == 0
    assert not codes_equal
    assert not decoded_equal
    assert check_gemm(case, integer_path_code.tolist(), exact_fallback=True).verdict == "pass"
    assert check_gemm(case, fp_code.tolist(), exact_fallback=True).verdict == "fail"

    print(f"Case: {name}; bias in accumulator units: {sb == sx * sw}")
    print(f"Exact result: {exact_result}")
    print(f"Exact code: {exact_code}")
    print(f"Integer path before rounding: {scaled_result.item()!r} ({scaled_result.dtype})")
    print(f"FP32 reference before rounding: {fp_result.item()!r} ({fp_result.dtype})")
    print(f"Integer path code: {integer_path_code.tolist()}")
    print(f"FP32 reference code: {fp_code.tolist()}")
    print(f"Integer outputs equal: {codes_equal}")
    print(f"Decoded outputs equal: {decoded_equal}")


def main():
    print(f"NumPy: {np.__version__}")
    reproduce("independent bias scale", np.array([[1]], dtype=np.int8),
              np.array([[1]], dtype=np.int8), np.array([[1]], dtype=np.int8),
              0.5, 1.0, 2.0**-26, 1.0)
    reproduce("bias scale = input scale * weight scale",
              np.full((1, 2048), -128, dtype=np.int8),
              np.full((2048, 1), -64, dtype=np.int8), np.array([[1]], dtype=np.int32),
              2.0**-25, 1.0, 2.0**-25, 1.0)


if __name__ == "__main__":
    main()
