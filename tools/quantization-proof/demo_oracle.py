"""Run from the repository root: python tools/quantization-proof/demo_oracle.py."""

from fractions import Fraction as F

from oracle import GemmCase, Quantizer, check_gemm


def main():
    case = GemmCase(((1,),), ((1,),), (1,), Quantizer(F(1, 2)),
                    (Quantizer(1),), (Quantizer(F(1, 2**26)),), (Quantizer(1),))
    for label, actual, bits, fallback in (
        ('coarse enclosure, candidate 0', ((0,),), 24, False),
        ('coarse enclosure, candidate 2', ((2,),), 24, False),
        ('refined enclosure, candidate 1', ((1,),), 27, False),
        ('exact fallback, candidate 0', ((0,),), 24, True),
        ('exact fallback, candidate 1', ((1,),), 24, True),
    ):
        result = check_gemm(case, actual, bits, fallback)
        element = result.elements[0][0]
        print(f'{label}: {result.verdict}; bounds={element.code_bounds}; '
              f'expected={element.expected}; method={element.method}')


if __name__ == '__main__':
    main()
