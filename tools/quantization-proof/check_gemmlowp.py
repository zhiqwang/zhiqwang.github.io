"""Compare the Python arithmetic model with pinned upstream C++ primitives.

Needs network access and g++. Downloads headers into a temporary directory;
neither the checkout nor the machine's installed packages are changed.
"""

from pathlib import Path
import random
import subprocess
import tempfile
from urllib.request import urlopen

from gemmlowp_model import high_mul, rounding_right_shift, requantize

REVISION = '16e8662c34917be0065110bfcd9cc27d30f52fdf'


def main():
    rng = random.Random(20260905)
    vectors = []
    values = (-(2**31), -(2**31) + 1, -3, -2, -1, 0, 1, 2, 3, 2**31-1)
    for a in values:
        for b in (*values, 2**30):
            vectors.append((f'h {a} {b}', high_mul(a, b)))
        for shift in range(32):
            vectors.append((f'd {a} {shift}', rounding_right_shift(a, shift)))
        for multiplier in (0, 1, 2**30, 2**31-1):
            for shift in (0, 1, 7, 31):
                vectors.append((f'q {a} {multiplier} {shift} 0', requantize(a, multiplier, shift)))
    for _ in range(1000):
        a, b = rng.randrange(-(2**31), 2**31), rng.randrange(-(2**31), 2**31)
        vectors.append((f'h {a} {b}', high_mul(a, b)))
        a, multiplier, shift, zero = (rng.randrange(-10000, 10001),
                                     rng.randrange(2**31), rng.randrange(32),
                                     rng.randrange(-128, 128))
        vectors.append((f'q {a} {multiplier} {shift} {zero}',
                        requantize(a, multiplier, shift, zero)))
    with tempfile.TemporaryDirectory(prefix='quantization-gemmlowp-') as directory:
        root = Path(directory)
        for name in ('fixedpoint/fixedpoint.h', 'internal/detect_platform.h'):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(f'https://raw.githubusercontent.com/google/gemmlowp/{REVISION}/{name}',
                         timeout=30) as response:
                target.write_bytes(response.read())
        executable = root / 'probe'
        subprocess.run(['g++', '-std=c++11', '-O2',
                        '-I', str(root), str(Path(__file__).with_name('gemmlowp_probe.cpp')),
                        '-o', str(executable)], check=True)
        result = subprocess.run([str(executable)], check=True, text=True, capture_output=True,
                                input=''.join(f'{line}\n' for line, _ in vectors))
        actual = [int(line) for line in result.stdout.splitlines()]
        if len(actual) != len(vectors):
            raise AssertionError('upstream probe returned the wrong number of outputs')
        for (vector, expected), observed in zip(vectors, actual):
            if expected != observed:
                raise AssertionError(f'{vector}: Python {expected}, upstream C++ {observed}')
    print(f'gemmlowp {REVISION}: {len(vectors)} scalar arithmetic comparisons passed.')


if __name__ == '__main__':
    main()
