"""Run the same article, Python and Lean checks locally and in CI.

python tools/quantization-proof/verify.py --python-only
python tools/quantization-proof/verify.py --backend

The full check needs the pinned Lean toolchain and Mathlib cache. --backend also
needs g++ and network access to the pinned gemmlowp headers.
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parent.parent
ARTICLE = ROOT / '_posts/2026-09-05-quantization-unit-tests-proof.md'


def run(*args, cwd=ROOT):
    subprocess.run(args, cwd=cwd, check=True)


def check_python():
    run(sys.executable, '-m', 'unittest', 'discover', '-s', str(PROJECT), '-p', 'test_*.py', '-v')
    run(sys.executable, str(PROJECT / 'leimao_gemm_counterexample.py'))
    run(sys.executable, str(PROJECT / 'demo_oracle.py'))
    blocks = re.findall(r'^```python\n(.*?)^```\s*$', ARTICLE.read_text(), re.M | re.S)
    if not blocks:
        raise AssertionError('no article Python examples found')
    for i, block in enumerate(blocks, 1):
        print(f'Checking article Python example {i}', flush=True)
        run(sys.executable, '-c', block)


def audit_axioms(output, declarations):
    reports = dict(re.findall(
        r"'([^']+)' (does not depend on any axioms|depends on axioms:\s*\[[^\]]*\])",
        output))
    allowed = {'propext', 'Classical.choice', 'Quot.sound'}
    for declaration in declarations:
        name = f'QuantizationProof.{declaration}'
        if name not in reports:
            raise AssertionError(f'missing axiom audit for {name}')
        report = reports[name]
        if report.startswith('depends'):
            dependencies = {name.strip() for name in report.split('[', 1)[1][:-1].split(',')}
            if dependencies - allowed:
                raise AssertionError(f'unexpected axioms for {name}: {dependencies - allowed}')


def check_lean():
    manifest = PROJECT / 'lake-manifest.json'
    locked = manifest.read_bytes()
    run('lake', 'build', cwd=PROJECT)
    result = subprocess.run(['lake', 'env', 'lean', '-DwarningAsError=true',
                             'QuantizationProof.lean'], cwd=PROJECT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end='', flush=True)
    result.check_returncode()
    if manifest.read_bytes() != locked:
        raise AssertionError('verification changed the dependency lockfile')
    source = (PROJECT / 'QuantizationProof.lean').read_text()
    declarations = re.findall(r'^#print axioms (\S+)$', source, re.M)
    if not declarations:
        raise AssertionError('no theorem axiom audits found')
    audit_axioms(result.stdout, declarations)
    print(f'Lean checked; {len(declarations)} axiom reports accepted.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python-only', action='store_true', help='skip Lean verification')
    parser.add_argument('--backend', action='store_true', help='also compare pinned upstream C++')
    args = parser.parse_args()
    if not __debug__:
        parser.error('verification must run without Python -O or PYTHONOPTIMIZE')
    check_python()
    if not args.python_only:
        check_lean()
    if args.backend:
        run(sys.executable, str(PROJECT / 'check_gemmlowp.py'))


if __name__ == '__main__':
    main()
