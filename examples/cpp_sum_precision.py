"""NFloat-style analysis of two C/C++ float summation examples.

The examples mirror the two snippets in which ``addend = 2^-24`` is repeatedly
added to an initial single-precision value of 1.

Code 1 keeps the running sum in ``float``. Around 1.0, ``2^-24`` is half of one
float32 ulp, so round-to-nearest float32 arithmetic can repeatedly lose the
increment.

Code 2 promotes the running sum to ``double`` and casts back to ``float`` only
once at the end. This is expected to be much more stable.

Running the full 100,000,000-loop NFloat scalar simulation in Python would be
slow, so this example reports the deterministic C-style full-loop results and
uses a configurable NFloat probe loop to expose the numerical stability of the
two accumulation patterns.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass

import numpy as np

from noisefloat import NFloat, configure


DEFAULT_CPP_ITERATIONS = 100_000_000
DEFAULT_NFloat_ITERATIONS = 20_000


@dataclass(frozen=True)
class SumReport:
    name: str
    deterministic_float32: np.float32
    exact_real_value: float
    nfloat_mean: float
    nfloat_std: float
    nfloat_digits: float
    is_numerically_stable: bool
    nfloat_iterations: int


def cpp_code1_float_sum(iterations: int) -> np.float32:
    """Return the round-to-nearest float32 result of Code 1.

    Because the addend is exactly half an ulp at 1.0 and float32 uses
    round-to-nearest-even on ordinary platforms, the first addition rounds back
    to 1.0. The state therefore remains 1.0 for the whole loop.
    """

    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    return np.float32(1.0)


def cpp_code2_double_sum_then_float(iterations: int) -> np.float32:
    """Return the C-style result of Code 2 using double accumulation."""

    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    sum_value = np.float32(1.0)
    addend = np.float32(2.0**-24)
    temp_sum = np.float64(sum_value)
    temp_sum = temp_sum + np.float64(addend) * np.float64(iterations - 1)
    return np.float32(temp_sum)


def exact_real_sum(iterations: int) -> float:
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    return 1.0 + float(iterations - 1) * 2.0**-24


def _nfloat_scalar_stats(
    value: NFloat, threshold: float
) -> tuple[float, float, float, bool]:
    digits = float(np.asarray(value.digits).mean())
    return (
        float(np.asarray(value.mean).mean()),
        float(np.asarray(value.std).mean()),
        digits,
        bool(digits >= threshold),
    )


def nfloat_code1_float_probe(iterations: int) -> NFloat:
    """NFloat probe for Code 1: stochastic single-precision-style loop."""

    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    total = NFloat(1.0)
    addend = NFloat(2.0**-24)
    for _ in range(iterations - 1):
        total = total + addend
    return total


def nfloat_code2_double_probe(iterations: int) -> NFloat:
    """NFloat probe for Code 2: accumulate once in the promoted expression."""

    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    return NFloat(1.0) + NFloat(2.0**-24) * NFloat(float(iterations - 1))


def analyze_sums(
    *,
    cpp_iterations: int = DEFAULT_CPP_ITERATIONS,
    nfloat_iterations: int = DEFAULT_NFloat_ITERATIONS,
    digits_threshold: float = 6.0,
    random_state: int = 42,
) -> tuple[SumReport, SumReport]:
    """Analyze the two C/C++ summation patterns with noisefloat."""

    configure(
        backend="numpy",
        exp_bits=8,
        sig_bits=23,
        n_samples=3,
        random_state=random_state,
        digits_threshold=digits_threshold,
        zero_digits_threshold=digits_threshold,
        trace=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code1_nfloat = nfloat_code1_float_probe(nfloat_iterations)
    code1_mean, code1_std, code1_digits, code1_stable = _nfloat_scalar_stats(
        code1_nfloat, digits_threshold
    )

    configure(
        backend="numpy",
        exp_bits=8,
        sig_bits=23,
        n_samples=3,
        random_state=random_state,
        digits_threshold=digits_threshold,
        zero_digits_threshold=digits_threshold,
        trace=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code2_nfloat = nfloat_code2_double_probe(cpp_iterations)
    code2_mean, code2_std, code2_digits, code2_stable = _nfloat_scalar_stats(
        code2_nfloat, digits_threshold
    )

    return (
        SumReport(
            name="Code 1: float running sum",
            deterministic_float32=cpp_code1_float_sum(cpp_iterations),
            exact_real_value=exact_real_sum(cpp_iterations),
            nfloat_mean=code1_mean,
            nfloat_std=code1_std,
            nfloat_digits=code1_digits,
            is_numerically_stable=code1_stable,
            nfloat_iterations=nfloat_iterations,
        ),
        SumReport(
            name="Code 2: double temp_sum then float cast",
            deterministic_float32=cpp_code2_double_sum_then_float(cpp_iterations),
            exact_real_value=exact_real_sum(cpp_iterations),
            nfloat_mean=code2_mean,
            nfloat_std=code2_std,
            nfloat_digits=code2_digits,
            is_numerically_stable=code2_stable,
            nfloat_iterations=cpp_iterations,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-iterations", type=int, default=DEFAULT_CPP_ITERATIONS)
    parser.add_argument(
        "--nfloat-iterations", type=int, default=DEFAULT_NFloat_ITERATIONS
    )
    parser.add_argument("--digits-threshold", type=float, default=6.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def print_report(report: SumReport, threshold: float) -> None:
    stable = "stable" if report.is_numerically_stable else "unstable"
    print(report.name)
    print("-" * len(report.name))
    print(f"C-style float32 result : {float(report.deterministic_float32):.30f}")
    print(f"Exact real sum         : {report.exact_real_value:.30f}")
    print(f"NFloat probe iterations: {report.nfloat_iterations}")
    print(f"NFloat representative  : {report.nfloat_mean:.30f}")
    print(f"NFloat std             : {report.nfloat_std:.6e}")
    print(f"Significant digits    : {report.nfloat_digits:.3f}")
    print(f"Threshold             : {threshold:.3f}")
    print(f"Numerical stability   : {stable}")
    print()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    reports = analyze_sums(
        cpp_iterations=args.cpp_iterations,
        nfloat_iterations=args.nfloat_iterations,
        digits_threshold=args.digits_threshold,
        random_state=args.random_state,
    )
    print("NFloat analysis for two C/C++ summation patterns")
    print("=" * 56)
    print(
        "The C-style results use the full --cpp-iterations count. "
        "The Code 1 NFloat probe uses --nfloat-iterations to keep the Python "
        "scalar loop practical."
    )
    print()
    for report in reports:
        print_report(report, args.digits_threshold)


if __name__ == "__main__":
    main()
