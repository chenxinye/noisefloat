"""NFloat-style analysis of the Rump'83 polynomial example.

This example mirrors the C polynomial snippet:

    P(x, y) = 9*x^4 - y^4 + 2*y^2

with ``double`` / ``double_st`` inputs.  The first input pair,
``x = 10864`` and ``y = 18817``, is a classic cancellation example: the exact
integer result is 1, while ordinary IEEE double evaluation of the expression
returns 2 on typical platforms.  The NFloat-style stochastic run reports this
case as ``@.0`` because the cancellation leaves no reliable significant digit
in the sample ensemble.

By default, the script prints only the noisefloat/NFloat-style result for each
input pair.  Use ``--verbose`` to inspect the intermediate double, exact, and
sample-level diagnostics.
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

import numpy as np

from noisefloat import NFloat, clear_diagnostics, configure, print_diagnostics_summary


DEFAULT_DIGITS_THRESHOLD = 6.0
DOUBLE_EXP_BITS = 11
DOUBLE_SIG_BITS = 52
MAX_DOUBLE_DIGITS = 15.0


@dataclass(frozen=True)
class RumpInput:
    label: str
    x: Fraction
    y: Fraction


@dataclass(frozen=True)
class RumpReport:
    label: str
    x: Fraction
    y: Fraction
    double_result: float
    exact_result: Fraction
    nfloat_mean: float
    nfloat_std: float
    nfloat_digits: float
    nfloat_samples: tuple[float, ...]
    reference_digits: float
    sample_based_stable: bool
    reference_based_stable: bool


DEFAULT_INPUTS = (
    RumpInput("P1", Fraction(10864, 1), Fraction(18817, 1)),
    RumpInput("P2", Fraction(1, 3), Fraction(2, 3)),
)


def rump_float64(x: float, y: float) -> float:
    """Evaluate the polynomial with ordinary Python/NumPy float64 arithmetic."""

    return float(np.float64(9.0) * x * x * x * x - y * y * y * y + 2.0 * y * y)


def rump_exact(x: Fraction, y: Fraction) -> Fraction:
    """Evaluate the polynomial exactly for integer or rational inputs."""

    return Fraction(9, 1) * x * x * x * x - y * y * y * y + Fraction(2, 1) * y * y


def rump_nfloat(x: float, y: float) -> NFloat:
    """Evaluate the polynomial with noisefloat NFloat samples."""

    x_st = NFloat(x)
    y_st = NFloat(y)
    return 9.0 * x_st * x_st * x_st * x_st - y_st * y_st * y_st * y_st + 2.0 * y_st * y_st


def significant_digits_against_reference(
    approximate: float, exact: Fraction, max_digits: float = MAX_DOUBLE_DIGITS
) -> float:
    """Return decimal significant digits of ``approximate`` against ``exact``."""

    approx = Fraction.from_float(float(approximate))
    if exact == 0:
        if approx == 0:
            return max_digits
        return 0.0

    error = abs(approx - exact)
    if error == 0:
        return max_digits

    relative_error = error / abs(exact)
    digits = -math.log10(float(relative_error))
    return max(0.0, min(max_digits, digits))


def configure_like_nfloat_c_double(
    *, random_state: int, digits_threshold: float
) -> None:
    """Configure noisefloat for double-like stochastic precision."""

    configure(
        backend="numpy",
        exp_bits=DOUBLE_EXP_BITS,
        sig_bits=DOUBLE_SIG_BITS,
        n_samples=3,
        random_state=random_state,
        digits_threshold=digits_threshold,
        zero_digits_threshold=digits_threshold,
        trace=False,
    )


def analyze_rump83(
    *,
    inputs: Iterable[RumpInput] = DEFAULT_INPUTS,
    digits_threshold: float = DEFAULT_DIGITS_THRESHOLD,
    random_state: int = 42,
) -> tuple[RumpReport, ...]:
    """Analyze the Rump'83 polynomial with noisefloat and exact references."""

    clear_diagnostics()
    configure_like_nfloat_c_double(
        random_state=random_state, digits_threshold=digits_threshold
    )
    reports = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for item in inputs:
            double_result = rump_float64(float(item.x), float(item.y))
            exact_result = rump_exact(item.x, item.y)
            nfloat_result = rump_nfloat(float(item.x), float(item.y))
            nfloat_mean = float(np.asarray(nfloat_result.mean).mean())
            nfloat_std = float(np.asarray(nfloat_result.std).mean())
            nfloat_digits = float(np.asarray(nfloat_result.digits).mean())
            reference_digits = significant_digits_against_reference(
                nfloat_mean, exact_result
            )
            reports.append(
                RumpReport(
                    label=item.label,
                    x=item.x,
                    y=item.y,
                    double_result=double_result,
                    exact_result=exact_result,
                    nfloat_mean=nfloat_mean,
                    nfloat_std=nfloat_std,
                    nfloat_digits=nfloat_digits,
                    nfloat_samples=tuple(float(value) for value in nfloat_result.samples),
                    reference_digits=reference_digits,
                    sample_based_stable=bool(nfloat_digits >= digits_threshold),
                    reference_based_stable=bool(reference_digits >= digits_threshold),
                )
            )
    return tuple(reports)


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return f"{float(value):.15e}"
    return f"{float(value):.15e} ({value.numerator}/{value.denominator})"


def _nfloat_style_value(report: RumpReport) -> str:
    if report.nfloat_digits < DEFAULT_DIGITS_THRESHOLD:
        return "@.0  (no correct digits)"
    return f"{report.nfloat_mean:.15e}"


def print_compact_report(reports: Iterable[RumpReport]) -> None:
    print("Noisefloat results")
    print("-" * 72)
    for report in reports:
        stability = "stable" if report.sample_based_stable else "unstable"
        print(
            f"{report.label}= {_nfloat_style_value(report)}"
            f"    significant_digits={report.nfloat_digits:.3f}"
            f"    std={report.nfloat_std:.3e}"
            f"    stability={stability}"
        )

    print()
    print_diagnostics_summary()
    print()


def print_verbose_report(report: RumpReport, threshold: float) -> None:
    sample_stability = "stable" if report.sample_based_stable else "unstable"
    reference_stability = "stable" if report.reference_based_stable else "unstable"
    sample_text = ", ".join(f"{value:.15e}" for value in report.nfloat_samples)

    print(
        f"{report.label}: x={_format_fraction(report.x)}, "
        f"y={_format_fraction(report.y)}"
    )
    print("-" * 72)
    print(f"C++ double result             : {report.double_result:.15e}")
    print(f"Exact rational result         : {_format_fraction(report.exact_result)}")
    print(f"Noisefloat NFloat samples      : [{sample_text}]")
    print(f"Noisefloat representative     : {report.nfloat_mean:.15e}")
    print(f"Noisefloat sample std         : {report.nfloat_std:.6e}")
    print(f"Sample-based digits           : {report.nfloat_digits:.3f}")
    print(f"Reference-aware digits        : {report.reference_digits:.3f}")
    print(f"Digits threshold              : {threshold:.3f}")
    print(f"Sample-based stability        : {sample_stability}")
    print(f"Reference-aware stability     : {reference_stability}")
    print(f"NFloat-style printed result    : {_nfloat_style_value(report)}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits-threshold", type=float, default=DEFAULT_DIGITS_THRESHOLD)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print detailed double/exact/sample diagnostics",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    reports = analyze_rump83(
        digits_threshold=args.digits_threshold,
        random_state=args.random_state,
    )

    print("Rump'83 polynomial NFloat-style analysis")
    print("=" * 72)
    print(
        "Configuration: backend=numpy, exp_bits=11, sig_bits=52, "
        "n_samples=3 (double-like stochastic precision)"
    )
    print()
    print_compact_report(reports)

    if args.verbose:
        print("Verbose diagnostics")
        print("=" * 72)
        for report in reports:
            print_verbose_report(report, args.digits_threshold)


if __name__ == "__main__":
    main()
