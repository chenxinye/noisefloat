"""Term-by-term low-precision analysis of ``sum_{i=1}^n i^3``.

This example mirrors the classical summation test

    S = sum_{i=1}^n i^3 = (n(n+1)/2)^2.

For each requested ``n`` and floating-point format it performs two sequential
accumulations:

* deterministic round-to-nearest after every multiplication and addition;
* noisefloat-style stochastic arithmetic with three samples and random
  rounding after every multiplication and addition.

The tested formats are FP32, FP16, BF16, and E5M2.  The exact closed form is
used as the reference so each row reports both deterministic reference digits
and stochastic CESTAC/sample digits.
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from noisefloat import configure, reset_chopper_cache
from noisefloat.core import _chopper


DEFAULT_N_VALUES = (10_000, 100_000, 1_000_000, 10_000_000)
SAMPLES = 3
CONFIDENCE = 0.95
TAU_DF2_95 = 4.303
MAX_REFERENCE_DIGITS = 15.0


@dataclass(frozen=True)
class FormatSpec:
    name: str
    exp_bits: int
    sig_bits: int


FORMATS = {
    "fp32": FormatSpec("fp32", exp_bits=8, sig_bits=23),
    "fp16": FormatSpec("fp16", exp_bits=5, sig_bits=10),
    "bf16": FormatSpec("bf16", exp_bits=8, sig_bits=7),
    "e5m2": FormatSpec("e5m2", exp_bits=5, sig_bits=2),
}


@dataclass(frozen=True)
class SumCubesReport:
    n: int
    format_name: str
    exp_bits: int
    sig_bits: int
    exact_sum: int
    deterministic_sum: float
    deterministic_abs_error: float
    deterministic_relative_error: float
    deterministic_reference_digits: float
    stochastic_representative: float
    stochastic_std: float
    stochastic_sample_digits: float
    stochastic_reference_digits: float
    stochastic_abs_error: float
    stochastic_relative_error: float
    stochastic_samples: tuple[float, ...]
    engine: str


def exact_sum_cubes(n: int) -> int:
    """Return the exact integer value of ``sum_{i=1}^n i^3``."""

    if n < 1:
        raise ValueError("n must be positive")
    triangular = n * (n + 1) // 2
    return triangular * triangular


def reference_digits(
    approximate: float,
    exact: int,
    max_digits: float = MAX_REFERENCE_DIGITS,
) -> float:
    """Return decimal significant digits against the exact integer reference."""

    if not math.isfinite(float(approximate)):
        return 0.0
    exact_float = float(exact)
    relative_error = abs(float(approximate) - exact_float) / abs(exact_float)
    if relative_error == 0.0:
        return max_digits
    return max(0.0, min(max_digits, -math.log10(relative_error)))


def sample_digits(samples: Sequence[float]) -> tuple[float, float, float]:
    """Return ``(mean, std, CESTAC digits)`` for three stochastic samples."""

    values = np.asarray(samples, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    if not np.all(np.isfinite(values)):
        return mean, std, 0.0
    if std == 0.0:
        return mean, std, MAX_REFERENCE_DIGITS
    if mean == 0.0:
        return mean, std, 0.0
    ratio = TAU_DF2_95 * std / (math.sqrt(values.size) * abs(mean))
    digits = max(0.0, min(MAX_REFERENCE_DIGITS, -math.log10(ratio)))
    return mean, std, digits


def format_limits(exp_bits: int, sig_bits: int) -> tuple[float, float, float]:
    max_exp = float(2 ** (exp_bits - 1) - 1)
    min_exp = -max_exp + 1.0
    max_finite = (2.0 - 2.0 ** (-sig_bits)) * (2.0**max_exp)
    return min_exp, max_exp, max_finite


def round_nearest_scalar(value: float, exp_bits: int, sig_bits: int) -> float:
    """Round one value to the target format with deterministic nearest-even."""

    if not math.isfinite(value) or value == 0.0:
        return value
    sign = -1.0 if value < 0.0 else 1.0
    abs_value = abs(value)
    min_exp, _, max_finite = format_limits(exp_bits, sig_bits)
    exponent = math.floor(math.log(abs_value) / math.log(2.0))
    spacing_exp = max(exponent, min_exp) - sig_bits
    scale = 2.0**spacing_exp
    scaled = abs_value / scale
    floor_value = math.floor(scaled)
    fraction = scaled - floor_value
    if fraction < 0.5:
        rounded_units = floor_value
    elif fraction > 0.5:
        rounded_units = floor_value + 1.0
    else:
        rounded_units = floor_value if floor_value % 2.0 == 0.0 else floor_value + 1.0
    rounded = rounded_units * scale
    if rounded > max_finite:
        return math.copysign(math.inf, sign)
    return sign * rounded


def configure_format(spec: FormatSpec, random_state: int, digits_threshold: float) -> None:
    """Configure noisefloat's Python stochastic fallback for one format."""

    configure(
        backend="numpy",
        exp_bits=spec.exp_bits,
        sig_bits=spec.sig_bits,
        n_samples=SAMPLES,
        random_state=random_state,
        confidence=CONFIDENCE,
        digits_threshold=digits_threshold,
        zero_digits_threshold=digits_threshold,
        trace=False,
    )
    reset_chopper_cache()


def deterministic_sum_cubes_python(n: int, spec: FormatSpec) -> float:
    """Sequential deterministic round-to-nearest accumulation."""

    total = 0.0
    for i in range(1, n + 1):
        value = round_nearest_scalar(float(i), spec.exp_bits, spec.sig_bits)
        square = round_nearest_scalar(value * value, spec.exp_bits, spec.sig_bits)
        cube = round_nearest_scalar(square * value, spec.exp_bits, spec.sig_bits)
        total = round_nearest_scalar(total + cube, spec.exp_bits, spec.sig_bits)
        if not math.isfinite(total):
            break
    return float(total)


def stochastic_round_numpy(values: np.ndarray) -> np.ndarray:
    """Apply noisefloat's configured stochastic rounding to a tiny vector."""

    return _chopper.numpy(np.asarray(values, dtype=np.float64))


def stochastic_sum_cubes_python(
    n: int,
    spec: FormatSpec,
    random_state: int,
    digits_threshold: float,
) -> tuple[float, ...]:
    """Pure-Python fallback for sequential stochastic accumulation."""

    configure_format(spec, random_state, digits_threshold)
    total = np.zeros(SAMPLES, dtype=np.float64)
    for i in range(1, n + 1):
        value = stochastic_round_numpy(np.full(SAMPLES, float(i), dtype=np.float64))
        square = stochastic_round_numpy(value * value)
        cube = stochastic_round_numpy(square * value)
        total = stochastic_round_numpy(total + cube)
        if not np.all(np.isfinite(total)):
            break
    return tuple(float(value) for value in total)


def _try_numba():
    try:
        from numba import njit
    except Exception:
        return None
    return njit


_NJIT = _try_numba()


if _NJIT is not None:

    @_NJIT(cache=True)
    def _format_limits_numba(exp_bits, sig_bits):
        max_exp = float(2 ** (exp_bits - 1) - 1)
        min_exp = -max_exp + 1.0
        max_finite = (2.0 - 2.0 ** (-float(sig_bits))) * (2.0**max_exp)
        return min_exp, max_finite


    @_NJIT(cache=True)
    def _round_nearest_numba(x, exp_bits, sig_bits):
        if not math.isfinite(x) or x == 0.0:
            return x
        sign = 1.0
        if x < 0.0:
            sign = -1.0
            x = -x
        min_exp, max_finite = _format_limits_numba(exp_bits, sig_bits)
        exponent = math.floor(math.log(x) / math.log(2.0))
        spacing_exp = exponent
        if spacing_exp < min_exp:
            spacing_exp = min_exp
        spacing_exp -= float(sig_bits)
        scale = 2.0**spacing_exp
        scaled = x / scale
        floor_value = math.floor(scaled)
        fraction = scaled - floor_value
        if fraction < 0.5:
            rounded_units = floor_value
        elif fraction > 0.5:
            rounded_units = floor_value + 1.0
        elif floor_value % 2.0 == 0.0:
            rounded_units = floor_value
        else:
            rounded_units = floor_value + 1.0
        rounded = rounded_units * scale
        if rounded > max_finite:
            return sign * math.inf
        return sign * rounded


    @_NJIT(cache=True)
    def _round_stochastic_numba(x, exp_bits, sig_bits):
        if not math.isfinite(x) or x == 0.0:
            return x
        sign = 1.0
        if x < 0.0:
            sign = -1.0
            x = -x
        min_exp, max_finite = _format_limits_numba(exp_bits, sig_bits)
        exponent = math.floor(math.log(x) / math.log(2.0))
        spacing_exp = exponent
        if spacing_exp < min_exp:
            spacing_exp = min_exp
        spacing_exp -= float(sig_bits)
        scale = 2.0**spacing_exp
        scaled = x / scale
        floor_value = math.floor(scaled)
        if np.random.random() < 0.5:
            rounded = floor_value * scale
        else:
            rounded = (floor_value + 1.0) * scale
        if rounded > max_finite:
            return sign * math.inf
        return sign * rounded


    @_NJIT(cache=True)
    def _deterministic_sum_cubes_numba(n, exp_bits, sig_bits):
        total = 0.0
        for i in range(1, n + 1):
            value = _round_nearest_numba(float(i), exp_bits, sig_bits)
            square = _round_nearest_numba(value * value, exp_bits, sig_bits)
            cube = _round_nearest_numba(square * value, exp_bits, sig_bits)
            total = _round_nearest_numba(total + cube, exp_bits, sig_bits)
            if not math.isfinite(total):
                break
        return total


    @_NJIT(cache=True)
    def _stochastic_sum_cubes_numba(n, exp_bits, sig_bits, random_state):
        np.random.seed(random_state)
        totals = np.zeros(3, dtype=np.float64)
        for i in range(1, n + 1):
            all_done = True
            for sample in range(3):
                if math.isfinite(totals[sample]):
                    all_done = False
                    value = _round_stochastic_numba(float(i), exp_bits, sig_bits)
                    square = _round_stochastic_numba(value * value, exp_bits, sig_bits)
                    cube = _round_stochastic_numba(square * value, exp_bits, sig_bits)
                    totals[sample] = _round_stochastic_numba(
                        totals[sample] + cube, exp_bits, sig_bits
                    )
            if all_done:
                break
        return totals


def resolve_engine(engine: str) -> str:
    if engine == "auto":
        return "numba" if _NJIT is not None else "python"
    if engine == "numba" and _NJIT is None:
        raise RuntimeError(
            "The numba engine was requested but numba is not installed. "
            "Use --engine python for the pure-Python fallback."
        )
    return engine


def deterministic_sum_cubes(n: int, spec: FormatSpec, engine: str) -> float:
    if engine == "numba" and _NJIT is not None:
        return float(_deterministic_sum_cubes_numba(n, spec.exp_bits, spec.sig_bits))
    return deterministic_sum_cubes_python(n, spec)


def stochastic_sum_cubes(
    n: int,
    spec: FormatSpec,
    random_state: int,
    digits_threshold: float,
    engine: str,
) -> tuple[float, ...]:
    if engine == "numba" and _NJIT is not None:
        return tuple(
            float(value)
            for value in _stochastic_sum_cubes_numba(
                n, spec.exp_bits, spec.sig_bits, random_state
            )
        )
    return stochastic_sum_cubes_python(n, spec, random_state, digits_threshold)


def analyze_sum_cubes(
    *,
    n: int,
    spec: FormatSpec,
    random_state: int,
    digits_threshold: float,
    engine: str,
) -> SumCubesReport:
    """Compute one ``n``/format pair with deterministic and stochastic rounding."""

    exact = exact_sum_cubes(n)
    deterministic = deterministic_sum_cubes(n, spec, engine)
    deterministic_abs_error = abs(deterministic - float(exact))
    deterministic_relative_error = deterministic_abs_error / abs(float(exact))
    deterministic_digits = reference_digits(deterministic, exact)

    configure_format(spec, random_state, digits_threshold)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stochastic_samples = stochastic_sum_cubes(
            n, spec, random_state, digits_threshold, engine
        )

    representative, std, sample_digit_count = sample_digits(stochastic_samples)
    stochastic_abs_error = abs(representative - float(exact))
    stochastic_relative_error = stochastic_abs_error / abs(float(exact))
    stochastic_reference_digits = reference_digits(representative, exact)

    return SumCubesReport(
        n=n,
        format_name=spec.name,
        exp_bits=spec.exp_bits,
        sig_bits=spec.sig_bits,
        exact_sum=exact,
        deterministic_sum=deterministic,
        deterministic_abs_error=deterministic_abs_error,
        deterministic_relative_error=deterministic_relative_error,
        deterministic_reference_digits=deterministic_digits,
        stochastic_representative=representative,
        stochastic_std=std,
        stochastic_sample_digits=sample_digit_count,
        stochastic_reference_digits=stochastic_reference_digits,
        stochastic_abs_error=stochastic_abs_error,
        stochastic_relative_error=stochastic_relative_error,
        stochastic_samples=stochastic_samples,
        engine=engine,
    )


def analyze_many(
    *,
    n_values: Iterable[int],
    formats: Iterable[str],
    random_state: int,
    digits_threshold: float,
    engine: str,
) -> list[SumCubesReport]:
    resolved_engine = resolve_engine(engine)
    reports = []
    for format_name in formats:
        spec = FORMATS[format_name]
        for n in n_values:
            reports.append(
                analyze_sum_cubes(
                    n=n,
                    spec=spec,
                    random_state=random_state,
                    digits_threshold=digits_threshold,
                    engine=resolved_engine,
                )
            )
    return reports


def row_for_report(report: SumCubesReport) -> dict[str, object]:
    return {
        "n": report.n,
        "format": report.format_name,
        "exp_bits": report.exp_bits,
        "sig_bits": report.sig_bits,
        "exact_sum": report.exact_sum,
        "deterministic_sum": report.deterministic_sum,
        "deterministic_abs_error": report.deterministic_abs_error,
        "deterministic_relative_error": report.deterministic_relative_error,
        "deterministic_reference_digits": report.deterministic_reference_digits,
        "stochastic_representative": report.stochastic_representative,
        "stochastic_std": report.stochastic_std,
        "stochastic_sample_digits": report.stochastic_sample_digits,
        "stochastic_reference_digits": report.stochastic_reference_digits,
        "stochastic_abs_error": report.stochastic_abs_error,
        "stochastic_relative_error": report.stochastic_relative_error,
        "stochastic_samples": ";".join(
            f"{value:.17g}" for value in report.stochastic_samples
        ),
        "engine": report.engine,
    }


def write_csv(reports: Sequence[SumCubesReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row_for_report(report) for report in reports]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_reports(reports: Sequence[SumCubesReport], digits_threshold: float) -> None:
    print("Term-by-term sum of cubes under deterministic and stochastic rounding")
    print("=" * 104)
    print("S = sum_{i=1}^n i^3 = (n(n+1)/2)^2")
    if reports:
        print(f"Engine: {reports[0].engine}")
    print()
    print(
        f"{'format':<7} {'n':>10} {'det_digits':>11} {'stoch_digits':>13} "
        f"{'stoch_ref':>11} {'det_relerr':>13} {'stoch_relerr':>13}"
    )
    print("-" * 104)
    for report in reports:
        print(
            f"{report.format_name:<7} {report.n:10d} "
            f"{report.deterministic_reference_digits:11.3f} "
            f"{report.stochastic_sample_digits:13.3f} "
            f"{report.stochastic_reference_digits:11.3f} "
            f"{report.deterministic_relative_error:13.6e} "
            f"{report.stochastic_relative_error:13.6e}"
        )
    print()
    print(f"Digits threshold: {digits_threshold:.3f}")
    print("Rows with stochastic sample-based digits below the threshold are unstable.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_N_VALUES),
        help="Term counts to evaluate with sequential accumulation.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Backward-compatible shorthand for a single n value.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=tuple(FORMATS),
        default=list(FORMATS),
        help="Floating-point formats to test.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--digits-threshold", type=float, default=6.0)
    parser.add_argument(
        "--engine",
        choices=("auto", "numba", "python"),
        default="auto",
        help="Execution engine for the sequential stochastic loop.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("examples/outputs/sum_cubes_precision.csv"),
        help="CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    n_values = [args.n] if args.n is not None else args.n_values
    reports = analyze_many(
        n_values=n_values,
        formats=args.formats,
        random_state=args.random_state,
        digits_threshold=args.digits_threshold,
        engine=args.engine,
    )
    print_reports(reports, args.digits_threshold)
    write_csv(reports, args.output_csv)
    print(f"\nwrote CSV to: {args.output_csv}")


if __name__ == "__main__":
    main()
