"""Run noisefloat recreations of CADNA tutorial examples 1--7.

The reference examples are CADNA C/C++ tutorial programs.  This script mirrors
those algorithms with ``NFloat`` values, prints CADNA-style instability
summaries after each example, and writes a CSV report with source-of-instability
and runtime comparisons.
"""

from __future__ import annotations

import argparse
import csv
import io
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

import numpy as np

from noisefloat import (
    NFloat,
    clear_diagnostics,
    configure,
    get_diagnostics_summary,
    print_diagnostics_summary,
    reset_chopper_cache,
    sqrt,
)
from noisefloat.diagnostics import (
    BRANCHING,
    CANCELLATION,
    INTRINSIC,
    MATHEMATICAL,
    UNSTABLE_DIVISION,
    UNSTABLE_MULTIPLICATION,
    UNSTABLE_POWER,
)


CADNA_KIND_ORDER = (
    UNSTABLE_DIVISION,
    UNSTABLE_POWER,
    UNSTABLE_MULTIPLICATION,
    BRANCHING,
    MATHEMATICAL,
    INTRINSIC,
    CANCELLATION,
)

CADNA_KIND_LABELS = {
    UNSTABLE_DIVISION: "UNSTABLE DIVISION(S)",
    UNSTABLE_POWER: "UNSTABLE POWER FUNCTION(S)",
    UNSTABLE_MULTIPLICATION: "UNSTABLE MULTIPLICATION(S)",
    BRANCHING: "UNSTABLE BRANCHING(S)",
    MATHEMATICAL: "UNSTABLE MATHEMATICAL FUNCTION(S)",
    INTRINSIC: "UNSTABLE INTRINSIC FUNCTION(S)",
    CANCELLATION: "LOSS(ES) OF ACCURACY DUE TO CANCELLATION(S)",
}


@dataclass(frozen=True)
class CadnaReference:
    """Structured CADNA C reference counts for one example."""

    counts: dict[str, int]
    source: str

    @property
    def total(self) -> int:
        return int(sum(self.counts.values()))


@dataclass(frozen=True)
class Example:
    number: int
    title: str
    cadna_references: dict[str, CadnaReference]
    run: Callable[[], None]
    run_deterministic_quiet: Callable[[], None]


def configure_cadna_like(precision: str, random_state: int | None) -> None:
    global CURRENT_PRECISION
    if precision == "single":
        exp_bits, sig_bits = 8, 23
    elif precision == "double":
        exp_bits, sig_bits = 11, 52
    else:  # pragma: no cover - defensive programming
        raise ValueError(f"unknown precision: {precision}")

    configure(
        backend="numpy",
        exp_bits=exp_bits,
        sig_bits=sig_bits,
        n_samples=3,
        random_state=random_state,
        diagnostics_level="summary",
        trace=False,
        digits_threshold=0.5,
        zero_digits_threshold=0.5,
        cancellation_ratio_threshold=1e-6,
    )
    CURRENT_PRECISION = precision
    reset_chopper_cache()
    clear_diagnostics()


JACOBI_MAX_ITERATIONS = 1000
PRECISION_ORDER = ("single", "double")
CURRENT_PRECISION = "double"


def fmt(value) -> str:
    if isinstance(value, NFloat):
        digits = float(np.asarray(value.digits).mean())
        if digits < 0.5:
            return "@.0"
        return f"{float(np.asarray(value.mean)):.12E}  digits={digits:.3f}"
    return f"{float(value):.12E}"


def solution_label() -> str:
    return "float" if CURRENT_PRECISION == "single" else "double"


def mean_scalar(value: NFloat) -> float:
    return float(np.asarray(value.mean))


def fabsf_like(value: NFloat) -> float:
    """Mimic C ``fabsf`` on a stochastic value via float32 conversion."""
    return abs(float(np.float32(mean_scalar(value))))


def deterministic_scalar(value: float | int):
    """Return a deterministic scalar matching the currently requested precision."""
    if CURRENT_PRECISION == "single":
        return np.float32(value)
    return float(value)


def deterministic_abs(value) -> float:
    if CURRENT_PRECISION == "single":
        return abs(float(np.float32(value)))
    return abs(float(value))


def print_summary() -> None:
    print("Noisefloat detected source(s):")
    print_diagnostics_summary()
    print()


def total_count(counts: dict[str, int]) -> int:
    return int(sum(int(value) for value in counts.values()))


def format_counts(counts: dict[str, int]) -> str:
    nonzero = [
        f"{CADNA_KIND_LABELS.get(kind, kind)}={int(count)}"
        for kind, count in counts.items()
        if int(count) != 0
    ]
    return "; ".join(nonzero) if nonzero else "none"


def print_reference_comparison(reference: CadnaReference, detected: dict[str, int]) -> None:
    print("CADNA C reference:")
    if reference.total == 0:
        print("No instability detected")
    else:
        print(f"There are {reference.total} numerical instabilities")
        for kind in CADNA_KIND_ORDER:
            count = int(reference.counts.get(kind, 0))
            if count:
                print(f"{count} {CADNA_KIND_LABELS.get(kind, kind)}")
    print("Comparison by source:")
    for kind in CADNA_KIND_ORDER:
        detected_count = int(detected.get(kind, 0))
        reference_count = int(reference.counts.get(kind, 0))
        if detected_count or reference_count:
            print(
                f"  {CADNA_KIND_LABELS.get(kind, kind)}: "
                f"noisefloat={detected_count}, reference={reference_count}"
            )
    print(f"Reference source: {reference.source}")
    print()


def write_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measure_runtime(function: Callable[[], None], repeats: int) -> tuple[float, float, float]:
    samples = []
    for _ in range(max(repeats, 1)):
        start = perf_counter()
        function()
        samples.append(perf_counter() - start)
    values = np.asarray(samples, dtype=np.float64)
    return float(values.mean()), float(np.median(values)), float(values.min())


def run_without_output(function: Callable[[], None]) -> None:
    with redirect_stdout(io.StringIO()):
        function()


def build_summary_row(
    example: Example,
    precision: str,
    detected: dict[str, int],
    reference: CadnaReference,
    deterministic_runtime: tuple[float, float, float],
    nfloat_runtime: tuple[float, float, float],
    runtime_repeats: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "example": example.number,
        "title": example.title,
        "precision": precision,
        "noisefloat_total_instabilities": total_count(detected),
        "reference_total_instabilities": reference.total,
        "noisefloat_sources": format_counts(detected),
        "reference_sources": format_counts(reference.counts),
        "reference_source_file": reference.source,
        "runtime_repeats": runtime_repeats,
        "deterministic_runtime_precision": precision,
        "deterministic_runtime_mean_seconds": deterministic_runtime[0],
        "deterministic_runtime_median_seconds": deterministic_runtime[1],
        "deterministic_runtime_min_seconds": deterministic_runtime[2],
        "deterministic_single_runtime_mean_seconds": (
            deterministic_runtime[0] if precision == "single" else ""
        ),
        "deterministic_single_runtime_median_seconds": (
            deterministic_runtime[1] if precision == "single" else ""
        ),
        "deterministic_single_runtime_min_seconds": (
            deterministic_runtime[2] if precision == "single" else ""
        ),
        "deterministic_double_runtime_mean_seconds": (
            deterministic_runtime[0] if precision == "double" else ""
        ),
        "deterministic_double_runtime_median_seconds": (
            deterministic_runtime[1] if precision == "double" else ""
        ),
        "deterministic_double_runtime_min_seconds": (
            deterministic_runtime[2] if precision == "double" else ""
        ),
        "noisefloat_runtime_mean_seconds": nfloat_runtime[0],
        "noisefloat_runtime_median_seconds": nfloat_runtime[1],
        "noisefloat_runtime_min_seconds": nfloat_runtime[2],
        "runtime_overhead_ratio": (
            nfloat_runtime[0] / deterministic_runtime[0]
            if deterministic_runtime[0] > 0.0
            else float("nan")
        ),
    }
    for kind in CADNA_KIND_ORDER:
        row[f"noisefloat_{kind}"] = int(detected.get(kind, 0))
        row[f"reference_{kind}"] = int(reference.counts.get(kind, 0))
    return row


def example_1_rump() -> None:
    x = NFloat(10864.0)
    y = NFloat(18817.0)
    p1 = 9.0 * x * x * x * x - y * y * y * y + 2.0 * y * y
    print(f"res={fmt(p1)}")

    x = NFloat(1.0 / 3.0)
    y = NFloat(2.0 / 3.0)
    p2 = 9.0 * x * x * x * x - y * y * y * y + 2.0 * y * y
    print(f"res={fmt(p2)}")


def example_2_quadratic() -> None:
    a = NFloat(0.3)
    b = NFloat(-2.1)
    c = NFloat(3.675)
    b = b / a
    c = c / a
    d = b * b - 4.0 * c
    print(f"d = {fmt(d)}")
    if bool(np.any(d.is_numerical_zero())):
        x1 = -b * 0.5
        print(f"Discriminant is zero. The {solution_label()} solution is {fmt(x1)}")
    elif d > 0.0:
        x1 = (-b - sqrt(d)) * 0.5
        x2 = (-b + sqrt(d)) * 0.5
        print(f"Two real solutions: x1={fmt(x1)}, x2={fmt(x2)}")
    else:
        x1 = -b * 0.5
        x2 = sqrt(-d) * 0.5
        print(f"Two complex solutions: {fmt(x1)} +/- i*{fmt(x2)}")


def example_3_hilbert_determinant() -> None:
    n = 11
    a = [[NFloat(1.0 / (i + j + 1.0)) for j in range(n)] for i in range(n)]
    det = NFloat(1.0)
    for i in range(n - 1):
        print(f"Pivot number {i + 1:2d} = {fmt(a[i][i])}")
        det = det * a[i][i]
        aux = 1.0 / a[i][i]
        for j in range(i + 1, n):
            a[i][j] = a[i][j] * aux
        for j in range(i + 1, n):
            aux = a[j][i]
            for k in range(i + 1, n):
                a[j][k] = a[j][k] - aux * a[i][k]
    print(f"Pivot number {n:2d} = {fmt(a[n - 1][n - 1])}")
    det = det * a[n - 1][n - 1]
    print(f"Determinant = {fmt(det)}")


def example_4_muller_recurrence() -> None:
    a = NFloat(5.5)
    b = NFloat(61.0 / 11.0)
    for i in range(3, 31):
        c = b
        b = 111.0 - 1130.0 / b + 3000.0 / (a * b)
        a = c
        print(f"U({i:2d}) = {fmt(b)}")
    print("The true limit is 6.")


def example_5_newton() -> None:
    eps = 1.0e-12
    y = NFloat(0.5)
    i = 0
    for i in range(1, 101):
        x = y
        numerator = 1.47 * x**3 + 1.19 * x**2 - 1.83 * x + 0.45
        denominator = 4.41 * x**2 + 2.38 * x - 1.83
        y = x - numerator / denominator
        step = abs(x - y)
        print(f"x({i:3d}) = {fmt(y)}, diff = {fmt(step)}")
        if step < eps:
            break


def example_6_gaussian_pivoting() -> None:
    a = [
        [21.0, 130.0, 0.0, 2.1, 153.1],
        [13.0, 80.0, 4.74e+8, 752.0, 849.74],
        [0.0, -0.4, 3.9816e+8, 4.2, 7.7816],
        [0.0, 0.0, 1.7, 9.0e-9, 2.6e-8],
    ]

    #{  21.0, 130.0,       0.0,    2.1,  153.1},
    #{  13.0,  80.0,   4.74e+8,  752.0, 849.74},
    #{   0.0,  -0.4, 3.9816e+8,    4.2, 7.7816},
    #{   0.0,   0.0,       1.7, 9.0E-9, 2.6e-8}};
    # double_st   xsol[IDIM]={1., 1., 1.e-8,1.};

    xsol = [1.0, 1.0, 1.0e-8, 1.0]
    amat = [[NFloat(value) for value in row] for row in a]
    n = 4
    pivot_rows = [0, 1, 3]
    for i in range(n - 1):
        pivot = pivot_rows[i]
        print(f"ll={pivot}")
        if pivot != i:
            amat[i][i : n + 1], amat[pivot][i : n + 1] = (
                amat[pivot][i : n + 1],
                amat[i][i : n + 1],
            )
        aux = 1.0 / amat[i][i]
        for j in range(i + 1, n + 1):
            amat[i][j] = amat[i][j] * aux
        for k in range(i + 1, n):
            aux = amat[k][i]
            for j in range(i + 1, n + 1):
                amat[k][j] = amat[k][j] - aux * amat[i][j]
    amat[n - 1][n] = amat[n - 1][n] / amat[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        for j in range(i + 1, n):
            amat[i][n] = amat[i][n] - amat[i][j] * amat[j][n]
    for i in range(n):
        print(f"x_sol({i + 1}) = {fmt(amat[i][n])}  (true value: {xsol[i]:.7E})")


def random1_state():
    state = {"nrand": 23}

    def random1() -> float:
        state["nrand"] = (state["nrand"] * 5363 + 143) % 1387
        return 2.0 * state["nrand"] / 1387.0 - 1.0

    return random1


def example_7_jacobi() -> None:
    eps = 1.0e-4
    ndim = 20
    niter = JACOBI_MAX_ITERATIONS
    xsol_values = [
        1.7,
        -4746.89,
        50.23,
        -245.32,
        4778.29,
        -75.73,
        3495.43,
        4.35,
        452.98,
        -2.76,
        8239.24,
        3.46,
        1000.0,
        -5.0,
        3642.4,
        735.36,
        1.7,
        -2349.17,
        -8247.52,
        9843.57,
    ]
    rand = random1_state()
    xsol = [NFloat(value) for value in xsol_values]
    a = [[NFloat(rand()) for _ in range(ndim)] for _ in range(ndim)]
    for i in range(ndim):
        a[i][i] = a[i][i] + 4.500002
    b = []
    y = []
    for i in range(ndim):
        aux = NFloat(0.0)
        for j in range(ndim):
            aux = aux + a[i][j] * xsol[j]
        b.append(aux)
        y.append(NFloat(10.0))

    iterations = min(niter, 38)
    for _iteration in range(1, iterations + 1):
        anorm = 0.0
        x = list(y)
        for j in range(ndim):
            aux = b[j]
            for k in range(ndim):
                if k != j:
                    aux = aux - a[j][k] * x[k]
            y[j] = aux / a[j][j]
            diff = fabsf_like(x[j] - y[j])
            if diff > anorm:
                anorm = diff

    print(f"niter = {iterations}  (verification cap: {JACOBI_MAX_ITERATIONS})")
    for i in range(ndim):
        residue = -b[i]
        for j in range(ndim):
            residue = residue + a[i][j] * y[j]
        print(
            f"x_sol({i + 1:2d}) = {fmt(y[i])}  "
            f"(true value: {xsol_values[i]:.7E}), residue({i + 1:2d}) = {fmt(residue)}"
        )


def example_1_rump_deterministic_quiet() -> None:
    d = deterministic_scalar
    x = d(10864.0)
    y = d(18817.0)
    _ = d(9.0) * x * x * x * x - y * y * y * y + d(2.0) * y * y

    x = d(1.0) / d(3.0)
    y = d(2.0) / d(3.0)
    _ = d(9.0) * x * x * x * x - y * y * y * y + d(2.0) * y * y


def example_2_quadratic_deterministic_quiet() -> None:
    dscalar = deterministic_scalar
    a = dscalar(0.3)
    b = dscalar(-2.1)
    c = dscalar(3.675)
    b = b / a
    c = c / a
    d = b * b - dscalar(4.0) * c
    if d == 0.0:
        _ = -b * dscalar(0.5)
    elif d > 0.0:
        _ = (-b - np.sqrt(d)) * dscalar(0.5)
        _ = (-b + np.sqrt(d)) * dscalar(0.5)
    else:
        _ = -b * dscalar(0.5)
        _ = np.sqrt(-d) * dscalar(0.5)


def example_3_hilbert_deterministic_quiet() -> None:
    d = deterministic_scalar
    n = 11
    a = [[d(1.0) / d(i + j + 1.0) for j in range(n)] for i in range(n)]
    det = d(1.0)
    for i in range(n - 1):
        det = det * a[i][i]
        aux = d(1.0) / a[i][i]
        for j in range(i + 1, n):
            a[i][j] = a[i][j] * aux
        for j in range(i + 1, n):
            aux = a[j][i]
            for k in range(i + 1, n):
                a[j][k] = a[j][k] - aux * a[i][k]
    det = det * a[n - 1][n - 1]
    _ = det


def example_4_muller_deterministic_quiet() -> None:
    d = deterministic_scalar
    a = d(5.5)
    b = d(61.0) / d(11.0)
    for _ in range(3, 31):
        c = b
        b = d(111.0) - d(1130.0) / b + d(3000.0) / (a * b)
        a = c


def example_5_newton_deterministic_quiet() -> None:
    d = deterministic_scalar
    eps = d(1.0e-12)
    y = d(0.5)
    for _ in range(1, 101):
        x = y
        numerator = d(1.47) * x**3 + d(1.19) * x**2 - d(1.83) * x + d(0.45)
        denominator = d(4.41) * x**2 + d(2.38) * x - d(1.83)
        y = x - numerator / denominator
        step = deterministic_abs(x - y)
        if step < eps:
            break


def example_6_gaussian_deterministic_quiet() -> None:
    d = deterministic_scalar
    a = [
        [d(21.0), d(130.0), d(0.0), d(2.1), d(153.1)],
        [d(13.0), d(80.0), d(4.74e+8), d(752.0), d(849.74)],
        [d(0.0), d(-0.4), d(3.9816e+8), d(4.2), d(7.7816)],
        [d(0.0), d(0.0), d(1.7), d(9.0e-9), d(2.6e-8)],
    ]
    amat = [list(row) for row in a]
    n = 4
    pivot_rows = [0, 1, 3]
    for i in range(n - 1):
        pivot = pivot_rows[i]
        if pivot != i:
            amat[i][i : n + 1], amat[pivot][i : n + 1] = (
                amat[pivot][i : n + 1],
                amat[i][i : n + 1],
            )
        aux = d(1.0) / amat[i][i]
        for j in range(i + 1, n + 1):
            amat[i][j] = amat[i][j] * aux
        for k in range(i + 1, n):
            aux = amat[k][i]
            for j in range(i + 1, n + 1):
                amat[k][j] = amat[k][j] - aux * amat[i][j]
    amat[n - 1][n] = amat[n - 1][n] / amat[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        for j in range(i + 1, n):
            amat[i][n] = amat[i][n] - amat[i][j] * amat[j][n]


def example_7_jacobi_deterministic_quiet() -> None:
    d = deterministic_scalar
    ndim = 20
    xsol_values = [
        d(1.7),
        d(-4746.89),
        d(50.23),
        d(-245.32),
        d(4778.29),
        d(-75.73),
        d(3495.43),
        d(4.35),
        d(452.98),
        d(-2.76),
        d(8239.24),
        d(3.46),
        d(1000.0),
        d(-5.0),
        d(3642.4),
        d(735.36),
        d(1.7),
        d(-2349.17),
        d(-8247.52),
        d(9843.57),
    ]
    rand = random1_state()
    a = [[d(rand()) for _ in range(ndim)] for _ in range(ndim)]
    for i in range(ndim):
        a[i][i] = a[i][i] + d(4.500002)
    b = []
    y = []
    for i in range(ndim):
        aux = d(0.0)
        for j in range(ndim):
            aux = aux + a[i][j] * xsol_values[j]
        b.append(aux)
        y.append(d(10.0))

    iterations = min(JACOBI_MAX_ITERATIONS, 38)
    for _iteration in range(1, iterations + 1):
        anorm = 0.0
        x = list(y)
        for j in range(ndim):
            aux = b[j]
            for k in range(ndim):
                if k != j:
                    aux = aux - a[j][k] * x[k]
            y[j] = aux / a[j][j]
            diff = deterministic_abs(x[j] - y[j])
            if diff > anorm:
                anorm = diff


EXAMPLES: tuple[Example, ...] = (
    Example(
        1,
        "Polynomial function of two variables: Rump equation",
        {
            "single": CadnaReference(
                counts={CANCELLATION: 1},
                source="rump_cad.cc / output_float.txt",
            ),
            "double": CadnaReference(
                counts={CANCELLATION: 2},
                source="rump_cad.cc / output.txt",
            ),
        },
        example_1_rump,
        example_1_rump_deterministic_quiet,
    ),
    Example(
        2,
        "Second-order equation",
        {
            "single": CadnaReference(
                counts={CANCELLATION: 1},
                source="ex2_cad.cc / output_float.txt",
            ),
            "double": CadnaReference(
                counts={CANCELLATION: 1},
                source="ex2_cad.cc / output.txt",
            ),
        },
        example_2_quadratic,
        example_2_quadratic_deterministic_quiet,
    ),
    Example(
        3,
        "Determinant of Hilbert's matrix",
        {
            "single": CadnaReference(
                counts={UNSTABLE_DIVISION: 4, UNSTABLE_MULTIPLICATION: 44},
                source="hilbert_cad.cc / output_float.txt",
            ),
            "double": CadnaReference(
                counts={},
                source="hilbert_cad.cc / output.txt",
            ),
        },
        example_3_hilbert_determinant,
        example_3_hilbert_deterministic_quiet,
    ),
    Example(
        4,
        "J.-M. Muller recurrence",
        {
            "single": CadnaReference(
                counts={UNSTABLE_DIVISION: 5, UNSTABLE_MULTIPLICATION: 2},
                source="muller_cad.cc / output_float.txt",
            ),
            "double": CadnaReference(
                counts={UNSTABLE_DIVISION: 6, UNSTABLE_MULTIPLICATION: 2},
                source="muller_cad.cc / output.txt",
            ),
        },
        example_4_muller_recurrence,
        example_4_muller_deterministic_quiet,
    ),
    Example(
        5,
        "Newton's method",
        {
            "single": CadnaReference(
                counts={
                    UNSTABLE_DIVISION: 91,
                    BRANCHING: 89,
                    INTRINSIC: 60,
                    CANCELLATION: 96,
                },
                source="newton_cad.cc / output_float.txt",
            ),
            "double": CadnaReference(
                counts={
                    UNSTABLE_DIVISION: 76,
                    BRANCHING: 77,
                    INTRINSIC: 54,
                    CANCELLATION: 273,
                },
                source="newton_cad.cc / output.txt",
            ),
        },
        example_5_newton,
        example_5_newton_deterministic_quiet,
    ),
    Example(
        6,
        "Gaussian elimination with partial pivoting",
        {
            "single": CadnaReference(
                counts={BRANCHING: 1, CANCELLATION: 1},
                source="gauss_cad.cc / output_float.txt",
            ),
            "double": CadnaReference(
                counts={CANCELLATION: 1},
                source="gauss_cad.cc / output.txt",
            ),
        },
        example_6_gaussian_pivoting,
        example_6_gaussian_deterministic_quiet,
    ),
    Example(
        7,
        "Jacobi iterative method",
        {
            "single": CadnaReference(
                counts={BRANCHING: 111, INTRINSIC: 35, CANCELLATION: 230},
                source="jacobi_cad.cc / output_float.txt",
            ),
            "double": CadnaReference(
                counts={CANCELLATION: 406},
                source="jacobi_cad.cc / output.txt",
            ),
        },
        example_7_jacobi,
        example_7_jacobi_deterministic_quiet,
    ),
)


def run_example(
    example: Example,
    precision: str,
    random_state: int | None,
    runtime_repeats: int,
) -> dict[str, object]:
    reference = example.cadna_references[precision]
    configure_cadna_like(precision, random_state)
    print("=" * 80)
    print(f"Example {example.number}: {example.title}")
    print(f"Precision: {precision}")
    print("-" * 80)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        example.run()
    print("-" * 80)
    summary = get_diagnostics_summary()
    detected = dict(summary.get("by_kind", {}))
    print_summary()
    print_reference_comparison(reference, detected)

    deterministic_runtime = measure_runtime(
        example.run_deterministic_quiet,
        runtime_repeats,
    )

    def run_nfloat_timed() -> None:
        configure_cadna_like(precision, random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_without_output(example.run)

    nfloat_runtime = measure_runtime(run_nfloat_timed, runtime_repeats)
    row = build_summary_row(
        example=example,
        precision=precision,
        detected=detected,
        reference=reference,
        deterministic_runtime=deterministic_runtime,
        nfloat_runtime=nfloat_runtime,
        runtime_repeats=runtime_repeats,
    )
    print(
        "Runtime (mean seconds): "
        f"deterministic_{precision}={row['deterministic_runtime_mean_seconds']:.6e}, "
        f"noisefloat={row['noisefloat_runtime_mean_seconds']:.6e}, "
        f"overhead={row['runtime_overhead_ratio']:.3f}x"
    )
    print()
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples",
        default="1,2,3,4,5,6,7",
        help="comma-separated CADNA example numbers to run",
    )
    parser.add_argument(
        "--jacobi-max-iterations",
        type=int,
        default=1000,
        help=(
            "cap for Example 7; the official CADNA program uses 1000 and "
            "stops early when all iteration differences are stochastic zero"
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=12,
        help=(
            "deterministic stochastic-rounding seed for the Python "
            "recreation; CADNA C uses its own internal random stream"
        ),
    )
    parser.add_argument(
        "--precisions",
        default="single,double",
        help="comma-separated precisions to run: single, double, or both",
    )
    parser.add_argument(
        "--runtime-repeats",
        type=int,
        default=3,
        help="number of repeated runtime measurements for each mode",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cadna_examples_1_7"),
        help="directory for CSV summary outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global JACOBI_MAX_ITERATIONS
    JACOBI_MAX_ITERATIONS = args.jacobi_max_iterations
    selected = {int(item.strip()) for item in args.examples.split(",") if item.strip()}
    requested_precisions = [
        item.strip().lower() for item in args.precisions.split(",") if item.strip()
    ]
    invalid_precisions = [
        precision for precision in requested_precisions if precision not in PRECISION_ORDER
    ]
    if invalid_precisions:
        raise ValueError(
            f"unsupported precision(s): {invalid_precisions}; "
            f"expected subset of {PRECISION_ORDER}"
        )
    precision_order = [
        precision for precision in PRECISION_ORDER if precision in requested_precisions
    ]
    rows = []
    for precision in precision_order:
        for example in EXAMPLES:
            if example.number in selected:
                rows.append(
                    run_example(
                        example,
                        precision,
                        args.random_state,
                        args.runtime_repeats,
                    )
                )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "cadna_examples_1_7_summary.csv"
    write_csv(rows, summary_csv)
    print("=" * 80)
    print("Compact noisefloat summary")
    for row in rows:
        print(
            f"[{row['precision']}] Example {row['example']}: "
            f"total={row['noisefloat_total_instabilities']} "
            f"by_kind={format_counts(example_counts(row, 'noisefloat'))} | "
            f"CADNA reference total={row['reference_total_instabilities']} "
            f"by_kind={format_counts(example_counts(row, 'reference'))}"
        )
    print(f"Wrote summary CSV to: {summary_csv}")


def example_counts(row: dict[str, object], prefix: str) -> dict[str, int]:
    return {
        kind: int(row[f"{prefix}_{kind}"])
        for kind in CADNA_KIND_ORDER
        if int(row[f"{prefix}_{kind}"]) != 0
    }


if __name__ == "__main__":
    main()
