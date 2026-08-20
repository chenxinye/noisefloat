"""Run noisefloat recreations of CADNA tutorial examples 1--7 with PyTorch.

This script mirrors :mod:`cadna_examples_1_7` one-to-one, but configures
``noisefloat`` with the ``torch`` backend and explicitly places scalar tensors
on a requested device so the user can study CPU or CUDA execution.
"""

from __future__ import annotations

import argparse
import io
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import torch

import cadna_examples_1_7 as base
from noisefloat import (
    NFloat,
    clear_diagnostics,
    configure,
    get_diagnostics_summary,
    reset_chopper_cache,
    sqrt,
)
from noisefloat.functions import matmul


JACOBI_MAX_ITERATIONS = 1000
PRECISION_ORDER = ("single", "double")
CURRENT_PRECISION = "double"
CURRENT_DEVICE = torch.device("cpu")
FAST_JACOBI = False


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ts(value: float | int) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64, device=CURRENT_DEVICE)


def deterministic_dtype() -> torch.dtype:
    return torch.float32 if CURRENT_PRECISION == "single" else torch.float64


def tdet(value: float | int) -> torch.Tensor:
    return torch.tensor(value, dtype=deterministic_dtype(), device=CURRENT_DEVICE)


def nf(value: float | int) -> NFloat:
    return NFloat(ts(value))


def nf_from_samples(samples: torch.Tensor) -> NFloat:
    return NFloat(0.0, _samples=samples)


def nf_component(value: NFloat, index: int) -> NFloat:
    return nf_from_samples(value.samples[:, index])


def configure_cadna_like(
    precision: str,
    random_state: int | None,
    device: torch.device,
) -> None:
    global CURRENT_PRECISION, CURRENT_DEVICE
    if precision == "single":
        exp_bits, sig_bits = 8, 23
    elif precision == "double":
        exp_bits, sig_bits = 11, 52
    else:  # pragma: no cover - defensive programming
        raise ValueError(f"unknown precision: {precision}")

    CURRENT_PRECISION = precision
    CURRENT_DEVICE = device
    base.CURRENT_PRECISION = precision
    configure(
        backend="torch",
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
    reset_chopper_cache()
    clear_diagnostics()


def mean_scalar(value: NFloat) -> float:
    return float(np.asarray(value.mean))


def fabsf_like(value: NFloat) -> float:
    return abs(float(np.float32(mean_scalar(value))))


def solution_label() -> str:
    return "float" if CURRENT_PRECISION == "single" else "double"


def run_without_output(function: Callable[[], None]) -> None:
    with redirect_stdout(io.StringIO()):
        function()


def measure_runtime(function: Callable[[], None], repeats: int) -> tuple[float, float, float]:
    samples = []
    for _ in range(max(repeats, 1)):
        if CURRENT_DEVICE.type == "cuda":
            torch.cuda.synchronize(CURRENT_DEVICE)
        start = perf_counter()
        function()
        if CURRENT_DEVICE.type == "cuda":
            torch.cuda.synchronize(CURRENT_DEVICE)
        samples.append(perf_counter() - start)
    values = np.asarray(samples, dtype=np.float64)
    return float(values.mean()), float(np.median(values)), float(values.min())


def example_1_rump() -> None:
    x = nf(10864.0)
    y = nf(18817.0)
    p1 = nf(9.0) * x * x * x * x - y * y * y * y + nf(2.0) * y * y
    print(f"res={base.fmt(p1)}")

    x = nf(1.0 / 3.0)
    y = nf(2.0 / 3.0)
    p2 = nf(9.0) * x * x * x * x - y * y * y * y + nf(2.0) * y * y
    print(f"res={base.fmt(p2)}")


def example_2_quadratic() -> None:
    a = nf(0.3)
    b = nf(-2.1)
    c = nf(3.675)
    b = b / a
    c = c / a
    d = b * b - nf(4.0) * c
    print(f"d = {base.fmt(d)}")
    if bool(np.any(d.is_numerical_zero())):
        x1 = -b * ts(0.5)
        print(f"Discriminant is zero. The {solution_label()} solution is {base.fmt(x1)}")
    elif d > 0.0:
        x1 = (-b - sqrt(d)) * ts(0.5)
        x2 = (-b + sqrt(d)) * ts(0.5)
        print(f"Two real solutions: x1={base.fmt(x1)}, x2={base.fmt(x2)}")
    else:
        x1 = -b * ts(0.5)
        x2 = sqrt(-d) * ts(0.5)
        print(f"Two complex solutions: {base.fmt(x1)} +/- i*{base.fmt(x2)}")


def example_3_hilbert_determinant() -> None:
    n = 11
    a = [[nf(1.0 / (i + j + 1.0)) for j in range(n)] for i in range(n)]
    det = nf(1.0)
    for i in range(n - 1):
        print(f"Pivot number {i + 1:2d} = {base.fmt(a[i][i])}")
        det = det * a[i][i]
        aux = 1.0 / a[i][i]
        for j in range(i + 1, n):
            a[i][j] = a[i][j] * aux
        for j in range(i + 1, n):
            aux = a[j][i]
            for k in range(i + 1, n):
                a[j][k] = a[j][k] - aux * a[i][k]
    print(f"Pivot number {n:2d} = {base.fmt(a[n - 1][n - 1])}")
    det = det * a[n - 1][n - 1]
    print(f"Determinant = {base.fmt(det)}")


def example_4_muller_recurrence() -> None:
    a = nf(5.5)
    b = nf(61.0 / 11.0)
    for i in range(3, 31):
        c = b
        b = nf(111.0) - nf(1130.0) / b + nf(3000.0) / (a * b)
        a = c
        print(f"U({i:2d}) = {base.fmt(b)}")
    print("The true limit is 6.")


def example_5_newton() -> None:
    eps = 1.0e-12
    y = nf(0.5)
    for i in range(1, 101):
        x = y
        numerator = nf(1.47) * x**3 + nf(1.19) * x**2 - nf(1.83) * x + nf(0.45)
        denominator = nf(4.41) * x**2 + nf(2.38) * x - nf(1.83)
        y = x - numerator / denominator
        step = abs(x - y)
        print(f"x({i:3d}) = {base.fmt(y)}, diff = {base.fmt(step)}")
        if step < eps:
            break


def example_6_gaussian_pivoting() -> None:
    a = [
        [21.0, 130.0, 0.0, 2.1, 153.1],
        [13.0, 80.0, 4.74e8, 752.0, 849.74],
        [0.0, -0.4, 3.9816e8, 4.2, 7.7816],
        [0.0, 0.0, 1.7, 9.0e-9, 2.6e-8],
    ]
    xsol = [1.0, 1.0, 1.0e-8, 1.0]
    amat = [[nf(value) for value in row] for row in a]
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
        print(f"x_sol({i + 1}) = {base.fmt(amat[i][n])}  (true value: {xsol[i]:.7E})")


def example_7_jacobi_fast() -> None:
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
    rand = base.random1_state()
    a_data = torch.tensor(
        [[rand() for _ in range(ndim)] for _ in range(ndim)],
        dtype=torch.float64,
        device=CURRENT_DEVICE,
    )
    a_data.diagonal().add_(4.500002)
    xsol_data = torch.tensor(xsol_values, dtype=torch.float64, device=CURRENT_DEVICE)
    a = NFloat(a_data)
    xsol = NFloat(xsol_data)
    b = matmul(a, xsol)
    y = NFloat(torch.full((ndim,), 10.0, dtype=torch.float64, device=CURRENT_DEVICE))

    offdiag_samples = a.samples.clone()
    diag_samples = torch.diagonal(offdiag_samples, dim1=1, dim2=2).clone()
    diag_index = torch.arange(ndim, device=CURRENT_DEVICE)
    offdiag_samples[:, diag_index, diag_index] = 0.0
    offdiag = nf_from_samples(offdiag_samples)
    diag = nf_from_samples(diag_samples)

    iterations = min(niter, 38)
    for _iteration in range(1, iterations + 1):
        x = y
        y = (b - matmul(offdiag, x)) / diag

    print(f"niter = {iterations}  (verification cap: {JACOBI_MAX_ITERATIONS})")
    residues = matmul(a, y) - b
    for i in range(ndim):
        print(
            f"x_sol({i + 1:2d}) = {base.fmt(nf_component(y, i))}  "
            f"(true value: {xsol_values[i]:.7E}), residue({i + 1:2d}) = {base.fmt(nf_component(residues, i))}"
        )


def example_1_rump_deterministic_quiet() -> None:
    x = tdet(10864.0)
    y = tdet(18817.0)
    _ = tdet(9.0) * x * x * x * x - y * y * y * y + tdet(2.0) * y * y

    x = tdet(1.0 / 3.0)
    y = tdet(2.0 / 3.0)
    _ = tdet(9.0) * x * x * x * x - y * y * y * y + tdet(2.0) * y * y


def example_2_quadratic_deterministic_quiet() -> None:
    a = tdet(0.3)
    b = tdet(-2.1)
    c = tdet(3.675)
    b = b / a
    c = c / a
    d = b * b - tdet(4.0) * c
    if bool(d == 0.0):
        _ = -b * tdet(0.5)
    elif bool(d > 0.0):
        _ = (-b - torch.sqrt(d)) * tdet(0.5)
        _ = (-b + torch.sqrt(d)) * tdet(0.5)
    else:
        _ = -b * tdet(0.5)
        _ = torch.sqrt(-d) * tdet(0.5)


def example_3_hilbert_deterministic_quiet() -> None:
    n = 11
    a = [[tdet(1.0 / (i + j + 1.0)) for j in range(n)] for i in range(n)]
    det = tdet(1.0)
    for i in range(n - 1):
        det = det * a[i][i]
        aux = 1.0 / a[i][i]
        for j in range(i + 1, n):
            a[i][j] = a[i][j] * aux
        for j in range(i + 1, n):
            aux = a[j][i]
            for k in range(i + 1, n):
                a[j][k] = a[j][k] - aux * a[i][k]
    det = det * a[n - 1][n - 1]
    _ = det


def example_4_muller_deterministic_quiet() -> None:
    a = tdet(5.5)
    b = tdet(61.0 / 11.0)
    for _ in range(3, 31):
        c = b
        b = tdet(111.0) - tdet(1130.0) / b + tdet(3000.0) / (a * b)
        a = c


def example_5_newton_deterministic_quiet() -> None:
    eps = tdet(1.0e-12)
    y = tdet(0.5)
    for _ in range(1, 101):
        x = y
        numerator = tdet(1.47) * x**3 + tdet(1.19) * x**2 - tdet(1.83) * x + tdet(0.45)
        denominator = tdet(4.41) * x**2 + tdet(2.38) * x - tdet(1.83)
        y = x - numerator / denominator
        step = torch.abs(x - y)
        if bool(step < eps):
            break


def example_6_gaussian_deterministic_quiet() -> None:
    a = [
        [21.0, 130.0, 0.0, 2.1, 153.1],
        [13.0, 80.0, 4.74e8, 752.0, 849.74],
        [0.0, -0.4, 3.9816e8, 4.2, 7.7816],
        [0.0, 0.0, 1.7, 9.0e-9, 2.6e-8],
    ]
    amat = [[tdet(value) for value in row] for row in a]
    n = 4
    pivot_rows = [0, 1, 3]
    for i in range(n - 1):
        pivot = pivot_rows[i]
        if pivot != i:
            amat[i][i : n + 1], amat[pivot][i : n + 1] = (
                amat[pivot][i : n + 1],
                amat[i][i : n + 1],
            )
        aux = tdet(1.0) / amat[i][i]
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


def example_7_jacobi_deterministic_quiet_fast() -> None:
    ndim = 20
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
    rand = base.random1_state()
    a = torch.tensor(
        [[rand() for _ in range(ndim)] for _ in range(ndim)],
        dtype=deterministic_dtype(),
        device=CURRENT_DEVICE,
    )
    a.diagonal().add_(4.500002)
    xsol = torch.tensor(xsol_values, dtype=deterministic_dtype(), device=CURRENT_DEVICE)
    b = torch.matmul(a, xsol)
    y = torch.full((ndim,), 10.0, dtype=deterministic_dtype(), device=CURRENT_DEVICE)
    offdiag = a.clone()
    diag = torch.diagonal(offdiag).clone()
    diag_index = torch.arange(ndim, device=CURRENT_DEVICE)
    offdiag[diag_index, diag_index] = 0.0

    iterations = min(JACOBI_MAX_ITERATIONS, 38)
    for _iteration in range(1, iterations + 1):
        y = (b - torch.matmul(offdiag, y)) / diag


def example_7_jacobi_scalar() -> None:
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
    rand = base.random1_state()
    xsol = [nf(value) for value in xsol_values]
    a = [[nf(rand()) for _ in range(ndim)] for _ in range(ndim)]
    for i in range(ndim):
        a[i][i] = a[i][i] + ts(4.500002)
    b = []
    y = []
    for i in range(ndim):
        aux = nf(0.0)
        for j in range(ndim):
            aux = aux + a[i][j] * xsol[j]
        b.append(aux)
        y.append(nf(10.0))

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
            f"x_sol({i + 1:2d}) = {base.fmt(y[i])}  "
            f"(true value: {xsol_values[i]:.7E}), residue({i + 1:2d}) = {base.fmt(residue)}"
        )


def example_7_jacobi_deterministic_quiet_scalar() -> None:
    ndim = 20
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
    rand = base.random1_state()
    a = [[tdet(rand()) for _ in range(ndim)] for _ in range(ndim)]
    for i in range(ndim):
        a[i][i] = a[i][i] + tdet(4.500002)
    b = []
    y = []
    for i in range(ndim):
        aux = tdet(0.0)
        for j in range(ndim):
            aux = aux + a[i][j] * tdet(xsol_values[j])
        b.append(aux)
        y.append(tdet(10.0))

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
            diff = base.deterministic_abs((x[j] - y[j]).detach().cpu().item())
            if diff > anorm:
                anorm = diff


def example_7_jacobi() -> None:
    if FAST_JACOBI:
        example_7_jacobi_fast()
        return
    example_7_jacobi_scalar()


def example_7_jacobi_deterministic_quiet() -> None:
    if FAST_JACOBI:
        example_7_jacobi_deterministic_quiet_fast()
        return
    example_7_jacobi_deterministic_quiet_scalar()


REFERENCE_BY_NUMBER = {example.number: example for example in base.EXAMPLES}

EXAMPLES = (
    base.Example(
        1,
        REFERENCE_BY_NUMBER[1].title,
        REFERENCE_BY_NUMBER[1].cadna_references,
        example_1_rump,
        example_1_rump_deterministic_quiet,
    ),
    base.Example(
        2,
        REFERENCE_BY_NUMBER[2].title,
        REFERENCE_BY_NUMBER[2].cadna_references,
        example_2_quadratic,
        example_2_quadratic_deterministic_quiet,
    ),
    base.Example(
        3,
        REFERENCE_BY_NUMBER[3].title,
        REFERENCE_BY_NUMBER[3].cadna_references,
        example_3_hilbert_determinant,
        example_3_hilbert_deterministic_quiet,
    ),
    base.Example(
        4,
        REFERENCE_BY_NUMBER[4].title,
        REFERENCE_BY_NUMBER[4].cadna_references,
        example_4_muller_recurrence,
        example_4_muller_deterministic_quiet,
    ),
    base.Example(
        5,
        REFERENCE_BY_NUMBER[5].title,
        REFERENCE_BY_NUMBER[5].cadna_references,
        example_5_newton,
        example_5_newton_deterministic_quiet,
    ),
    base.Example(
        6,
        REFERENCE_BY_NUMBER[6].title,
        REFERENCE_BY_NUMBER[6].cadna_references,
        example_6_gaussian_pivoting,
        example_6_gaussian_deterministic_quiet,
    ),
    base.Example(
        7,
        REFERENCE_BY_NUMBER[7].title,
        REFERENCE_BY_NUMBER[7].cadna_references,
        example_7_jacobi,
        example_7_jacobi_deterministic_quiet,
    ),
)


def run_example(
    example: base.Example,
    precision: str,
    random_state: int | None,
    runtime_repeats: int,
    device: torch.device,
) -> dict[str, object]:
    reference = example.cadna_references[precision]
    configure_cadna_like(precision, random_state, device)
    print("=" * 80)
    print(f"Example {example.number}: {example.title}")
    print(f"Precision: {precision}")
    print(f"Backend: torch on {device}")
    print("-" * 80)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        example.run()
    print("-" * 80)
    summary = get_diagnostics_summary()
    detected = dict(summary.get("by_kind", {}))
    base.print_summary()
    base.print_reference_comparison(reference, detected)

    deterministic_runtime = measure_runtime(
        example.run_deterministic_quiet,
        runtime_repeats,
    )

    def run_nfloat_timed() -> None:
        configure_cadna_like(precision, random_state, device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_without_output(example.run)

    nfloat_runtime = measure_runtime(run_nfloat_timed, runtime_repeats)
    row = base.build_summary_row(
        example=example,
        precision=precision,
        detected=detected,
        reference=reference,
        deterministic_runtime=deterministic_runtime,
        nfloat_runtime=nfloat_runtime,
        runtime_repeats=runtime_repeats,
    )
    row["backend"] = "torch"
    row["device"] = str(device)
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
        "--device",
        default=None,
        help="torch device, for example cpu, cuda, or cuda:0",
    )
    parser.add_argument(
        "--fast-jacobi",
        action="store_true",
        help=(
            "use a tensorized Jacobi implementation for Example 7. This is much "
            "faster on torch/cuda but does not preserve the original scalar-level "
            "instability counting path."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cadna_examples_1_7_torch"),
        help="directory for CSV summary outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global FAST_JACOBI, JACOBI_MAX_ITERATIONS
    JACOBI_MAX_ITERATIONS = args.jacobi_max_iterations
    FAST_JACOBI = args.fast_jacobi
    device = resolve_device(args.device)
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
                        device,
                    )
                )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "cadna_examples_1_7_summary.csv"
    base.write_csv(rows, summary_csv)
    print("=" * 80)
    print("Compact noisefloat summary")
    for row in rows:
        print(
            f"[{row['precision']}] Example {row['example']}: "
            f"total={row['noisefloat_total_instabilities']} "
            f"by_kind={base.format_counts(base.example_counts(row, 'noisefloat'))} | "
            f"CADNA reference total={row['reference_total_instabilities']} "
            f"by_kind={base.format_counts(base.example_counts(row, 'reference'))}"
        )
    print(f"Wrote summary CSV to: {summary_csv}")


if __name__ == "__main__":
    main()
