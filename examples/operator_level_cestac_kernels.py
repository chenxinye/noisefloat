"""Operator-level CESTAC benchmark kernels.

This example mirrors the deep-learning instrumentation strategy used by
``noisefloat.nn``: each backend operator is evaluated independently on the
stochastic samples, then random rounding is applied once to the operator
output and a kernel report is recorded.

The kernels below are small, deterministic references that are commonly used
or closely related to deep-learning operator stability discussions:

* Naive logsumexp/softmax overflow contrasts with stable shifted forms.
* LayerNorm on near-constant activations exposes small-denominator sensitivity.
* Near-tie attention logits expose decision/weight sensitivity in attention.
"""

from __future__ import annotations

import argparse
import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import torch

from noisefloat import configure, reset_chopper_cache
from noisefloat.nn import NFloatTensor, clear_kernel_reports, get_kernel_reports
from noisefloat.nn.context import round_samples
from noisefloat.nn.report import record_kernel


@dataclass(frozen=True)
class KernelCase:
    name: str
    category: str
    expected: str
    notes: str
    function: Callable[[], torch.Tensor]


DIAGNOSTIC_KINDS = {
    "loss_of_accuracy_due_to_cancellation": "cancellation_count",
    "branching_instability": "branching_count",
    "mathematical_instability": "mathematical_count",
    "intrinsic_instability": "intrinsic_count",
}


def make_nfloat(value, *, n_samples: int | None = None) -> NFloatTensor:
    return NFloatTensor(torch.as_tensor(value, dtype=torch.float64), n_samples=n_samples)


def record_operator(name: str, samples: torch.Tensor) -> NFloatTensor:
    result = NFloatTensor.from_samples(round_samples(samples))
    record_kernel(name, "forward", result)
    return result


def _diagnostics_from_source(source: str, *, scope: str, count: int = 1) -> dict[str, object]:
    columns = {
        "diagnostic_scope": scope,
        "instability_count": int(count if source else 0),
        "source_of_accuracy_loss": source,
        "source_summary": f"{source}:{count}" if source else "",
    }
    for kind, column in DIAGNOSTIC_KINDS.items():
        columns[column] = int(count if source == kind else 0)
    return columns


def infer_operator_source(case: KernelCase, output: NFloatTensor, is_stable: bool) -> dict[str, object]:
    samples = output.samples.detach()
    has_nonfinite = bool(torch.logical_not(torch.isfinite(samples)).any().item())
    name = case.name.lower()
    notes = case.notes.lower()
    source = ""
    if has_nonfinite or "overflow" in name or "overflow" in notes:
        source = "mathematical_instability"
    elif not is_stable:
        if any(token in name or token in notes for token in ("cancellation", "rump", "difference")):
            source = "loss_of_accuracy_due_to_cancellation"
        else:
            source = "intrinsic_instability"
    return _diagnostics_from_source(source, scope="operator_level_inferred")


def shifted_softmax(samples: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = samples - samples.max(dim=dim, keepdim=True).values
    exp_values = torch.exp(shifted)
    return exp_values / exp_values.sum(dim=dim, keepdim=True)


def naive_softmax(samples: torch.Tensor, dim: int = -1) -> torch.Tensor:
    exp_values = torch.exp(samples)
    return exp_values / exp_values.sum(dim=dim, keepdim=True)


def stable_logsumexp(samples: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shift = samples.max(dim=dim, keepdim=True).values
    return shift.squeeze(dim) + torch.log(torch.exp(samples - shift).sum(dim=dim))


def naive_logsumexp(samples: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.log(torch.exp(samples).sum(dim=dim))


def layer_norm(samples: torch.Tensor, eps: float) -> torch.Tensor:
    mean = samples.mean(dim=-1, keepdim=True)
    variance = (samples - mean).pow(2).mean(dim=-1, keepdim=True)
    return (samples - mean) / torch.sqrt(variance + eps)


def attention(samples: torch.Tensor, *, near_tie: bool) -> torch.Tensor:
    # Shape: (samples, batch, seq, dim)
    query = samples[:, :, :1, :]
    if near_tie:
        key = samples
        value = torch.tensor(
            [[[[1.0, -1.0, 0.5, -0.5], [-1.0, 1.0, -0.5, 0.5]]]],
            dtype=torch.float64,
            device=samples.device,
        ).expand(samples.shape[0], samples.shape[1], -1, -1)
    else:
        key = samples
        value = torch.tensor(
            [[[[1.0, 0.25, -0.25, 0.0], [0.0, -0.25, 0.25, 1.0]]]],
            dtype=torch.float64,
            device=samples.device,
        ).expand(samples.shape[0], samples.shape[1], -1, -1)
    scores = query @ key.transpose(-2, -1) / (samples.shape[-1] ** 0.5)
    weights = shifted_softmax(scores, dim=-1)
    return weights @ value


def build_cases() -> List[KernelCase]:
    rng = torch.Generator().manual_seed(2026)

    logits_moderate = make_nfloat(
        torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, -1.0, 0.5, 1.0]])
    )
    logits_large = make_nfloat(
        torch.tensor([[1000.0, 1001.0, 1002.0, 1003.0], [1002.0, 999.0, 998.5, 997.0]])
    )

    normal_activations = make_nfloat(torch.randn(16, 32, generator=rng))
    near_constant = make_nfloat(
        torch.ones(16, 32, dtype=torch.float64)
        + 2.0 ** -20 * torch.randn(16, 32, generator=rng)
    )

    separated_attention_input = make_nfloat(
        torch.tensor([[[2.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0]]])
    )
    near_tie_samples = torch.tensor(
        [
            [[[1.0, 1.0, 1.0, 1.0], [1.0 - 2.0**-40, 1.0, 1.0, 1.0]]],
            [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            [[[1.0, 1.0, 1.0, 1.0], [1.0 + 2.0**-40, 1.0, 1.0, 1.0]]],
        ],
        dtype=torch.float64,
    )
    near_tie_attention_input = NFloatTensor.from_samples(near_tie_samples)

    return [
        KernelCase(
            "stable_logsumexp",
            "softmax",
            "stable",
            "Shifted logsumexp avoids overflow.",
            lambda: stable_logsumexp(logits_large.samples, dim=-1),
        ),
        KernelCase(
            "naive_logsumexp_overflow",
            "softmax",
            "unstable",
            "Naive exp-sum-log overflows for large logits.",
            lambda: naive_logsumexp(logits_large.samples, dim=-1),
        ),
        KernelCase(
            "stable_softmax_shifted",
            "softmax",
            "stable",
            "Shifted softmax on moderate logits.",
            lambda: shifted_softmax(logits_moderate.samples, dim=-1),
        ),
        KernelCase(
            "naive_softmax_overflow",
            "softmax",
            "unstable",
            "Naive softmax overflows for large logits.",
            lambda: naive_softmax(logits_large.samples, dim=-1),
        ),
        KernelCase(
            "layernorm_normal_variance",
            "normalization",
            "stable",
            "LayerNorm on ordinary activations.",
            lambda: layer_norm(normal_activations.samples, eps=1.0e-5),
        ),
        KernelCase(
            "layernorm_near_constant",
            "normalization",
            "unstable",
            "LayerNorm with near-zero variance activations.",
            lambda: layer_norm(near_constant.samples, eps=1.0e-12),
        ),
        KernelCase(
            "attention_separated_logits",
            "attention",
            "stable",
            "Attention with clearly separated logits.",
            lambda: attention(separated_attention_input.samples, near_tie=False),
        ),
        KernelCase(
            "attention_near_tie_logits",
            "attention",
            "unstable",
            "Attention with nearly tied logits and opposing values.",
            lambda: attention(near_tie_attention_input.samples, near_tie=True),
        ),
    ]


def run_benchmark(exp_bits: int, sig_bits: int, seed: int) -> List[dict[str, object]]:
    configure(
        backend="torch",
        n_samples=3,
        random_state=seed,
        exp_bits=exp_bits,
        sig_bits=sig_bits,
        confidence=0.95,
    )
    reset_chopper_cache()
    torch.manual_seed(seed)
    clear_kernel_reports()

    rows = []
    for case in build_cases():
        output = record_operator(case.name, case.function())
        report = get_kernel_reports()[-1]
        mean_abs = output.value.detach().abs().mean().item()
        std_mean = output.samples.detach().std(dim=0, unbiased=True).mean().item()
        diagnostics = infer_operator_source(case, output, bool(report.is_stable))
        rows.append(
            {
                "suite": "operator_level_cestac_kernels",
                "precision": f"exp{exp_bits}_sig{sig_bits}",
                "expected_behavior": case.expected,
                "category": case.category,
                "kernel": case.name,
                "variant": "",
                "input_label": "",
                "avg_digits": report.avg_digits,
                "min_digits": report.min_digits,
                "max_digits": report.max_digits,
                "mean_abs": mean_abs,
                "std_mean": std_mean,
                "num_elements": report.num_elements,
                "is_stable": report.is_stable,
                **diagnostics,
                "notes": case.notes,
            }
        )

    print(f"\nOperator-level CESTAC benchmark: exp_bits={exp_bits}, sig_bits={sig_bits}, seed={seed}")
    print("-" * 124)
    print(
        f"{'expected':<9} {'category':<14} {'kernel':<30} "
        f"{'avg_digits':>10} {'min_digits':>10} {'max_digits':>10} "
        f"{'mean_abs':>12} {'std_mean':>12}"
    )
    print("-" * 124)
    for row in rows:
        print(
            f"{str(row['expected_behavior']):<9} {str(row['category']):<14} "
            f"{str(row['kernel']):<30} "
            f"{float(row['avg_digits']):10.3f} {float(row['min_digits']):10.3f} "
            f"{float(row['max_digits']):10.3f} {float(row['mean_abs']):12.3e} "
            f"{float(row['std_mean']):12.3e}"
        )
    return rows


def write_csv(rows: List[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--precision",
        choices=("float32", "float64", "both"),
        default="both",
        help="Software rounding format to emulate.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("examples/outputs/operator_level_cestac_kernels.csv"),
        help="CSV path for benchmark rows.",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    rows: List[dict[str, object]] = []
    if args.precision in {"float32", "both"}:
        rows.extend(run_benchmark(exp_bits=8, sig_bits=23, seed=args.seed))
    if args.precision in {"float64", "both"}:
        rows.extend(run_benchmark(exp_bits=11, sig_bits=52, seed=args.seed))
    write_csv(rows, args.output_csv)
    print(f"\nwrote CSV to: {args.output_csv}")


if __name__ == "__main__":
    main()
