from __future__ import annotations

import argparse

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from noisyfloat.exps.full_cestac.arithmetic_cestac_operator_instability import (
    aggregate_summary,
    build_cases,
    detection_metrics,
    run_benchmark,
)


def test_arithmetic_cestac_operator_instability_benchmark_separates_cases():
    args = argparse.Namespace(
        trials=1,
        seed=2026,
        n_samples=3,
        confidence=0.95,
        threshold_digits=None,
    )

    rows = run_benchmark(args, exp_bits=8, sig_bits=23)
    summary = aggregate_summary(rows)
    metrics = detection_metrics(rows)

    assert len(rows) == len(build_cases())
    assert len(summary) == len(build_cases())
    assert {row["threshold_digits"] for row in rows} == {3.0}
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
