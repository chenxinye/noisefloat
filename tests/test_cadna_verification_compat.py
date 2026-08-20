from pathlib import Path

import pytest

from verifications import cadna_examples_1_7 as cadna_examples
from verifications import plot_cadna_examples_runtime as runtime_plot


def test_examples_include_single_and_double_cadna_references():
    for example in cadna_examples.EXAMPLES:
        for precision in cadna_examples.PRECISION_ORDER:
            reference = example.cadna_references[precision]
            assert isinstance(reference.counts, dict)
            assert reference.source


def test_run_example_writes_runtime_fields_and_csv(tmp_path):
    row = cadna_examples.run_example(
        cadna_examples.EXAMPLES[0],
        precision="single",
        random_state=12,
        runtime_repeats=1,
    )

    assert row["example"] == 1
    assert row["precision"] == "single"
    assert row["runtime_repeats"] == 1
    assert row["deterministic_double_runtime_mean_seconds"] >= 0.0
    assert row["noisefloat_runtime_mean_seconds"] >= 0.0
    assert "runtime_overhead_ratio" in row
    assert row["noisefloat_sources"]
    assert row["reference_sources"]

    output_csv = tmp_path / "cadna_examples_1_7_summary.csv"
    cadna_examples.write_csv([row], output_csv)
    assert output_csv.exists()


def test_runtime_plot_smoke(tmp_path):
    pytest.importorskip("matplotlib", reason="matplotlib not installed")

    rows = [
        {
            "example": "1",
            "precision": "single",
            "deterministic_double_runtime_mean_seconds": "0.0010",
            "noisefloat_runtime_mean_seconds": "0.0030",
            "runtime_overhead_ratio": "3.0",
        },
        {
            "example": "2",
            "precision": "double",
            "deterministic_double_runtime_mean_seconds": "0.0020",
            "noisefloat_runtime_mean_seconds": "0.0060",
            "runtime_overhead_ratio": "3.0",
        },
    ]
    output_stem = tmp_path / "cadna_examples_runtime_ratio"
    runtime_plot.plot_runtime_comparison(
        rows,
        output_stem,
        config=runtime_plot.PlotConfig(font_size=10, dpi=120),
        formats=("jpg",),
        log_scale=False,
    )
    assert Path(f"{output_stem}.jpg").exists()
