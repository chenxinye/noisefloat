from __future__ import annotations

import json
import subprocess
import sys

import pytest

from noisefloat.nn import TensorFlowNFloatTensor, clear_kernel_reports
from noisefloat.nn.report import record_kernel

@pytest.fixture(scope="module")
def tf():
    probe = subprocess.run(
        [sys.executable, "-c", "import tensorflow"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip(f"tensorflow not available: {probe.stderr.strip()}")
    try:
        import tensorflow as tensorflow
    except Exception as exc:
        pytest.skip(f"tensorflow not available: {exc}")
    return tensorflow


@pytest.fixture(autouse=True)
def reset_noisefloat(tf):
    from exps.common_tensorflow import configure_noisefloat_tensorflow

    configure_noisefloat_tensorflow(n_samples=3, random_state=42)
    clear_kernel_reports()
    yield
    clear_kernel_reports()


def test_tensorflow_tensor_helpers_preserve_nfloat_type(tf):
    from exps.common_tensorflow import (
        tensor_add,
        tensor_concat,
        tensor_flatten,
        tensor_permute,
        tensor_scale,
    )

    base = TensorFlowNFloatTensor(tf.random.normal((2, 3), dtype=tf.float64))
    other = tf.random.normal((2, 3), dtype=tf.float64)

    added = tensor_add(base, other)
    scaled = tensor_scale(base, 2.0)
    flattened = tensor_flatten(base, start_axis=1)
    concatenated = tensor_concat((base, other), axis=1)
    permuted = tensor_permute(
        TensorFlowNFloatTensor(tf.random.normal((2, 3, 4), dtype=tf.float64)),
        (1, 0, 2),
    )

    assert isinstance(added, TensorFlowNFloatTensor)
    assert isinstance(scaled, TensorFlowNFloatTensor)
    assert isinstance(flattened, TensorFlowNFloatTensor)
    assert isinstance(concatenated, TensorFlowNFloatTensor)
    assert isinstance(permuted, TensorFlowNFloatTensor)
    assert concatenated.shape == (2, 6)
    assert flattened.shape == (2, 3)
    assert permuted.shape == (3, 2, 4)


def test_tensorflow_tracker_exports_metadata(tf, tmp_path):
    from exps.common_tensorflow import KernelDigitTracker

    record_kernel(
        "tf_demo_kernel",
        "forward",
        TensorFlowNFloatTensor(tf.ones((2, 2), dtype=tf.float64)),
    )
    tracker = KernelDigitTracker("tf_demo_task")
    tracker.capture(
        epoch=1,
        iteration=2,
        split="train",
        global_iteration=5,
        metadata={"task_seed": 42},
    )
    tracker.export(tmp_path)

    rows = (tmp_path / "kernel_digits.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["epoch"] == 1
    assert row["iteration"] == 2
    assert row["details"]["task_seed"] == 42
    assert (tmp_path / "kernel_digits.csv").exists()
    assert (tmp_path / "summary.json").exists()
