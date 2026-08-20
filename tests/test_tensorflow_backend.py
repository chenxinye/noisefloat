from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import noisefloat as nf
from noisefloat import NFloat, configure


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
def reset():
    configure(
        backend="tensorflow",
        exp_bits=8,
        sig_bits=23,
        n_samples=3,
        random_state=42,
        confidence=0.95,
        trace=False,
        digits_threshold=0.5,
        zero_digits_threshold=0.5,
    )
    nf.reset_chopper_cache()
    yield


def test_nfloat_tensorflow_tensor_samples_preserve_backend(tf):
    x = NFloat(tf.constant([1.0, 2.0], dtype=tf.float64))

    assert x.backend_name == "tensorflow"
    assert isinstance(x.samples, tf.Tensor)
    assert tuple(x.samples.shape) == (3, 2)
    np.testing.assert_allclose(x.mean, [1.0, 2.0], rtol=1e-3)


def test_tensorflow_functions_dispatch_without_numpy_roundtrip(tf):
    x = NFloat(tf.constant([[1.0, 4.0]], dtype=tf.float64))
    y = nf.sqrt(x)
    total = nf.sum(y)

    assert y.backend_name == "tensorflow"
    assert total.backend_name == "tensorflow"
    assert isinstance(y.samples, tf.Tensor)
    assert np.asarray(total.mean).shape == ()


def test_tensorflow_matmul(tf):
    x = NFloat(tf.ones((2, 3), dtype=tf.float64))
    y = NFloat(tf.ones((3, 4), dtype=tf.float64))

    out = nf.matmul(x, y)

    assert out.backend_name == "tensorflow"
    assert isinstance(out.samples, tf.Tensor)
    assert out.shape == (2, 4)


def test_tensorflow_norm_preserves_backend(tf):
    x = NFloat(tf.constant([3.0, 4.0], dtype=tf.float64))

    out = nf.norm(x)

    assert out.backend_name == "tensorflow"
    assert isinstance(out.samples, tf.Tensor)
    assert abs(out.mean - 5.0) < 1e-3
