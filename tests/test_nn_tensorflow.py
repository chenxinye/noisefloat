from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from noisefloat import configure, reset_chopper_cache
from noisefloat.nn import (
    NFloatLinear,
    NFloatReLU,
    NFloatTensor,
    TensorFlowNFloatLinear,
    TensorFlowNFloatDense,
    TensorFlowNFloatModule,
    TensorFlowNFloatReLU,
    TensorFlowNFloatSoftmax,
    TensorFlowNFloatTensor,
    nfloat_analysis,
    nfloat_operator,
    clear_kernel_reports,
    get_kernel_reports,
    wrap_keras,
)
from noisefloat.nn.context import maybe_nfloat_tensor


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
        kernel_report_detail="summary",
    )
    reset_chopper_cache()
    clear_kernel_reports()
    yield
    clear_kernel_reports()


def test_tensorflow_nfloat_tensor_creation(tf):
    ct = TensorFlowNFloatTensor(tf.ones((2, 3), dtype=tf.float64))

    assert ct.n_samples == 3
    assert ct.shape == (2, 3)
    assert isinstance(ct.samples, tf.Tensor)
    np.testing.assert_allclose(ct.mean, np.ones((2, 3)), rtol=1e-3)
    assert isinstance(ct.value, tf.Tensor)
    np.testing.assert_allclose(ct.to_tensor().numpy(), np.ones((2, 3)), rtol=1e-3)
    assert ct.used_ste is False


def test_backend_generic_tensorflow_tensor_preserves_ste_flag(tf):
    ct = NFloatTensor(tf.ones((2, 3), dtype=tf.float64), ste=True)

    assert isinstance(ct, TensorFlowNFloatTensor)
    assert ct.used_ste is True


def test_tensorflow_auto_convert_preserves_analysis_ste_flag(tf):
    plain = tf.ones((2, 3), dtype=tf.float64)

    with nfloat_analysis(enabled=True, use_ste=True):
        ct = maybe_nfloat_tensor(plain)

    assert isinstance(ct, TensorFlowNFloatTensor)
    assert ct.used_ste is True


def test_tensorflow_tensor_native_digits_work_in_tf_function(tf):
    samples = tf.ones((3, 2, 3), dtype=tf.float64)

    @tf.function
    def digits_for(samples):
        return TensorFlowNFloatTensor.from_samples(samples).digits_tensor()

    digits = digits_for(samples)

    assert isinstance(digits, tf.Tensor)
    np.testing.assert_allclose(digits.numpy(), np.full((2, 3), 15.0))


def test_tensorflow_dense_wrapper_records_report(tf):
    layer = TensorFlowNFloatDense(4)
    x = TensorFlowNFloatTensor(tf.ones((2, 3), dtype=tf.float64))

    out = layer(x)

    assert layer.get_config()["vectorized"] is True
    assert isinstance(out, TensorFlowNFloatTensor)
    assert out.shape == (2, 4)
    reports = get_kernel_reports()
    assert reports[-1].kernel_name == "nfloat/Dense"
    assert reports[-1].details["backend"] == "tensorflow"
    assert reports[-1].details["n_samples"] == 3
    assert reports[-1].details["detail_level"] == "summary"
    assert tuple(reports[-1].details["samples_shape"]) == (3, 2, 4)
    assert tuple(reports[-1].details["representative_shape"]) == (2, 4)
    assert tuple(reports[-1].details["digits_shape"]) == (2, 4)
    assert "samples" not in reports[-1].details
    assert "representative_value" not in reports[-1].details
    assert "digits" not in reports[-1].details


def test_backend_generic_layers_dispatch_to_tensorflow(tf):
    model = [
        NFloatLinear(3, 5),
        NFloatReLU(),
        NFloatLinear(5, 2),
    ]
    x = NFloatTensor(tf.ones((4, 3), dtype=tf.float64))

    out = x
    for layer in model:
        out = layer(out)

    assert isinstance(model[0], TensorFlowNFloatLinear)
    assert isinstance(out, TensorFlowNFloatTensor)
    assert out.shape == (4, 2)
    reports = get_kernel_reports()
    assert [report.kernel_name for report in reports] == [
        "nfloat/Linear",
        "nfloat/ReLU",
        "nfloat/Linear",
    ]
    assert all(report.details["backend"] == "tensorflow" for report in reports)


def test_tensorflow_custom_operator_records_nfloat_samples(tf):
    op = nfloat_operator(tf.matmul, name="tf_matmul")
    lhs = TensorFlowNFloatTensor(tf.ones((2, 3), dtype=tf.float64))
    rhs = tf.ones((3, 4), dtype=tf.float64)

    out = op(lhs, rhs)

    assert isinstance(out, TensorFlowNFloatTensor)
    assert out.shape == (2, 4)
    report = get_kernel_reports()[-1]
    assert report.kernel_name == "nfloat/tf_matmul"
    assert report.details["backend"] == "tensorflow"
    assert tuple(report.details["samples_shape"]) == (3, 2, 4)


def test_tensorflow_full_report_details_are_opt_in(tf):
    configure(
        backend="tensorflow",
        n_samples=3,
        random_state=42,
        kernel_report_detail="full",
    )
    layer = TensorFlowNFloatDense(4)
    x = TensorFlowNFloatTensor(tf.ones((2, 3), dtype=tf.float64))

    _ = layer(x)

    report = get_kernel_reports()[-1]
    assert report.details["detail_level"] == "full"
    assert tuple(report.details["samples"].shape) == (3, 2, 4)
    assert tuple(report.details["representative_value"].shape) == (2, 4)
    assert report.details["digits"].shape == (2, 4)


def test_tensorflow_activation_chain(tf):
    dense = TensorFlowNFloatDense(5)
    relu = TensorFlowNFloatReLU()
    softmax = TensorFlowNFloatSoftmax(axis=-1)
    x = TensorFlowNFloatTensor(tf.random.normal((4, 8), dtype=tf.float64))

    out = softmax(relu(dense(x)))

    assert isinstance(out, TensorFlowNFloatTensor)
    assert out.shape == (4, 5)


def test_wrap_keras_passthrough_and_nfloat_path(tf):
    wrapped = wrap_keras(
        tf.keras.layers.Activation("tanh", dtype="float64"), name="tanh"
    )
    plain = tf.constant([[1.0]], dtype=tf.float64)
    nfloat = TensorFlowNFloatTensor(plain)

    plain_out = wrapped(plain)
    nfloat_out = wrapped(nfloat)

    assert isinstance(wrapped, TensorFlowNFloatModule)
    assert isinstance(plain_out, tf.Tensor)
    assert isinstance(nfloat_out, TensorFlowNFloatTensor)
