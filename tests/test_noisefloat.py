"""
Tests for noisefloat – NumPy backend (runs by default).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import noisefloat as nf
from noisefloat import (
    NFloat,
    configure,
    get_config,
    is_unstable_comparison,
    clear_diagnostics,
    get_diagnostics,
    get_diagnostics_summary,
    print_diagnostics,
    print_diagnostics_summary,
)
from noisefloat import sqrt, exp, log, sin, cos, dot, matmul
from noisefloat import sum as nf_sum, mean as nf_mean, norm as nf_norm
from noisefloat.exceptions import (
    UnstableComparisonWarning,
    NumericalZeroWarning,
    UnstableOperationWarning,
)


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def reset_config():
    """Reset noisefloat config to defaults before each test."""
    configure(
        backend="numpy",
        exp_bits=8,
        sig_bits=23,
        n_samples=3,
        random_state=42,
        confidence=0.95,
        trace=False,
        digits_threshold=0.5,
        zero_digits_threshold=0.5,
        diagnostics_level="summary",
        kernel_report_detail="summary",
        cancellation_ratio_threshold=1e-6,
        cancellation_digits_loss=4.0,
    )
    clear_diagnostics()
    # Reset chopper to pick up new config
    from noisefloat.core import _chopper

    _chopper._config_hash = ()
    yield


# --------------------------------------------------------------------------- #
#  1. Scalar arithmetic                                                        #
# --------------------------------------------------------------------------- #


class TestScalarArithmetic:
    def test_add(self):
        a = NFloat(1.0)
        b = NFloat(2.0)
        c = a + b
        assert abs(c.mean - 3.0) < 1e-4, f"Expected ~3, got {c.mean}"

    def test_sub(self):
        a = NFloat(5.0)
        b = NFloat(2.0)
        c = a - b
        assert abs(c.mean - 3.0) < 1e-4

    def test_mul(self):
        a = NFloat(3.0)
        b = NFloat(4.0)
        c = a * b
        assert abs(c.mean - 12.0) < 1e-3

    def test_div(self):
        a = NFloat(6.0)
        b = NFloat(3.0)
        c = a / b
        assert abs(c.mean - 2.0) < 1e-4

    def test_pow(self):
        a = NFloat(2.0)
        b = NFloat(3.0)
        c = a**b
        assert abs(c.mean - 8.0) < 1e-2

    def test_neg(self):
        a = NFloat(3.5)
        b = -a
        assert abs(b.mean - (-3.5)) < 1e-4

    def test_abs(self):
        a = NFloat(-2.5)
        b = abs(a)
        assert abs(b.mean - 2.5) < 1e-4

    def test_radd(self):
        a = NFloat(1.5)
        c = 2.0 + a
        assert abs(c.mean - 3.5) < 1e-4

    def test_rsub(self):
        a = NFloat(1.5)
        c = 5.0 - a
        assert abs(c.mean - 3.5) < 1e-4

    def test_rmul(self):
        a = NFloat(4.0)
        c = 3.0 * a
        assert abs(c.mean - 12.0) < 1e-3

    def test_rtruediv(self):
        a = NFloat(2.0)
        c = 6.0 / a
        assert abs(c.mean - 3.0) < 1e-4

    def test_samples_shape_scalar(self):
        a = NFloat(1.0, n_samples=5)
        assert a.samples.shape == (5,), a.samples.shape

    def test_float_conversion(self):
        a = NFloat(2.718)
        f = float(a)
        assert abs(f - 2.718) < 1e-3


# --------------------------------------------------------------------------- #
#  2. Array arithmetic and broadcasting                                         #
# --------------------------------------------------------------------------- #


class TestArrayArithmetic:
    def test_array_add(self):
        a = NFloat(np.array([1.0, 2.0, 3.0]))
        b = NFloat(np.array([0.5, 0.5, 0.5]))
        c = a + b
        expected = np.array([1.5, 2.5, 3.5])
        np.testing.assert_allclose(c.mean, expected, rtol=1e-4)

    def test_array_mul(self):
        a = NFloat(np.array([1.0, 2.0, 3.0]))
        c = a * 2.0
        np.testing.assert_allclose(c.mean, [2.0, 4.0, 6.0], rtol=1e-4)

    def test_array_shape(self):
        a = NFloat(np.ones((4, 5)))
        assert a.shape == (4, 5)
        assert a.samples.shape == (3, 4, 5)

    def test_array_broadcasting(self):
        a = NFloat(np.array([1.0, 2.0, 3.0]))
        b = NFloat(1.0)
        c = a + b
        assert c.shape == (3,)
        np.testing.assert_allclose(c.mean, [2.0, 3.0, 4.0], rtol=1e-4)

    def test_matrix_matmul(self):
        A = NFloat(np.eye(3))
        x = NFloat(np.array([1.0, 2.0, 3.0]))
        y = matmul(A, x)
        np.testing.assert_allclose(y.mean, [1.0, 2.0, 3.0], rtol=1e-4)

    def test_dot_product(self):
        x = NFloat(np.array([1.0, 2.0, 3.0]))
        y = NFloat(np.array([1.0, 1.0, 1.0]))
        d = dot(x, y)
        assert abs(d.mean - 6.0) < 1e-3

    def test_sum_reduction(self):
        a = NFloat(np.array([1.0, 2.0, 3.0]))
        s = nf_sum(a)
        assert abs(s.mean - 6.0) < 1e-3

    def test_mean_reduction(self):
        a = NFloat(np.array([1.0, 2.0, 3.0]))
        m = nf_mean(a)
        assert abs(m.mean - 2.0) < 1e-3

    def test_norm(self):
        a = NFloat(np.array([3.0, 4.0]))
        n = nf_norm(a)
        assert abs(n.mean - 5.0) < 1e-3


# --------------------------------------------------------------------------- #
#  3. Reproducibility                                                          #
# --------------------------------------------------------------------------- #


class TestReproducibility:
    def test_same_seed_same_result(self):
        configure(random_state=123)
        from noisefloat.core import _chopper

        _chopper._config_hash = ()

        a1 = NFloat(3.14)
        s1 = a1.samples.copy()

        configure(random_state=123)
        _chopper._config_hash = ()
        a2 = NFloat(3.14)
        s2 = a2.samples.copy()

        np.testing.assert_array_equal(s1, s2)

    def test_different_seed_different_result(self):
        configure(random_state=0)
        from noisefloat.core import _chopper

        _chopper._config_hash = ()
        a1 = NFloat(3.14)
        s1 = a1.samples.copy()

        configure(random_state=999)
        _chopper._config_hash = ()
        a2 = NFloat(3.14)
        s2 = a2.samples.copy()

        # Very high probability they differ (stochastic rounding)
        # Note: occasionally they could match by chance with n_samples=3
        # We use n_samples=10 to reduce this risk
        configure(n_samples=10, random_state=0)
        _chopper._config_hash = ()
        a3 = NFloat(3.14)
        s3 = a3.samples.copy()

        configure(n_samples=10, random_state=999)
        _chopper._config_hash = ()
        a4 = NFloat(3.14)
        s4 = a4.samples.copy()

        # With 10 samples and different seeds, samples should differ
        # (probability of all 10 matching by chance is negligible)
        assert not np.array_equal(s3, s4)


# --------------------------------------------------------------------------- #
#  3b. Backend-native stochastic rounding                                      #
# --------------------------------------------------------------------------- #


class TestBackendNativeRounding:
    def test_numpy_rounding_is_vectorized_and_reproducible(self):
        configure(backend="numpy", exp_bits=5, sig_bits=3, n_samples=4, random_state=7)
        nf.reset_chopper_cache()
        from noisefloat.core import _chop_batch

        samples = np.broadcast_to(np.linspace(-3.0, 3.0, 17), (4, 17)).copy()
        rounded1 = _chop_batch(samples)

        nf.reset_chopper_cache()
        rounded2 = _chop_batch(samples)

        assert isinstance(rounded1, np.ndarray)
        assert rounded1.shape == (4, 17)
        np.testing.assert_array_equal(rounded1, rounded2)

    def test_numpy_random_directed_rounding_perturbs_nearest_grid_values(self):
        configure(backend="numpy", exp_bits=8, sig_bits=23, n_samples=4, random_state=7)
        nf.reset_chopper_cache()
        from noisefloat.core import _chop_batch

        samples = np.broadcast_to(np.array([1.0, 2.0, -1.0, -2.0]), (4, 4)).copy()
        rounded = _chop_batch(samples)

        assert rounded.shape == samples.shape
        assert not np.array_equal(rounded, samples)
        np.testing.assert_allclose(rounded, samples, rtol=2e-7, atol=0.0)

    def test_numpy_three_samples_use_random_directed_rounding(self):
        configure(backend="numpy", exp_bits=8, sig_bits=23, n_samples=3, random_state=7)
        nf.reset_chopper_cache()
        from noisefloat.core import _chop_batch

        spacing = 2.0**-23
        samples = np.full((3, 8), 1.0 + 0.25 * spacing, dtype=np.float64)
        rounded1 = _chop_batch(samples)

        nf.reset_chopper_cache()
        rounded2 = _chop_batch(samples)

        assert rounded1.shape == samples.shape
        np.testing.assert_array_equal(rounded1, rounded2)
        assert set(np.unique(rounded1)).issubset({1.0, 1.0 + spacing})
        assert np.unique(rounded1).size == 2

    def test_numpy_three_samples_do_not_use_fixed_nearest_up_down_tracks(self):
        configure(backend="numpy", exp_bits=8, sig_bits=23, n_samples=3, random_state=7)
        nf.reset_chopper_cache()
        from noisefloat.core import _chop_batch

        spacing = 2.0**-23
        rounded = _chop_batch(np.ones((3, 8), dtype=np.float64))

        assert set(np.unique(rounded)).issubset({1.0 - spacing, 1.0 + spacing})
        assert not np.any(rounded == 1.0)
        assert not np.all(rounded[0] == 1.0)

    def test_torch_rounding_is_vectorized_native_and_reproducible(self):
        configure(backend="torch", exp_bits=5, sig_bits=3, n_samples=4, random_state=7)
        nf.reset_chopper_cache()
        from noisefloat.core import _chop_batch

        base = torch.linspace(-3.0, 3.0, 17, dtype=torch.float64)
        samples = base.expand(4, -1).clone()
        rounded1 = _chop_batch(samples)

        nf.reset_chopper_cache()
        rounded2 = _chop_batch(samples)

        assert isinstance(rounded1, torch.Tensor)
        assert rounded1.device == samples.device
        assert rounded1.shape == (4, 17)
        np.testing.assert_array_equal(rounded1.numpy(), rounded2.numpy())

    def test_torch_random_directed_rounding_perturbs_nearest_grid_values(self):
        configure(backend="torch", exp_bits=8, sig_bits=23, n_samples=4, random_state=7)
        nf.reset_chopper_cache()
        from noisefloat.core import _chop_batch

        samples = torch.tensor([[1.0, 2.0, -1.0, -2.0]] * 4, dtype=torch.float64)
        rounded = _chop_batch(samples)

        assert tuple(rounded.shape) == tuple(samples.shape)
        assert not np.array_equal(rounded.numpy(), samples.numpy())
        np.testing.assert_allclose(rounded.numpy(), samples.numpy(), rtol=2e-7, atol=0.0)

    def test_torch_three_samples_use_random_directed_rounding(self):
        configure(backend="torch", exp_bits=8, sig_bits=23, n_samples=3, random_state=7)
        nf.reset_chopper_cache()
        from noisefloat.core import _chop_batch

        spacing = 2.0**-23
        samples = torch.full((3, 8), 1.0 + 0.25 * spacing, dtype=torch.float64)
        rounded1 = _chop_batch(samples)

        nf.reset_chopper_cache()
        rounded2 = _chop_batch(samples)

        assert tuple(rounded1.shape) == tuple(samples.shape)
        np.testing.assert_array_equal(rounded1.numpy(), rounded2.numpy())
        assert set(np.unique(rounded1.numpy())).issubset({1.0, 1.0 + spacing})
        assert np.unique(rounded1.numpy()).size == 2

    def test_torch_three_samples_do_not_use_fixed_nearest_up_down_tracks(self):
        configure(backend="torch", exp_bits=8, sig_bits=23, n_samples=3, random_state=7)
        nf.reset_chopper_cache()
        from noisefloat.core import _chop_batch

        spacing = 2.0**-23
        samples = torch.ones((3, 8), dtype=torch.float64)
        rounded = _chop_batch(samples).numpy()

        assert set(np.unique(rounded)).issubset({1.0 - spacing, 1.0 + spacing})
        assert not np.any(rounded == 1.0)
        assert not np.all(rounded[0] == 1.0)

    def test_jax_rounding_is_vectorized_native_and_reproducible(self):
        jax = pytest.importorskip("jax", reason="jax not installed")
        jnp = pytest.importorskip("jax.numpy", reason="jax not installed")
        jax.config.update("jax_enable_x64", True)

        configure(backend="jax", exp_bits=5, sig_bits=3, n_samples=4, random_state=7)
        nf.reset_chopper_cache()
        from noisefloat.core import _chop_batch

        base = jnp.linspace(-3.0, 3.0, 17, dtype=jnp.float64)
        samples = jnp.broadcast_to(base, (4, 17))
        rounded1 = _chop_batch(samples)

        nf.reset_chopper_cache()
        rounded2 = _chop_batch(samples)

        assert "jax" in type(rounded1).__module__
        assert rounded1.shape == (4, 17)
        np.testing.assert_array_equal(np.asarray(rounded1), np.asarray(rounded2))

    def test_jax_random_directed_rounding_perturbs_nearest_grid_values(self):
        jax = pytest.importorskip("jax", reason="jax not installed")
        jnp = pytest.importorskip("jax.numpy", reason="jax not installed")
        jax.config.update("jax_enable_x64", True)

        configure(backend="jax", exp_bits=8, sig_bits=23, n_samples=4, random_state=7)
        nf.reset_chopper_cache()
        from noisefloat.core import _chop_batch

        samples = jnp.broadcast_to(
            jnp.asarray([1.0, 2.0, -1.0, -2.0], dtype=jnp.float64), (4, 4)
        )
        rounded = _chop_batch(samples)

        rounded_np = np.asarray(rounded)
        samples_np = np.asarray(samples)
        assert rounded_np.shape == samples_np.shape
        assert not np.array_equal(rounded_np, samples_np)
        np.testing.assert_allclose(rounded_np, samples_np, rtol=2e-7, atol=0.0)

    def test_tensorflow_rounding_is_vectorized_native_and_reproducible(self):
        tf = pytest.importorskip("tensorflow", reason="tensorflow not installed")

        configure(
            backend="tensorflow", exp_bits=5, sig_bits=3, n_samples=4, random_state=7
        )
        nf.reset_chopper_cache()
        from noisefloat.core import _chop_batch

        base = tf.linspace(
            tf.constant(-3.0, dtype=tf.float64),
            tf.constant(3.0, dtype=tf.float64),
            17,
        )
        samples = tf.broadcast_to(base, (4, 17))
        rounded1 = _chop_batch(samples)

        nf.reset_chopper_cache()
        rounded2 = _chop_batch(samples)

        assert isinstance(rounded1, tf.Tensor)
        assert tuple(rounded1.shape) == (4, 17)
        np.testing.assert_array_equal(rounded1.numpy(), rounded2.numpy())

    def test_tensorflow_random_directed_rounding_perturbs_nearest_grid_values(self):
        tf = pytest.importorskip("tensorflow", reason="tensorflow not installed")

        configure(
            backend="tensorflow", exp_bits=8, sig_bits=23, n_samples=4, random_state=7
        )
        nf.reset_chopper_cache()
        from noisefloat.core import _chop_batch

        samples = tf.broadcast_to(
            tf.constant([1.0, 2.0, -1.0, -2.0], dtype=tf.float64), (4, 4)
        )
        rounded = _chop_batch(samples)

        assert tuple(rounded.shape) == tuple(samples.shape)
        assert not np.array_equal(rounded.numpy(), samples.numpy())
        np.testing.assert_allclose(rounded.numpy(), samples.numpy(), rtol=2e-7, atol=0.0)


# --------------------------------------------------------------------------- #
#  4. Significant digits                                                       #
# --------------------------------------------------------------------------- #


class TestSignificantDigits:
    def test_stable_value_high_digits(self):
        a = NFloat(3.14159)
        # Float32 precision gives ~7 decimal digits
        assert a.digits > 4.0, f"Expected > 4 digits, got {a.digits}"

    def test_exact_zero_digits(self):
        # Zero exactly: all samples are 0 → std=0 → 15 digits
        z = NFloat(0.0)
        # Both samples are chopped versions of 0 → should equal 0
        assert z.mean == 0.0
        # is_numerical_zero should flag it
        assert z.is_numerical_zero()

    def test_digits_array(self):
        a = NFloat(np.array([1.0, 2.0, 3.0]))
        d = a.digits
        assert d.shape == (3,)
        assert np.all(d >= 0)

    def test_confidence_interval(self):
        a = NFloat(2.0)
        lo, hi = a.confidence_interval()
        assert lo <= a.mean <= hi


# --------------------------------------------------------------------------- #
#  5. Unstable comparisons                                                     #
# --------------------------------------------------------------------------- #


class TestUnstableComparisons:
    def _make_crossing(self):
        """Construct two NFloat objects whose samples cross."""
        a = NFloat.__new__(NFloat)
        a._samples = np.array([1.0, 2.0, 1.5])  # span below and above 1.7
        b = NFloat.__new__(NFloat)
        b._samples = np.array([1.7, 1.7, 1.7])
        return a, b

    def test_unstable_lt_warns(self):
        a, b = self._make_crossing()
        with pytest.warns(UnstableComparisonWarning):
            _ = a < b

    def test_unstable_gt_warns(self):
        a, b = self._make_crossing()
        with pytest.warns(UnstableComparisonWarning):
            _ = b > a

    def test_is_unstable_comparison(self):
        a, b = self._make_crossing()
        assert is_unstable_comparison(a, b, "lt")
        assert is_unstable_comparison(b, a, "gt")

    def test_stable_comparison_no_warn(self):
        a = NFloat(1.0)
        b = NFloat(3.0)
        # These should agree across all samples
        with warnings.catch_warnings():
            warnings.simplefilter("error", UnstableComparisonWarning)
            result = a < b
        assert result

    def test_comparison_returns_bool_scalar(self):
        a = NFloat(1.0)
        b = NFloat(2.0)
        result = a < b
        assert isinstance(result, (bool, np.bool_))

    def test_is_unstable_comparison_invalid_op(self):
        a = NFloat(1.0)
        b = NFloat(2.0)
        with pytest.raises(ValueError):
            is_unstable_comparison(a, b, "invalid")


# --------------------------------------------------------------------------- #
#  6. NaN / Inf diagnostics                                                    #
# --------------------------------------------------------------------------- #


class TestAnomalyDetection:
    def test_nan_warns(self):
        a = NFloat.__new__(NFloat)
        a._samples = np.array([np.nan, 1.0, 1.0])
        # Trigger _maybe_record by doing an op
        with pytest.warns(nf.FloatingPointAnomalyWarning):
            # Directly create result with NaN
            from noisefloat.diagnostics import warn_anomaly

            warn_anomaly("test nan")

    def test_log_negative_gives_nan(self):
        # log of negative number → NaN in samples
        a = NFloat(-1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = log(a)
        assert np.any(np.isnan(result.samples))

    def test_inf_detected_in_digits(self):
        a = NFloat.__new__(NFloat)
        a._samples = np.array([np.inf, np.inf, np.inf])
        assert a.digits == 0.0


# --------------------------------------------------------------------------- #
#  7. Math functions                                                           #
# --------------------------------------------------------------------------- #


class TestMathFunctions:
    def test_sqrt(self):
        a = sqrt(NFloat(4.0))
        assert abs(a.mean - 2.0) < 1e-4

    def test_exp(self):
        a = exp(NFloat(0.0))
        assert abs(a.mean - 1.0) < 1e-4

    def test_log(self):
        a = log(NFloat(np.e))
        assert abs(a.mean - 1.0) < 1e-4

    def test_sin_zero(self):
        a = sin(NFloat(0.0))
        assert abs(a.mean) < 1e-5

    def test_cos_zero(self):
        a = cos(NFloat(0.0))
        assert abs(a.mean - 1.0) < 1e-4

    def test_maximum(self):
        from noisefloat import maximum

        a = NFloat(1.0)
        b = NFloat(3.0)
        c = maximum(a, b)
        assert abs(c.mean - 3.0) < 1e-4

    def test_minimum(self):
        from noisefloat import minimum

        a = NFloat(1.0)
        b = NFloat(3.0)
        c = minimum(a, b)
        assert abs(c.mean - 1.0) < 1e-4

    def test_where(self):
        from noisefloat import where

        cond = np.array([True, False, True])
        x = NFloat(np.array([1.0, 2.0, 3.0]))
        y = NFloat(np.array([10.0, 20.0, 30.0]))
        result = where(cond, x, y)
        expected = np.array([1.0, 20.0, 3.0])
        np.testing.assert_allclose(result.mean, expected, rtol=1e-4)

    def test_where_uses_nfloat_condition_samplewise(self):
        from noisefloat import where

        cond = NFloat(0.0, _samples=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        x = NFloat(0.0, _samples=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        y = NFloat(0.0, _samples=np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]))

        result = where(cond, x, y)

        np.testing.assert_allclose(
            result.samples,
            np.array([[1.0, 20.0], [30.0, 4.0], [5.0, 6.0]]),
            rtol=1e-5,
        )

    def test_sqrt_array(self):
        a = sqrt(NFloat(np.array([1.0, 4.0, 9.0, 16.0])))
        np.testing.assert_allclose(a.mean, [1.0, 2.0, 3.0, 4.0], rtol=1e-4)


# --------------------------------------------------------------------------- #
#  8. Diagnostics system                                                       #
# --------------------------------------------------------------------------- #


class TestDiagnostics:
    def test_trace_records_events(self):
        configure(trace=True)
        from noisefloat.core import _chopper

        _chopper._config_hash = ()
        a = NFloat(2.0)
        b = NFloat(3.0)
        _ = a + b
        events = get_diagnostics()
        assert len(events) >= 1

    def test_clear_diagnostics(self):
        configure(trace=True)
        from noisefloat.core import _chopper

        _chopper._config_hash = ()
        a = NFloat(2.0) + NFloat(1.0)
        assert len(get_diagnostics()) >= 1
        clear_diagnostics()
        assert get_diagnostics() == []

    def test_print_diagnostics_empty(self, capsys):
        clear_diagnostics()
        print_diagnostics()
        captured = capsys.readouterr()
        assert "No diagnostics" in captured.out

    def test_print_diagnostics_with_events(self, capsys):
        configure(trace=True)
        from noisefloat.core import _chopper

        _chopper._config_hash = ()
        _ = NFloat(1.0) + NFloat(2.0)
        print_diagnostics()
        captured = capsys.readouterr()
        assert "op=" in captured.out

    def test_summary_records_cancellation_source(self):
        configure(
            diagnostics_level="summary",
            digits_threshold=6.0,
            cancellation_ratio_threshold=1e-6,
        )
        left = NFloat(0.0, _samples=np.array([1e8 + 2.0, 1e8 - 2.0, 1e8 + 4.0]))
        right = NFloat(0.0, _samples=np.array([1e8, 1e8, 1e8]))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = left - right

        summary = get_diagnostics_summary()
        assert summary["by_kind"]["loss_of_accuracy_due_to_cancellation"] >= 1

    def test_summary_records_addition_cancellation_source(self):
        configure(
            diagnostics_level="summary",
            digits_threshold=6.0,
            cancellation_ratio_threshold=1e-6,
        )
        left = NFloat(0.0, _samples=np.array([1e8 + 2.0, 1e8 - 2.0, 1e8 + 4.0]))
        right = NFloat(0.0, _samples=np.array([-1e8, -1e8, -1e8]))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = left + right

        summary = get_diagnostics_summary()
        assert summary["by_kind"]["loss_of_accuracy_due_to_cancellation"] >= 1

    def test_rump83_p1_records_two_cancellations(self):
        configure(
            backend="numpy",
            exp_bits=11,
            sig_bits=52,
            n_samples=3,
            random_state=42,
            diagnostics_level="summary",
            digits_threshold=6.0,
            zero_digits_threshold=6.0,
            cancellation_ratio_threshold=1e-6,
        )
        x = NFloat(10864.0)
        y = NFloat(18817.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = 9.0 * x * x * x * x - y * y * y * y + 2.0 * y * y

        summary = get_diagnostics_summary()
        assert summary["by_kind"]["loss_of_accuracy_due_to_cancellation"] == 2

    def test_summary_records_cadna_style_unstable_division(self):
        configure(diagnostics_level="summary", zero_digits_threshold=0.5)
        numerator = NFloat(1.0)
        denominator = NFloat(
            0.0,
            _samples=np.array([1.0e-6, -1.0e-6, 2.0e-6], dtype=np.float64),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = numerator / denominator

        summary = get_diagnostics_summary()
        assert summary["by_kind"]["unstable_division"] >= 1

    def test_summary_records_cadna_style_unstable_multiplication(self):
        configure(diagnostics_level="summary", zero_digits_threshold=0.5)
        left = NFloat(
            0.0,
            _samples=np.array([1.0, -1.0, 2.0], dtype=np.float64),
        )
        right = NFloat(
            0.0,
            _samples=np.array([3.0, -3.0, 1.0], dtype=np.float64),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = left * right

        summary = get_diagnostics_summary()
        assert summary["by_kind"]["unstable_multiplication"] >= 1

    def test_summary_records_branching_source(self):
        configure(diagnostics_level="summary")
        value = NFloat(0.0, _samples=np.array([-1.0, 1.0, -1.0]))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = value > 0.0

        summary = get_diagnostics_summary()
        assert summary["by_kind"]["branching_instability"] >= 1

    def test_summary_records_mathematical_source(self):
        configure(diagnostics_level="summary")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = log(NFloat(-1.0))

        summary = get_diagnostics_summary()
        assert summary["by_kind"]["mathematical_instability"] >= 1

    def test_print_diagnostics_summary(self, capsys):
        configure(diagnostics_level="summary", digits_threshold=6.0)
        left = NFloat(0.0, _samples=np.array([1e8 + 2.0, 1e8 - 2.0, 1e8 + 4.0]))
        right = NFloat(0.0, _samples=np.array([1e8, 1e8, 1e8]))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = left - right
        print_diagnostics_summary()

        captured = capsys.readouterr()
        assert "numerical instabilities" in captured.out
        assert "CANCELLATION" in captured.out


# --------------------------------------------------------------------------- #
#  9. Config API                                                               #
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_configure_n_samples(self):
        configure(n_samples=7)
        a = NFloat(1.0)
        assert a.n_samples == 7

    def test_configure_unknown_key(self):
        with pytest.raises(ValueError):
            configure(nonexistent_key=42)

    def test_get_config_returns_copy(self):
        cfg = get_config()
        cfg.n_samples = 999
        assert get_config().n_samples != 999


# --------------------------------------------------------------------------- #
#  10. NumPy protocol                                                          #
# --------------------------------------------------------------------------- #


class TestNumpyProtocol:
    def test_array_protocol(self):
        a = NFloat(np.array([1.0, 2.0, 3.0]))
        arr = np.asarray(a)
        np.testing.assert_allclose(arr, a.mean, rtol=1e-10)

    def test_array_ufunc_add(self):
        a = NFloat(1.5)
        b = NFloat(2.5)
        c = np.add(a, b)
        assert isinstance(c, NFloat)
        assert abs(c.mean - 4.0) < 1e-4

    def test_array_priority(self):
        assert NFloat.__array_priority__ > 0


# --------------------------------------------------------------------------- #
#  11. Optional torch smoke test                                               #
# --------------------------------------------------------------------------- #

torch = pytest.importorskip("torch", reason="torch not installed")


class TestTorchSmoke:
    def test_torch_backend_import(self):
        from noisefloat.backends import TorchBackend

        backend = TorchBackend()
        assert backend.name == "torch"

    def test_torch_basic_ops(self):
        from noisefloat.backends import TorchBackend

        backend = TorchBackend()
        x = backend.asarray([1.0, 2.0, 3.0])
        y = backend.sqrt(x)
        result = backend.to_numpy(y)
        np.testing.assert_allclose(result, [1.0, np.sqrt(2), np.sqrt(3)], rtol=1e-5)


# --------------------------------------------------------------------------- #
#  12. Torch backend – full NFloat integration                                  #
# --------------------------------------------------------------------------- #


class TestTorchNFloat:
    """Test that NFloat works natively with PyTorch tensors."""

    def test_nfloat_from_torch_scalar(self):
        configure(backend="torch")
        from noisefloat.core import _chopper

        _chopper._config_hash = ()
        x = torch.tensor(3.14, dtype=torch.float64)
        c = NFloat(x)
        assert c.backend_name == "torch"
        assert isinstance(c.samples, torch.Tensor)
        assert abs(c.mean - 3.14) < 1e-3

    def test_nfloat_from_torch_array(self):
        configure(backend="torch")
        from noisefloat.core import _chopper

        _chopper._config_hash = ()
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        c = NFloat(x)
        assert c.backend_name == "torch"
        assert c.shape == (3,)
        assert isinstance(c.samples, torch.Tensor)
        np.testing.assert_allclose(c.mean, [1.0, 2.0, 3.0], rtol=1e-3)

    def test_torch_arithmetic(self):
        configure(backend="torch")
        from noisefloat.core import _chopper

        _chopper._config_hash = ()
        a = NFloat(torch.tensor(2.0, dtype=torch.float64))
        b = NFloat(torch.tensor(3.0, dtype=torch.float64))
        c = a + b
        assert isinstance(c.samples, torch.Tensor)
        assert abs(c.mean - 5.0) < 1e-3

    def test_torch_mul(self):
        configure(backend="torch")
        from noisefloat.core import _chopper

        _chopper._config_hash = ()
        a = NFloat(torch.tensor(4.0, dtype=torch.float64))
        b = NFloat(torch.tensor(3.0, dtype=torch.float64))
        c = a * b
        assert abs(c.mean - 12.0) < 1e-2

    def test_torch_div(self):
        configure(backend="torch")
        from noisefloat.core import _chopper

        _chopper._config_hash = ()
        a = NFloat(torch.tensor(6.0, dtype=torch.float64))
        b = NFloat(torch.tensor(2.0, dtype=torch.float64))
        c = a / b
        assert abs(c.mean - 3.0) < 1e-3

    def test_torch_neg(self):
        configure(backend="torch")
        from noisefloat.core import _chopper

        _chopper._config_hash = ()
        a = NFloat(torch.tensor(5.0, dtype=torch.float64))
        b = -a
        assert abs(b.mean - (-5.0)) < 1e-3

    def test_torch_significant_digits(self):
        configure(backend="torch")
        from noisefloat.core import _chopper

        _chopper._config_hash = ()
        a = NFloat(torch.tensor(3.14, dtype=torch.float64))
        d = a.digits
        assert d > 4.0

    def test_torch_comparison(self):
        configure(backend="torch")
        from noisefloat.core import _chopper

        _chopper._config_hash = ()
        a = NFloat(torch.tensor(1.0, dtype=torch.float64))
        b = NFloat(torch.tensor(3.0, dtype=torch.float64))
        assert a < b

    def test_torch_scalar_with_config_backend(self):
        """Plain Python scalar should use torch backend when configured."""
        configure(backend="torch")
        from noisefloat.core import _chopper

        _chopper._config_hash = ()
        a = NFloat(2.0)
        assert a.backend_name == "torch"
        assert isinstance(a.samples, torch.Tensor)

    def test_torch_norm_preserves_backend_and_value(self):
        configure(backend="torch")
        from noisefloat.core import _chopper

        _chopper._config_hash = ()
        a = NFloat(torch.tensor([3.0, 4.0], dtype=torch.float64))
        n = nf_norm(a)
        assert n.backend_name == "torch"
        assert isinstance(n.samples, torch.Tensor)
        assert abs(n.mean - 5.0) < 1e-3


# --------------------------------------------------------------------------- #
#  13. NFloatSTE – Straight-Through Estimator for automatic differentiation     #
# --------------------------------------------------------------------------- #


class TestNFloatSTE:
    """Test NFloatSTE differentiable stochastic arithmetic."""

    def test_ste_basic_creation(self):
        configure(backend="torch")
        from noisefloat.core import _chopper

        _chopper._config_hash = ()
        from noisefloat import NFloatSTE

        x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        c = NFloatSTE(x)
        assert isinstance(c, NFloatSTE)
        assert abs(c.mean - 2.0) < 1e-3

    def test_ste_gradient_flows(self):
        """Verify that gradients flow through NFloatSTE operations."""
        configure(backend="torch")
        from noisefloat.core import _chopper, _ste_chopper

        _chopper._config_hash = ()
        if _ste_chopper is not None:
            _ste_chopper._config_hash = ()
        from noisefloat import NFloatSTE

        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        c = NFloatSTE(x)
        # Sum all samples
        loss = c.samples.sum()
        loss.backward()
        # STE should pass gradients through – each sample gets grad=1
        # n_samples=3, so each input element appears 3 times
        assert x.grad is not None
        assert x.grad.shape == (3,)
        # Each element is used in n_samples copies
        np.testing.assert_allclose(
            x.grad.detach().numpy(),
            [3.0, 3.0, 3.0],
            rtol=1e-5,
        )

    def test_ste_arithmetic_gradient(self):
        """Gradients flow through STE arithmetic ops."""
        configure(backend="torch", n_samples=3)
        from noisefloat.core import _chopper, _ste_chopper

        _chopper._config_hash = ()
        if _ste_chopper is not None:
            _ste_chopper._config_hash = ()
        from noisefloat import NFloatSTE

        x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        a = NFloatSTE(x)
        b = NFloatSTE(torch.tensor(3.0, dtype=torch.float64))
        c = a + b
        loss = c.samples.sum()
        loss.backward()
        assert x.grad is not None
        # Gradient should be non-zero (STE passes through)
        assert x.grad.item() != 0.0

    def test_ste_mul_gradient(self):
        """Gradients flow through NFloatSTE multiplication."""
        configure(backend="torch", n_samples=3)
        from noisefloat.core import _chopper, _ste_chopper

        _chopper._config_hash = ()
        if _ste_chopper is not None:
            _ste_chopper._config_hash = ()
        from noisefloat import NFloatSTE

        x = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        c = NFloatSTE(x)
        result = c * 2.0
        loss = result.samples.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.item() != 0.0

    def test_ste_returns_nfloatste(self):
        """Operations on NFloatSTE return NFloatSTE instances."""
        configure(backend="torch")
        from noisefloat.core import _chopper, _ste_chopper

        _chopper._config_hash = ()
        if _ste_chopper is not None:
            _ste_chopper._config_hash = ()
        from noisefloat import NFloatSTE

        a = NFloatSTE(torch.tensor(1.0, dtype=torch.float64))
        b = NFloatSTE(torch.tensor(2.0, dtype=torch.float64))
        c = a + b
        assert isinstance(c, NFloatSTE)

    def test_ste_numpy_fallback(self):
        """NFloatSTE works with numpy (falls back to regular rounding)."""
        configure(backend="numpy")
        from noisefloat.core import _chopper

        _chopper._config_hash = ()
        from noisefloat import NFloatSTE

        c = NFloatSTE(3.14)
        assert abs(c.mean - 3.14) < 1e-3
        assert c.backend_name == "numpy"

    def test_ste_significant_digits(self):
        """NFloatSTE computes significant digits like NFloat."""
        configure(backend="torch")
        from noisefloat.core import _chopper, _ste_chopper

        _chopper._config_hash = ()
        if _ste_chopper is not None:
            _ste_chopper._config_hash = ()
        from noisefloat import NFloatSTE

        c = NFloatSTE(torch.tensor(3.14, dtype=torch.float64))
        d = c.digits
        assert d > 4.0

    def test_torch_ste_batch_rounding_is_vectorized_and_native(self):
        """STE batch rounding keeps PyTorch tensors native and propagates identity gradients."""
        configure(backend="torch", n_samples=3, random_state=42)
        nf.reset_chopper_cache()
        from noisefloat.core import _chop_batch_ste

        x = torch.tensor([1.25, 2.5, 3.75], dtype=torch.float64, requires_grad=True)
        samples = x.expand(3, -1).clone()
        rounded = _chop_batch_ste(samples)
        loss = rounded.sum()
        loss.backward()

        assert isinstance(rounded, torch.Tensor)
        assert rounded.shape == (3, 3)
        np.testing.assert_allclose(x.grad.detach().numpy(), [3.0, 3.0, 3.0])

    def test_jax_ste_batch_rounding_is_vectorized_and_native(self):
        """STE batch rounding keeps JAX arrays native and propagates identity gradients."""
        jax = pytest.importorskip("jax", reason="jax not installed")
        jnp = pytest.importorskip("jax.numpy", reason="jax not installed")
        jax.config.update("jax_enable_x64", True)

        configure(backend="jax", n_samples=3, random_state=42)
        nf.reset_chopper_cache()
        from noisefloat.core import _chop_batch_ste

        def loss_fn(x):
            samples = jnp.broadcast_to(x, (3,) + x.shape)
            return jnp.sum(_chop_batch_ste(samples))

        x = jnp.asarray([1.25, 2.5, 3.75], dtype=jnp.float64)
        rounded = _chop_batch_ste(jnp.broadcast_to(x, (3,) + x.shape))
        grad = jax.grad(loss_fn)(x)

        assert "jax" in type(rounded).__module__
        assert rounded.shape == (3, 3)
        np.testing.assert_allclose(np.asarray(grad), [3.0, 3.0, 3.0])

    def test_tensorflow_ste_batch_rounding_is_vectorized_and_native(self):
        """STE batch rounding keeps TensorFlow tensors native and propagates identity gradients."""
        tf = pytest.importorskip("tensorflow", reason="tensorflow not installed")

        configure(backend="tensorflow", n_samples=3, random_state=42)
        nf.reset_chopper_cache()
        from noisefloat.core import _chop_batch_ste

        x = tf.Variable([1.25, 2.5, 3.75], dtype=tf.float64)
        with tf.GradientTape() as tape:
            samples = tf.broadcast_to(x, (3,) + tuple(x.shape))
            rounded = _chop_batch_ste(samples)
            loss = tf.reduce_sum(rounded)
        grad = tape.gradient(loss, x)

        assert isinstance(rounded, tf.Tensor)
        assert tuple(rounded.shape) == (3, 3)
        np.testing.assert_allclose(grad.numpy(), [3.0, 3.0, 3.0])
