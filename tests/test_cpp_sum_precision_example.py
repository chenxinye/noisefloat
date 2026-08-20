from __future__ import annotations

import numpy as np

from examples.cpp_sum_precision import (
    analyze_sums,
    cpp_code1_float_sum,
    cpp_code2_double_sum_then_float,
    exact_real_sum,
)


def test_cpp_sum_deterministic_results_match_expected_patterns():
    iterations = 100_000_000

    code1 = cpp_code1_float_sum(iterations)
    code2 = cpp_code2_double_sum_then_float(iterations)
    exact = exact_real_sum(iterations)

    assert float(code1) == 1.0
    assert abs(float(code2) - exact) < 5e-7
    assert abs(float(code1) - exact) > 5.0


def test_noisefloat_detects_float_sum_less_stable_than_double_temp_sum():
    code1, code2 = analyze_sums(nfloat_iterations=1_000)

    assert not code1.is_numerically_stable
    assert code2.is_numerically_stable
    assert code1.nfloat_digits < code2.nfloat_digits
    assert np.isfinite(code1.nfloat_digits)
    assert np.isfinite(code2.nfloat_digits)
