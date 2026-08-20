from __future__ import annotations

from fractions import Fraction

import numpy as np

from examples.rump83_polynomial import (
    DEFAULT_INPUTS,
    analyze_rump83,
    rump_exact,
    rump_float64,
    significant_digits_against_reference,
)


def test_rump83_exact_and_float64_results_match_classic_failure():
    p1, p2 = DEFAULT_INPUTS

    assert rump_exact(p1.x, p1.y) == Fraction(1, 1)
    assert rump_float64(float(p1.x), float(p1.y)) == 2.0

    assert rump_exact(p2.x, p2.y) == Fraction(65, 81)
    assert np.isclose(rump_float64(float(p2.x), float(p2.y)), 65.0 / 81.0)


def test_reference_aware_digits_separate_unstable_and_stable_cases():
    p1_report, p2_report = analyze_rump83()

    assert p1_report.label == "P1"
    assert p1_report.reference_digits == 0.0
    assert not p1_report.reference_based_stable
    assert p1_report.double_result == 2.0
    assert p1_report.exact_result == Fraction(1, 1)

    assert p2_report.label == "P2"
    assert p2_report.reference_digits > 14.0
    assert p2_report.reference_based_stable
    assert p2_report.exact_result == Fraction(65, 81)


def test_reference_digit_helper_counts_wrong_integer_result_as_no_digits():
    assert significant_digits_against_reference(2.0, Fraction(1, 1)) == 0.0
    assert significant_digits_against_reference(65.0 / 81.0, Fraction(65, 81)) > 14.0
