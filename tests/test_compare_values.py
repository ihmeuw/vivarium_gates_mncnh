"""Tests for value comparison in vivarium_gates_mncnh.tools.compare.core."""

import math

import numpy as np
import pandas as pd
import pytest

from vivarium_gates_mncnh.tools.compare import ValueDiff, compare_values


def test_identical_values_compare_equal():
    """Identical numeric series return None at default (exact) tolerance."""
    reference = pd.Series([1.0, 2.5, -3.0, 1e-12, 1e12, 0.0])
    assert compare_values(reference, reference.copy()) is None


def test_changed_values_report_count_and_magnitude():
    """Changed values return a ValueDiff with n_changed and the max abs/rel diffs."""
    reference = pd.Series([1.0, 2.0, 4.0, 10.0])
    # Position 1 moves by 0.5 (rel 0.25); position 3 by 1.0 (rel 0.1). The largest
    # absolute and largest relative change come from different positions.
    candidate = pd.Series([1.0, 2.5, 4.0, 9.0])

    diff = compare_values(reference, candidate)

    assert isinstance(diff, ValueDiff)
    assert diff.n_changed == 2
    assert diff.max_abs_diff == pytest.approx(1.0)
    assert diff.max_rel_diff == pytest.approx(0.25)


def test_any_change_is_reported_by_default():
    """With default tolerances, even a change at the last float digit is reported."""
    reference = pd.Series([0.1, 0.2, 0.3])
    candidate = reference.copy()
    candidate[1] = np.nextafter(0.2, 1.0)

    diff = compare_values(reference, candidate)

    assert diff is not None
    assert diff.n_changed == 1
    assert diff.max_abs_diff > 0


def test_rtol_suppresses_small_relative_change():
    """A change within rtol of the reference is not reported; one outside it is."""
    reference = pd.Series([100.0, 1.0])
    # Both move by about 1e-5 absolute: within rtol=1e-6 of 100, outside it for 1.
    within = pd.Series([100.00001, 1.0])
    mixed = pd.Series([100.00001, 1.00001])

    assert compare_values(reference, within, rtol=1e-6) is None
    assert compare_values(reference, within) is not None

    diff = compare_values(reference, mixed, rtol=1e-6)
    assert diff is not None
    assert diff.n_changed == 1

    # The tolerance is measured against the reference, not the candidate:
    # |2 - 1| = 1 > 0.5 * |1|, but |1 - 2| = 1 <= 0.5 * |2|.
    assert compare_values(pd.Series([1.0]), pd.Series([2.0]), rtol=0.5) is not None
    assert compare_values(pd.Series([2.0]), pd.Series([1.0]), rtol=0.5) is None


def test_atol_suppresses_small_absolute_change():
    """A change within atol is not reported; one outside it is."""
    reference = pd.Series([0.0, 5.0, 1000.0])
    within = pd.Series([1e-9, 5.0 + 1e-8, 1000.0 - 1e-7])
    mixed = pd.Series([1e-9, 5.1, 1000.0])

    assert compare_values(reference, within, atol=1e-6) is None
    assert compare_values(reference, within) is not None

    diff = compare_values(reference, mixed, atol=1e-6)
    assert diff is not None
    assert diff.n_changed == 1
    assert diff.max_abs_diff == pytest.approx(0.1)


def test_nan_equals_nan():
    """NaN in both series at the same position is equal."""
    reference = pd.Series([1.0, np.nan, 3.0])
    candidate = pd.Series([1.0, np.nan, 3.0])
    assert compare_values(reference, candidate) is None


@pytest.mark.parametrize(
    "reference, candidate",
    [
        ([1.0, np.nan, 3.0], [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [1.0, np.nan, 3.0]),
        ([np.nan, 2.0, 3.0], [1.0, np.nan, 3.0]),
    ],
    ids=["nan_in_reference", "nan_in_candidate", "nan_on_each_side"],
)
def test_nan_against_number_is_a_change(reference, candidate):
    """NaN on one side only counts toward n_changed."""
    diff = compare_values(pd.Series(reference), pd.Series(candidate))

    expected_changed = sum(
        math.isnan(r) != math.isnan(c) for r, c in zip(reference, candidate)
    )
    assert diff is not None
    assert diff.n_changed == expected_changed
    # No changed position has two finite numbers, so there is no magnitude.
    assert diff.max_abs_diff is None
    assert diff.max_rel_diff is None


def test_relative_diff_against_zero_reference_is_inf():
    """A change from a reference of exactly zero gives max_rel_diff of inf."""
    reference = pd.Series([0.0, 1.0])
    candidate = pd.Series([0.5, 1.0])

    diff = compare_values(reference, candidate)

    assert diff is not None
    assert diff.n_changed == 1
    assert diff.max_abs_diff == pytest.approx(0.5)
    assert diff.max_rel_diff == math.inf


def test_int_and_float_with_equal_values_compare_equal():
    """An int series and a float series holding the same numbers compare equal."""
    ints = pd.Series([1, 2, 3, 0, -7], dtype="int64")
    floats = pd.Series([1.0, 2.0, 3.0, 0.0, -7.0], dtype="float64")

    assert compare_values(ints, floats) is None
    assert compare_values(floats, ints) is None


def test_non_numeric_values_compared_for_equality():
    """String values compare equal when identical and are counted when they differ."""
    reference = pd.Series(["Female", "Male", "Female", "Male"])

    assert compare_values(reference, reference.copy()) is None

    candidate = pd.Series(["Female", "female", "Female", "Unknown"])
    diff = compare_values(reference, candidate)
    assert diff is not None
    assert diff.n_changed == 2


@pytest.mark.parametrize(
    "candidate_index",
    [[0, 1, 3], [0, 1], [0, 1, 2, 3]],
    ids=["different_labels", "shorter", "longer"],
)
def test_misaligned_indexes_raise(candidate_index):
    """Series with different indexes raise ValueError."""
    reference = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
    candidate = pd.Series(np.arange(len(candidate_index), dtype=float), index=candidate_index)

    with pytest.raises(ValueError):
        compare_values(reference, candidate)
