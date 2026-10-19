"""Tests for value comparison in vivarium_gates_mncnh.tools.compare.core."""


def test_identical_values_compare_equal():
    """Identical numeric series return None at default (exact) tolerance."""


def test_changed_values_report_count_and_magnitude():
    """Changed values return a ValueDiff with n_changed and the max abs/rel diffs."""


def test_any_change_is_reported_by_default():
    """With default tolerances, even a change at the last float digit is reported."""


def test_rtol_suppresses_small_relative_change():
    """A change within rtol of the reference is not reported; one outside it is."""


def test_atol_suppresses_small_absolute_change():
    """A change within atol is not reported; one outside it is."""


def test_nan_equals_nan():
    """NaN in both series at the same position is equal."""


def test_nan_against_number_is_a_change():
    """NaN on one side only counts toward n_changed."""


def test_relative_diff_against_zero_reference_is_inf():
    """A change from a reference of exactly zero gives max_rel_diff of inf."""


def test_int_and_float_with_equal_values_compare_equal():
    """An int series and a float series holding the same numbers compare equal."""


def test_non_numeric_values_compared_for_equality():
    """String values compare equal when identical and are counted when they differ."""


def test_misaligned_indexes_raise():
    """Series with different indexes raise ValueError."""
