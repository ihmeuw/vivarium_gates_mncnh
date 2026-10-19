"""Tests for vivarium_gates_mncnh.tools.compare.artifacts.compare_artifacts."""


def test_identical_artifacts_compare_clean():
    """Two artifacts written separately with the same keys and data compare clean."""


def test_reordered_data_compares_clean():
    """The same key written with its rows and columns in a different order compares
    clean."""


def test_rerunning_on_unchanged_inputs_is_stable():
    """Comparing the same pair twice gives the same, clean result."""


def test_changed_values_name_key_and_magnitude():
    """A key whose values moved is reported with n_changed and max abs/rel diff;
    unchanged keys are not reported."""


def test_tolerance_is_applied():
    """A small change is reported by default and not reported within rtol/atol."""


def test_key_only_on_one_side():
    """A key present in only one artifact is reported as only in reference or only in
    candidate."""


def test_shape_change_reported():
    """A key whose index levels or columns differ is reported as a schema change."""


def test_non_dataframe_values_compared():
    """A non-DataFrame value (e.g. a list) is compared for equality and reported when
    it differs."""


def test_missing_path_raises_without_creating_file():
    """A path that does not exist raises ComparisonInputError and no file is created
    there."""


def test_inputs_not_modified():
    """Comparing does not modify either artifact file."""
