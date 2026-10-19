"""Tests for vivarium_gates_mncnh.tools.compare.results.compare_results."""


def test_identical_results_compare_clean():
    """Two directories holding the same measures and values have no differences."""


def test_rewritten_files_compare_clean():
    """The same data rewritten with different compression, row-group size, row order
    and column order compares clean."""


def test_flat_and_sharded_layouts_compare_clean():
    """A measure as <measure>.parquet on one side and as <measure>/ shards on the
    other, holding the same rows, compares clean."""


def test_rerunning_on_unchanged_inputs_is_stable():
    """Comparing the same pair twice gives the same, clean result."""


def test_changed_values_name_measure_and_magnitude():
    """A moved value is reported against its measure with n_changed and max abs/rel diff;
    unchanged measures are not reported."""


def test_tolerance_is_applied():
    """A small change is reported by default and not reported within rtol/atol."""


def test_measure_only_on_one_side():
    """A measure present in only one directory is reported as only in reference or
    only in candidate."""


def test_rows_only_on_one_side():
    """Rows present on only one side of a measure are reported against that measure."""


def test_schema_change_reported():
    """A measure whose identifying columns differ is reported as a schema change."""


def test_n_compared_counts_measures():
    """n_compared counts each measure appearing on either side once."""


def test_environment_changes_from_requirements():
    """When both run roots hold requirements.txt, packages with differing versions,
    and packages present on one side only, are recorded in environment_changes."""


def test_environment_changes_empty_without_requirements():
    """When either run root lacks requirements.txt, environment_changes is empty and
    the comparison still runs."""


def test_missing_directory_raises():
    """A path that does not exist or is a file raises ComparisonInputError."""


def test_directory_without_measures_raises():
    """A directory containing no parquet measures raises ComparisonInputError."""
