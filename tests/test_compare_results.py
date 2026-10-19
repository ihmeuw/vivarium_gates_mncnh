"""Tests for vivarium_gates_mncnh.tools.compare.results.compare_results."""

import pytest

from vivarium_gates_mncnh.tools.compare import (
    ComparisonInputError,
    DifferenceKind,
    PackageChange,
    compare_results,
)

from .compare_helpers import (
    make_results_dir,
    results_frame,
    standard_measures,
    write_measure,
)


def _differences_for(result, measure):
    return [d for d in result.differences if measure in d.item]


def test_identical_results_compare_clean(tmp_path):
    """Two directories holding the same measures and values have no differences."""
    reference = make_results_dir(tmp_path / "reference", standard_measures())
    candidate = make_results_dir(tmp_path / "candidate", standard_measures())

    result = compare_results(reference, candidate)

    assert result.differences == []
    assert not result.moved
    assert result.n_compared == 2


@pytest.mark.parametrize("compression", ["gzip", "zstd", "none"])
def test_rewritten_files_compare_clean(tmp_path, compression):
    """The same data rewritten with different compression, row-group size, row order
    and column order compares clean."""
    measures = standard_measures()
    reference = make_results_dir(tmp_path / "reference", measures, compression="snappy")

    candidate = tmp_path / "candidate" / "results"
    for name, frame in measures.items():
        rewritten = frame.sample(frac=1.0, random_state=7)
        rewritten = rewritten[list(reversed(rewritten.columns))]
        write_measure(candidate, name, rewritten, compression=compression, row_group_size=5)

    result = compare_results(reference, candidate)

    assert result.differences == []
    assert result.n_compared == 2


@pytest.mark.parametrize(
    "reference_layout, candidate_layout",
    [("flat", "sharded"), ("sharded", "flat"), ("sharded", "sharded")],
)
def test_flat_and_sharded_layouts_compare_clean(tmp_path, reference_layout, candidate_layout):
    """A measure as <measure>.parquet on one side and as <measure>/ shards on the
    other, holding the same rows, compares clean."""
    reference = make_results_dir(
        tmp_path / "reference", standard_measures(), layout=reference_layout
    )
    candidate = make_results_dir(
        tmp_path / "candidate", standard_measures(), layout=candidate_layout
    )

    result = compare_results(reference, candidate)

    assert result.differences == []
    # Each measure is one item whichever layout holds it.
    assert result.n_compared == 2


def test_rerunning_on_unchanged_inputs_is_stable(tmp_path):
    """Comparing the same pair twice gives the same, clean result."""
    reference = make_results_dir(tmp_path / "reference", standard_measures())
    candidate = make_results_dir(
        tmp_path / "candidate", standard_measures(), layout="sharded"
    )

    first = compare_results(reference, candidate)
    second = compare_results(reference, candidate)

    assert first == second
    assert not first.moved
    assert not second.moved


def test_changed_values_name_measure_and_magnitude(tmp_path):
    """A moved value is reported against its measure with n_changed and max abs/rel diff;
    unchanged measures are not reported."""
    measures = standard_measures()
    reference = make_results_dir(tmp_path / "reference", measures)

    changed = measures["deaths"].copy()
    original_0 = changed.loc[0, "value"]
    original_5 = changed.loc[5, "value"]
    changed.loc[0, "value"] = original_0 + 2.0
    changed.loc[5, "value"] = original_5 * 1.5
    candidate = make_results_dir(
        tmp_path / "candidate", {"deaths": changed, "births": measures["births"]}
    )

    result = compare_results(reference, candidate)

    assert result.moved
    assert len(result.differences) == 1
    difference = result.differences[0]
    assert "deaths" in difference.item
    assert difference.kind == DifferenceKind.VALUES
    assert difference.value_diff is not None
    assert difference.value_diff.n_changed == 2
    assert difference.value_diff.max_abs_diff == pytest.approx(max(2.0, 0.5 * original_5))
    assert difference.value_diff.max_rel_diff == pytest.approx(max(2.0 / original_0, 0.5))
    assert _differences_for(result, "births") == []


@pytest.mark.parametrize("tolerance", [{"rtol": 1e-6}, {"atol": 1e-3}], ids=["rtol", "atol"])
def test_tolerance_is_applied(tmp_path, tolerance):
    """A small change is reported by default and not reported within rtol/atol."""
    measures = standard_measures()
    reference = make_results_dir(tmp_path / "reference", measures)
    changed = measures["deaths"].copy()
    changed.loc[3, "value"] *= 1 + 1e-9
    candidate = make_results_dir(
        tmp_path / "candidate", {"deaths": changed, "births": measures["births"]}
    )

    by_default = compare_results(reference, candidate)
    assert by_default.moved
    assert [d.kind for d in _differences_for(by_default, "deaths")] == [DifferenceKind.VALUES]

    within_tolerance = compare_results(reference, candidate, **tolerance)
    assert not within_tolerance.moved


def test_measure_only_on_one_side(tmp_path):
    """A measure present in only one directory is reported as only in reference or
    only in candidate."""
    reference = make_results_dir(
        tmp_path / "reference",
        {"deaths": results_frame("deaths", 1), "births": results_frame("births", 2)},
    )
    candidate = make_results_dir(
        tmp_path / "candidate",
        {"deaths": results_frame("deaths", 1), "ylls": results_frame("ylls", 3)},
    )

    result = compare_results(reference, candidate)

    assert result.moved
    assert [d.kind for d in _differences_for(result, "births")] == [
        DifferenceKind.ONLY_IN_REFERENCE
    ]
    assert [d.kind for d in _differences_for(result, "ylls")] == [
        DifferenceKind.ONLY_IN_CANDIDATE
    ]
    assert _differences_for(result, "deaths") == []


@pytest.mark.parametrize("side", ["reference", "candidate"])
def test_rows_only_on_one_side(tmp_path, side):
    """Rows present on only one side of a measure are reported against that measure."""
    measures = standard_measures()
    fewer_rows = {**measures, "deaths": measures["deaths"].drop(index=[0, 1])}
    if side == "reference":
        reference_measures, candidate_measures = measures, fewer_rows
        expected_kind = DifferenceKind.ONLY_IN_REFERENCE
    else:
        reference_measures, candidate_measures = fewer_rows, measures
        expected_kind = DifferenceKind.ONLY_IN_CANDIDATE
    reference = make_results_dir(tmp_path / "reference", reference_measures)
    candidate = make_results_dir(tmp_path / "candidate", candidate_measures)

    result = compare_results(reference, candidate)

    assert result.moved
    deaths_differences = _differences_for(result, "deaths")
    assert deaths_differences
    # The rows held in common are unchanged, so only the unmatched rows are reported.
    assert {d.kind for d in deaths_differences} == {expected_kind}
    assert _differences_for(result, "births") == []


@pytest.mark.parametrize("change", ["added_column", "dropped_column", "renamed_column"])
def test_schema_change_reported(tmp_path, change):
    """A measure whose identifying columns differ is reported as a schema change."""
    measures = standard_measures()
    changed = measures["deaths"].copy()
    if change == "added_column":
        changed.insert(0, "location", "Ethiopia")
    elif change == "dropped_column":
        changed = changed.drop(columns="sub_entity")
    else:
        changed = changed.rename(columns={"scenario": "intervention_scenario"})
    reference = make_results_dir(tmp_path / "reference", measures)
    candidate = make_results_dir(
        tmp_path / "candidate", {"deaths": changed, "births": measures["births"]}
    )

    result = compare_results(reference, candidate)

    assert result.moved
    assert DifferenceKind.SCHEMA in {d.kind for d in _differences_for(result, "deaths")}
    assert _differences_for(result, "births") == []


@pytest.mark.parametrize(
    "reference_names, candidate_names, expected",
    [
        (["deaths", "births"], ["deaths", "births"], 2),
        (["deaths", "births", "ylls"], ["births", "ylls", "ylds"], 4),
        (["deaths"], ["births"], 2),
    ],
)
def test_n_compared_counts_measures(tmp_path, reference_names, candidate_names, expected):
    """n_compared counts each measure appearing on either side once."""
    reference = make_results_dir(
        tmp_path / "reference",
        {name: results_frame(name, i) for i, name in enumerate(reference_names)},
    )
    candidate = make_results_dir(
        tmp_path / "candidate",
        {name: results_frame(name, i) for i, name in enumerate(candidate_names)},
        layout="sharded",
    )

    result = compare_results(reference, candidate)

    assert result.n_compared == expected


def test_environment_changes_from_requirements(tmp_path):
    """When both run roots hold requirements.txt, packages with differing versions,
    and packages present on one side only, are recorded in environment_changes."""
    reference = make_results_dir(
        tmp_path / "reference",
        standard_measures(),
        requirements=["numpy==1.26.4", "oldpkg==1.0", "pandas==2.2.1", "scipy==1.13.0"],
    )
    candidate = make_results_dir(
        tmp_path / "candidate",
        standard_measures(),
        requirements=["newpkg==0.1", "numpy==1.26.4", "pandas==2.2.2", "scipy==1.13.0"],
    )

    result = compare_results(reference, candidate)

    assert set(result.environment_changes) == {
        PackageChange("pandas", "2.2.1", "2.2.2"),
        PackageChange("oldpkg", "1.0", None),
        PackageChange("newpkg", None, "0.1"),
    }
    # Environment changes are recorded, not counted as movement.
    assert not result.moved


@pytest.mark.parametrize("with_requirements", ["reference", "candidate", "neither"])
def test_environment_changes_empty_without_requirements(tmp_path, with_requirements):
    """When either run root lacks requirements.txt, environment_changes is empty and
    the comparison still runs."""
    requirements = ["pandas==2.2.1"]
    measures = standard_measures()
    changed = {**measures, "deaths": measures["deaths"].assign(value=lambda df: df.value + 1)}
    reference = make_results_dir(
        tmp_path / "reference",
        measures,
        requirements=requirements if with_requirements == "reference" else None,
    )
    candidate = make_results_dir(
        tmp_path / "candidate",
        changed,
        requirements=["pandas==2.2.2"] if with_requirements == "candidate" else None,
    )

    result = compare_results(reference, candidate)

    assert result.environment_changes == []
    assert result.n_compared == 2
    assert [d.kind for d in _differences_for(result, "deaths")] == [DifferenceKind.VALUES]


@pytest.mark.parametrize(
    "bad_side, kind",
    [
        ("reference", "missing"),
        ("candidate", "missing"),
        ("reference", "file"),
        ("candidate", "file"),
    ],
)
def test_missing_directory_raises(tmp_path, bad_side, kind):
    """A path that does not exist or is a file raises ComparisonInputError."""
    good = make_results_dir(tmp_path / "good", standard_measures())
    if kind == "missing":
        bad = tmp_path / "does_not_exist" / "results"
    else:
        bad = good / "deaths.parquet"
        assert bad.is_file()
    paths = (bad, good) if bad_side == "reference" else (good, bad)

    with pytest.raises(ComparisonInputError):
        compare_results(*paths)


@pytest.mark.parametrize("bad_side", ["reference", "candidate"])
@pytest.mark.parametrize("contents", ["empty", "non_parquet_files"])
def test_directory_without_measures_raises(tmp_path, bad_side, contents):
    """A directory containing no parquet measures raises ComparisonInputError."""
    good = make_results_dir(tmp_path / "good", standard_measures())
    bad = tmp_path / "bad" / "results"
    bad.mkdir(parents=True)
    if contents == "non_parquet_files":
        (bad / "notes.txt").write_text("not a measure\n")
        (bad / "deaths.csv").write_text("value\n1.0\n")
    paths = (bad, good) if bad_side == "reference" else (good, bad)

    with pytest.raises(ComparisonInputError):
        compare_results(*paths)


@pytest.mark.parametrize("float_side", ["reference", "candidate"])
def test_int_and_float_key_columns_compare_clean(tmp_path, float_side):
    """An identifying column stored as integers on one side and as floats on the other,
    with the same values, compares clean rather than as rows on one side only."""
    int_measures = standard_measures()
    assert str(int_measures["deaths"]["input_draw"].dtype) == "int64"
    float_measures = {
        name: frame.astype({"input_draw": "float64", "random_seed": "float64"})
        for name, frame in standard_measures().items()
    }
    if float_side == "reference":
        reference_measures, candidate_measures = float_measures, int_measures
    else:
        reference_measures, candidate_measures = int_measures, float_measures
    reference = make_results_dir(tmp_path / "reference", reference_measures)
    candidate = make_results_dir(tmp_path / "candidate", candidate_measures)

    result = compare_results(reference, candidate)

    assert result.differences == []
    assert not result.moved
    assert result.n_compared == 2
