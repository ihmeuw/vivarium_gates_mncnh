"""Tests for vivarium_gates_mncnh.tools.compare.results.compare_results."""

import pandas as pd
import pytest

from vivarium_gates_mncnh.tools.compare import (
    ComparisonInputError,
    DifferenceKind,
    PackageChange,
    compare_results,
)
from vivarium_gates_mncnh.tools.compare import results as results_module

from .compare_helpers import (
    make_results_dir,
    results_frame,
    standard_measures,
    write_measure,
)


def _differences_for(result, measure):
    return [d for d in result.differences if measure in d.item]


def test_identical_results_compare_clean(tmp_path):
    """Two directories holding the same measures and values have no differences, and
    comparing them again gives the same result."""
    reference = make_results_dir(tmp_path / "reference", standard_measures())
    candidate = make_results_dir(tmp_path / "candidate", standard_measures())

    result = compare_results(reference, candidate)

    assert result.differences == []
    assert not result.moved
    assert result.n_compared == 2
    assert compare_results(reference, candidate) == result


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


def _measures_with(change):
    """standard_measures with deaths changed in one way."""
    measures = standard_measures()
    deaths = measures["deaths"].copy()
    if change == "values":
        deaths.loc[[0, 15, 30], "value"] += 1.0
    elif change == "fewer_rows":
        deaths = deaths.drop(index=[0, 15, 30])
    elif change == "more_rows":
        deaths = pd.concat([deaths, deaths.iloc[:3].assign(scenario="new")])
    elif change == "fewer_draws":
        deaths = deaths[deaths["input_draw"] != 1]
    elif change == "more_draws":
        deaths = pd.concat([deaths, deaths[deaths["input_draw"] == 1].assign(input_draw=2)])
    elif change == "float_draws":
        deaths = deaths.astype({"input_draw": "float64"})
    elif change == "new_column":
        deaths = deaths.assign(extra="x")
    return {**measures, "deaths": deaths}


def _fail_on_whole_measure(path):
    raise AssertionError(f"whole measure loaded: {path}")


@pytest.mark.parametrize(
    "change",
    ["none", "values", "fewer_rows", "more_rows", "fewer_draws", "more_draws", "float_draws"],
)
@pytest.mark.parametrize(
    "layout, candidate_prefix",
    [("sharded", "job"), ("per_draw", "renamed")],
    ids=["same_names", "other_names"],
)
def test_sharded_comparison_reports_as_flat(
    tmp_path, monkeypatch, change, layout, candidate_prefix
):
    """Shards are compared a group at a time without loading whole measures: paired by
    file name, or by the draw and seed they hold when the names differ (as after
    psimulate changes how it names them). The report is the same as comparing the
    whole measures, including changes spread over shards."""
    results = {}
    for side_layout in ["flat", layout]:
        if side_layout != "flat":
            monkeypatch.setattr(results_module, "_load_measure", _fail_on_whole_measure)
        reference = make_results_dir(
            tmp_path / side_layout / "reference", standard_measures(), layout=side_layout
        )
        candidate = make_results_dir(
            tmp_path / side_layout / "candidate",
            _measures_with(change),
            layout=side_layout,
            shard_prefix=candidate_prefix,
        )
        results[side_layout] = compare_results(reference, candidate)

    assert results[layout].differences == results["flat"].differences
    assert results[layout].moved == (change not in ("none", "float_draws"))


@pytest.mark.parametrize("change", ["none", "values", "fewer_draws", "new_column"])
@pytest.mark.parametrize(
    "layout, candidate_prefix",
    [("flat", "job"), ("per_draw", "job"), ("per_draw", "renamed"), ("sharded", "job")],
    ids=["flat", "same_names", "other_names", "arbitrary_shards"],
)
def test_result_is_the_same_for_any_number_of_workers(
    tmp_path, change, layout, candidate_prefix
):
    """Spreading the comparison over processes changes nothing in the result."""
    reference = make_results_dir(tmp_path / "reference", standard_measures(), layout=layout)
    candidate = make_results_dir(
        tmp_path / "candidate",
        _measures_with(change),
        layout=layout,
        shard_prefix=candidate_prefix,
    )

    in_process = compare_results(reference, candidate, workers=1)
    in_pool = compare_results(reference, candidate, workers=3)

    assert in_pool == in_process
    assert in_pool.moved == (change != "none")


def test_unreadable_shard_raises_from_a_worker(tmp_path):
    """A shard that cannot be read is an input error with workers too, not a crash."""
    reference = make_results_dir(
        tmp_path / "reference", standard_measures(), layout="per_draw"
    )
    candidate = make_results_dir(
        tmp_path / "candidate", standard_measures(), layout="per_draw"
    )
    next((candidate / "deaths").glob("*.parquet")).write_text("not parquet\n")

    with pytest.raises(ComparisonInputError, match="Could not read"):
        compare_results(reference, candidate, workers=3)


def test_schema_change_found_when_no_draw_is_shared(tmp_path):
    """A changed column is reported even when the two sides share no draw and seed."""
    measures = standard_measures()
    reference_measures = {
        name: frame[frame["input_draw"] == 0] for name, frame in measures.items()
    }
    candidate_measures = {
        name: frame[frame["input_draw"] == 1].assign(extra="x")
        for name, frame in measures.items()
    }
    reference = make_results_dir(
        tmp_path / "reference", reference_measures, layout="per_draw", shard_prefix="old"
    )
    candidate = make_results_dir(
        tmp_path / "candidate", candidate_measures, layout="per_draw", shard_prefix="new"
    )

    result = compare_results(reference, candidate)

    assert {d.kind for d in result.differences} == {DifferenceKind.SCHEMA}


def test_rows_moved_between_shards_compare_clean(tmp_path):
    """The same rows split differently over shards of the same names compare clean."""
    measures = standard_measures()
    shuffled = {
        name: frame.sample(frac=1.0, random_state=4) for name, frame in measures.items()
    }
    reference = make_results_dir(tmp_path / "reference", measures, layout="sharded")
    candidate = make_results_dir(tmp_path / "candidate", shuffled, layout="sharded")

    result = compare_results(reference, candidate)

    assert result.differences == []


@pytest.mark.parametrize(
    "candidate_categories",
    [["Female", "Male"], ["Male", "Female", "Other"], None],
    ids=["same_categories", "other_categories", "not_categorical"],
)
def test_categorical_key_columns_compare_clean(tmp_path, candidate_categories):
    """A categorical identifying column matches by value, whether or not the other side
    stores it with the same categories or as a categorical at all."""
    reference_measures = {
        name: frame.astype({"sex": pd.CategoricalDtype(["Female", "Male"])})
        for name, frame in standard_measures().items()
    }
    candidate_measures = standard_measures()
    if candidate_categories is not None:
        candidate_measures = {
            name: frame.astype({"sex": pd.CategoricalDtype(candidate_categories)})
            for name, frame in candidate_measures.items()
        }
    reference = make_results_dir(tmp_path / "reference", reference_measures)
    candidate = make_results_dir(tmp_path / "candidate", candidate_measures)

    result = compare_results(reference, candidate)

    assert result.differences == []


def test_missing_key_values_match(tmp_path):
    """Rows whose identifying columns hold missing values match the same rows."""
    measures = {
        name: frame.assign(sub_entity=frame["sub_entity"].where(frame.index % 2 == 0))
        for name, frame in standard_measures().items()
    }
    reference = make_results_dir(tmp_path / "reference", measures)
    candidate = make_results_dir(tmp_path / "candidate", measures, layout="sharded")

    result = compare_results(reference, candidate)

    assert result.differences == []


def _with_a_repeated_key(frame):
    """The frame plus a copy of its first row's key that holds a larger value."""
    copy = frame.iloc[[0]].assign(value=frame["value"].iloc[0] + 10.0)
    return pd.concat([frame, copy], ignore_index=True)


@pytest.mark.parametrize("change", ["rows_reversed", "copy_value_changed", "copy_dropped"])
def test_rows_sharing_a_key_pair_by_value(tmp_path, change):
    """Rows that share every identifying column pair up in the order of their values,
    so their order in the file does not matter, and a moved value or a missing copy
    is still reported."""
    deaths = _with_a_repeated_key(standard_measures()["deaths"])
    last = deaths.index[-1]
    if change == "rows_reversed":
        changed = deaths.iloc[::-1]
    elif change == "copy_value_changed":
        changed = deaths.copy()
        changed.loc[last, "value"] += 0.5
    else:
        changed = deaths.drop(index=last)
    reference = make_results_dir(tmp_path / "reference", {"deaths": deaths})
    candidate = make_results_dir(tmp_path / "candidate", {"deaths": changed})

    result = compare_results(reference, candidate)

    if change == "rows_reversed":
        assert result.differences == []
    elif change == "copy_value_changed":
        assert [d.kind for d in result.differences] == [DifferenceKind.VALUES]
        assert result.differences[0].value_diff.n_changed == 1
        assert result.differences[0].value_diff.max_abs_diff == pytest.approx(0.5)
    else:
        assert [(d.kind, d.detail) for d in result.differences] == [
            (DifferenceKind.ONLY_IN_REFERENCE, "1 row(s) only in reference")
        ]


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


def test_allowed_difference_is_reported(tmp_path):
    """The change the tolerance allowed is given for the value that moved most."""
    measures = standard_measures()
    reference = make_results_dir(tmp_path / "reference", measures)
    changed = measures["deaths"].copy()
    original = changed.loc[3, "value"]
    changed.loc[3, "value"] = original + 5.0
    candidate = make_results_dir(
        tmp_path / "candidate", {"deaths": changed, "births": measures["births"]}
    )

    result = compare_results(reference, candidate, rtol=1e-6, atol=1e-9)

    value_diff = result.differences[0].value_diff
    assert value_diff.max_abs_diff == pytest.approx(5.0)
    assert value_diff.allowed_abs_diff == pytest.approx(1e-9 + 1e-6 * abs(original))


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


def test_rounding_noise_is_within_default_tolerance(tmp_path):
    """A change of the size summing in another order makes (around 1e-16 relative)
    compares clean by default; it is still reported with rtol=0."""
    measures = standard_measures()
    reference = make_results_dir(tmp_path / "reference", measures)
    changed = measures["deaths"].copy()
    changed.loc[3, "value"] *= 1 + 4e-16
    candidate = make_results_dir(
        tmp_path / "candidate", {"deaths": changed, "births": measures["births"]}
    )
    assert changed.loc[3, "value"] != measures["deaths"].loc[3, "value"]

    assert not compare_results(reference, candidate).moved
    assert compare_results(reference, candidate, rtol=0.0).moved


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


@pytest.mark.parametrize(
    "patterns, expected", [(["deaths"], {"deaths"}), (["d*", "births"], {"deaths", "births"})]
)
def test_measures_selects_by_glob(tmp_path, patterns, expected):
    """Only the measures that match a pattern are compared."""
    measures = standard_measures()
    changed = {
        name: frame.assign(value=frame["value"] * 2) for name, frame in measures.items()
    }
    reference = make_results_dir(tmp_path / "reference", measures)
    candidate = make_results_dir(tmp_path / "candidate", changed)

    result = compare_results(reference, candidate, measures=patterns)

    assert result.n_compared == len(expected)
    assert {d.item for d in result.differences} == expected
    # The report says which measures were left out.
    patterns_text = ", ".join(map(repr, patterns))
    assert result.notes == [
        f"Compared only the measures matching {patterns_text}: {len(expected)} of 2."
    ]


def test_measures_pattern_matching_nothing_raises(tmp_path):
    """A pattern that matches no measure is an input error, not a clean comparison."""
    reference = make_results_dir(tmp_path / "reference", standard_measures())
    candidate = make_results_dir(tmp_path / "candidate", standard_measures())

    with pytest.raises(ComparisonInputError, match="dead"):
        compare_results(reference, candidate, measures=["deaths", "dead"])
