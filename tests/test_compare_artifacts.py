"""Tests for vivarium_gates_mncnh.tools.compare.artifacts.compare_artifacts."""

import warnings

import pandas as pd
import pytest

from vivarium_gates_mncnh.tools.compare import (
    ComparisonInputError,
    DifferenceKind,
    compare_artifacts,
)

from .compare_helpers import artifact_frame, make_artifact, standard_artifact_data


def _differences_for(result, key):
    return [d for d in result.differences if key in d.item]


def _pair(tmp_path, candidate_data=None, reference_data=None):
    reference = make_artifact(
        tmp_path / "reference.hdf", reference_data or standard_artifact_data()
    )
    candidate = make_artifact(
        tmp_path / "candidate.hdf", candidate_data or standard_artifact_data()
    )
    return reference, candidate


def test_identical_artifacts_compare_clean(tmp_path):
    """Two artifacts written separately with the same keys and data compare clean, and
    comparing them again gives the same result."""
    reference, candidate = _pair(tmp_path)

    result = compare_artifacts(reference, candidate)

    assert result.differences == []
    assert not result.moved
    assert result.n_compared >= len(standard_artifact_data())
    assert compare_artifacts(reference, candidate) == result


def test_wide_data_compares_without_warnings(tmp_path):
    """A key with many draw columns, as real artifacts have, compares without pandas
    warning that the frame is fragmented."""
    data = {
        **standard_artifact_data(),
        "cause.some_cause.prevalence": artifact_frame(n_draws=500),
    }
    reference, candidate = _pair(tmp_path, candidate_data=data, reference_data=data)

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        result = compare_artifacts(reference, candidate)

    assert result.differences == []


def test_reordered_data_compares_clean(tmp_path):
    """The same key written with its rows and columns in a different order compares
    clean."""
    data = standard_artifact_data()
    reordered = dict(data)
    for key in ["population.structure", "cause.some_cause.prevalence"]:
        frame = data[key].sample(frac=1.0, random_state=3)
        reordered[key] = frame[list(reversed(frame.columns))]
    reference, candidate = _pair(tmp_path, candidate_data=reordered, reference_data=data)

    result = compare_artifacts(reference, candidate)

    assert result.differences == []


def test_changed_values_name_key_and_magnitude(tmp_path):
    """A key whose values moved is reported with n_changed and max abs/rel diff;
    unchanged keys are not reported."""
    data = standard_artifact_data()
    changed = data["population.structure"].copy()
    original_a = changed.iloc[0, 0]
    original_b = changed.iloc[3, 2]
    changed.iloc[0, 0] = original_a + 1.0
    changed.iloc[3, 2] = original_b * 2.0
    reference, candidate = _pair(
        tmp_path, candidate_data={**data, "population.structure": changed}
    )

    result = compare_artifacts(reference, candidate)

    assert result.moved
    assert len(result.differences) == 1
    difference = result.differences[0]
    assert "population.structure" in difference.item
    assert difference.kind == DifferenceKind.VALUES
    assert difference.value_diff is not None
    assert difference.value_diff.n_changed == 2
    assert difference.value_diff.max_abs_diff == pytest.approx(max(1.0, original_b))
    assert difference.value_diff.max_rel_diff == pytest.approx(max(1.0 / original_a, 1.0))
    assert _differences_for(result, "cause.some_cause.prevalence") == []


@pytest.mark.parametrize("tolerance", [{"rtol": 1e-6}, {"atol": 1e-6}], ids=["rtol", "atol"])
def test_tolerance_is_applied(tmp_path, tolerance):
    """A small change is reported by default and not reported within rtol/atol."""
    data = standard_artifact_data()
    changed = data["cause.some_cause.prevalence"].copy()
    changed.iloc[2, 1] *= 1 + 1e-9
    reference, candidate = _pair(
        tmp_path, candidate_data={**data, "cause.some_cause.prevalence": changed}
    )

    by_default = compare_artifacts(reference, candidate)
    assert [d.kind for d in by_default.differences] == [DifferenceKind.VALUES]
    assert "cause.some_cause.prevalence" in by_default.differences[0].item

    within_tolerance = compare_artifacts(reference, candidate, **tolerance)
    assert not within_tolerance.moved


def test_key_only_on_one_side(tmp_path):
    """A key present in only one artifact is reported as only in reference or only in
    candidate."""
    data = standard_artifact_data()
    reference, candidate = _pair(
        tmp_path,
        reference_data={**data, "cause.reference_only.incidence_rate": artifact_frame(5)},
        candidate_data={**data, "cause.candidate_only.incidence_rate": artifact_frame(6)},
    )

    result = compare_artifacts(reference, candidate)

    assert result.moved
    assert [d.kind for d in _differences_for(result, "cause.reference_only")] == [
        DifferenceKind.ONLY_IN_REFERENCE
    ]
    assert [d.kind for d in _differences_for(result, "cause.candidate_only")] == [
        DifferenceKind.ONLY_IN_CANDIDATE
    ]
    for key in ["population.structure", "cause.some_cause.prevalence"]:
        assert _differences_for(result, key) == []


@pytest.mark.parametrize(
    "change", ["added_column", "dropped_column", "added_index_level", "dropped_index_level"]
)
def test_shape_change_reported(tmp_path, change):
    """A key whose index levels or columns differ is reported as a schema change."""
    data = standard_artifact_data()
    frame = data["population.structure"]
    if change == "added_column":
        changed = frame.assign(draw_3=frame["draw_0"])
    elif change == "dropped_column":
        changed = frame.drop(columns="draw_2")
    elif change == "added_index_level":
        changed = frame.assign(location="Ethiopia").set_index("location", append=True)
    else:
        changed = frame.droplevel("year_end")
    reference, candidate = _pair(
        tmp_path, candidate_data={**data, "population.structure": changed}
    )

    result = compare_artifacts(reference, candidate)

    assert result.moved
    assert DifferenceKind.SCHEMA in {
        d.kind for d in _differences_for(result, "population.structure")
    }
    assert _differences_for(result, "cause.some_cause.prevalence") == []


@pytest.mark.parametrize(
    "key, reference_value, candidate_value",
    [
        ("metadata.locations", ["Ethiopia"], ["Nigeria"]),
        ("metadata.locations", ["Ethiopia", "Nigeria"], ["Ethiopia"]),
        (
            "metadata.age_bins",
            {"start": [0, 7], "end": [7, 28]},
            {"start": [0, 7], "end": [7, 29]},
        ),
    ],
    ids=["changed_list", "shorter_list", "changed_dict"],
)
def test_non_dataframe_values_compared(tmp_path, key, reference_value, candidate_value):
    """A non-DataFrame value (e.g. a list) is compared for equality and reported when
    it differs."""
    data = standard_artifact_data()

    same_reference, same_candidate = _pair(
        tmp_path / "same",
        reference_data={**data, key: reference_value},
        candidate_data={**data, key: reference_value},
    )
    assert not compare_artifacts(same_reference, same_candidate).moved

    reference, candidate = _pair(
        tmp_path / "different",
        reference_data={**data, key: reference_value},
        candidate_data={**data, key: candidate_value},
    )
    result = compare_artifacts(reference, candidate)

    assert result.moved
    assert _differences_for(result, key)
    for other in ["population.structure", "cause.some_cause.prevalence"]:
        assert _differences_for(result, other) == []


@pytest.mark.parametrize("bad_side", ["reference", "candidate"])
def test_missing_path_raises_without_creating_file(tmp_path, bad_side):
    """A path that does not exist raises ComparisonInputError and no file is created
    there."""
    good = make_artifact(tmp_path / "good.hdf", standard_artifact_data())
    missing = tmp_path / "missing.hdf"
    paths = (missing, good) if bad_side == "reference" else (good, missing)

    with pytest.raises(ComparisonInputError):
        compare_artifacts(*paths)

    assert not missing.exists()

    # A directory is not an artifact either.
    directory = tmp_path / "a_directory.hdf"
    directory.mkdir()
    paths = (directory, good) if bad_side == "reference" else (good, directory)
    with pytest.raises(ComparisonInputError):
        compare_artifacts(*paths)


def test_unreadable_artifact_raises(tmp_path):
    """A file that is not an HDF artifact is an input error naming the file."""
    reference, _ = _pair(tmp_path)
    not_an_artifact = tmp_path / "not_an_artifact.hdf"
    not_an_artifact.write_text("not an artifact\n")

    with pytest.raises(ComparisonInputError, match="Could not read candidate artifact"):
        compare_artifacts(reference, not_an_artifact)


def test_key_that_cannot_be_loaded_raises(tmp_path):
    """A key the artifact lists but cannot load is an input error naming the key."""
    reference, candidate = _pair(tmp_path)
    with pd.HDFStore(candidate) as store:
        store.remove("/population/structure")

    with pytest.raises(
        ComparisonInputError, match="Could not load 'population.structure' from candidate"
    ):
        compare_artifacts(reference, candidate)


def test_inputs_not_modified(tmp_path):
    """Comparing does not modify either artifact file."""
    data = standard_artifact_data()
    changed = data["population.structure"] * 1.1
    reference, candidate = _pair(
        tmp_path, candidate_data={**data, "population.structure": changed}
    )
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (reference, candidate)
    }
    files_before = sorted(tmp_path.iterdir())

    compare_artifacts(reference, candidate)
    compare_artifacts(reference, candidate, rtol=0.5)

    after = {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (reference, candidate)
    }
    assert after == before
    assert sorted(tmp_path.iterdir()) == files_before


def test_keys_selects_by_glob(tmp_path):
    """Only the keys that match a pattern are compared."""
    data = standard_artifact_data()
    changed = {
        **data,
        "population.structure": data["population.structure"] * 2,
        "cause.some_cause.prevalence": data["cause.some_cause.prevalence"] * 2,
    }
    reference = make_artifact(tmp_path / "reference.hdf", data)
    candidate = make_artifact(tmp_path / "candidate.hdf", changed)

    result = compare_artifacts(reference, candidate, keys=["cause.*"])

    assert result.n_compared == 1
    assert [d.item for d in result.differences] == ["cause.some_cause.prevalence"]
    (note,) = result.notes
    assert note.startswith("Compared only the keys matching 'cause.*': 1 of ")


def test_keys_pattern_matching_nothing_raises(tmp_path):
    """A pattern that matches no key is an input error, not a clean comparison."""
    data = standard_artifact_data()
    reference = make_artifact(tmp_path / "reference.hdf", data)
    candidate = make_artifact(tmp_path / "candidate.hdf", data)

    with pytest.raises(ComparisonInputError, match="risk"):
        compare_artifacts(reference, candidate, keys=["risk.*"])
