from pathlib import Path

import pytest

from vivarium_gates_mncnh.compare import (
    Comparison,
    ComparisonError,
    _same_bytes,
    compare_artifacts,
    compare_notebooks,
    compare_results,
)

SHARDS = {
    "deaths/aaaa.parquet": b"deaths draw 0",
    "deaths/bbbb.parquet": b"deaths draw 1",
    "births/aaaa.parquet": b"births draw 0",
}


def write_files(root: Path, files: dict[str, bytes]) -> Path:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


@pytest.fixture
def reference(tmp_path: Path) -> Path:
    return write_files(tmp_path / "reference", SHARDS)


def test_results_same(reference: Path, tmp_path: Path) -> None:
    candidate = write_files(tmp_path / "candidate", SHARDS)
    assert compare_results(reference, candidate) == Comparison()


def test_results_one_byte_differs(reference: Path, tmp_path: Path) -> None:
    candidate = write_files(
        tmp_path / "candidate", {**SHARDS, "deaths/bbbb.parquet": b"deaths draw 2"}
    )
    comparison = compare_results(reference, candidate)
    assert not comparison.same
    assert comparison == Comparison(different=["deaths/bbbb.parquet"])


def test_results_shard_on_one_side(reference: Path, tmp_path: Path) -> None:
    candidate_shards = {**SHARDS, "births/bbbb.parquet": b"births draw 1"}
    del candidate_shards["deaths/aaaa.parquet"]
    candidate = write_files(tmp_path / "candidate", candidate_shards)
    assert compare_results(reference, candidate) == Comparison(
        only_in_reference=["deaths/aaaa.parquet"],
        only_in_candidate=["births/bbbb.parquet"],
    )


def test_results_ignore_files_outside_measure_shards(reference: Path, tmp_path: Path) -> None:
    candidate = write_files(
        tmp_path / "candidate",
        {
            **SHARDS,
            "flat.parquet": b"not a shard",
            "deaths/notes.txt": b"not parquet",
            "deaths/nested/cccc.parquet": b"too deep",
        },
    )
    assert compare_results(reference, candidate).same


def test_notebooks_paired_by_relative_path(tmp_path: Path) -> None:
    notebooks = {"interactive/executed/a.ipynb": b"{1}", "results/executed/a.ipynb": b"{2}"}
    reference = write_files(tmp_path / "reference", notebooks)
    candidate = write_files(
        tmp_path / "candidate", {**notebooks, "results/executed/a.ipynb": b"{3}"}
    )
    assert compare_notebooks(reference, candidate) == Comparison(
        different=["results/executed/a.ipynb"]
    )


def test_notebooks_ignore_source_notebooks(tmp_path: Path) -> None:
    executed = {"interactive/executed/a.ipynb": b"{1}"}
    reference = write_files(
        tmp_path / "reference", {**executed, "interactive/a.ipynb": b"{1}"}
    )
    candidate = write_files(
        tmp_path / "candidate", {**executed, "interactive/a.ipynb": b"{2}"}
    )
    assert compare_notebooks(reference, candidate) == Comparison()


def test_artifacts_paired_regardless_of_name(tmp_path: Path) -> None:
    reference = write_files(tmp_path, {"old/ethiopia.hdf": b"data"}) / "old/ethiopia.hdf"
    same = write_files(tmp_path, {"new/renamed.hdf": b"data"}) / "new/renamed.hdf"
    changed = (
        write_files(tmp_path, {"changed/ethiopia.hdf": b"data!"}) / "changed/ethiopia.hdf"
    )
    assert compare_artifacts(reference, same) == Comparison()
    assert compare_artifacts(reference, changed).different == ["artifact"]


def test_missing_directory(reference: Path, tmp_path: Path) -> None:
    with pytest.raises(ComparisonError, match="candidate path is not a directory"):
        compare_results(reference, tmp_path / "missing")


def test_no_matching_files(reference: Path, tmp_path: Path) -> None:
    candidate = write_files(tmp_path / "candidate", {"metadata/aaaa.json": b"{}"})
    with pytest.raises(
        ComparisonError, match=r"no files match '\*/\*.parquet' in candidate path"
    ):
        compare_results(reference, candidate)


def test_artifact_must_be_a_file(reference: Path) -> None:
    with pytest.raises(ComparisonError, match="candidate artifact is not a file"):
        compare_artifacts(reference / "deaths/aaaa.parquet", reference)


def test_unreadable_file_is_an_error(tmp_path: Path) -> None:
    candidate = write_files(tmp_path, {"a": b""}) / "a"
    with pytest.raises(ComparisonError, match="cannot read"):
        _same_bytes(tmp_path / "missing", candidate)
