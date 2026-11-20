import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import tables
from vivarium.artifact import Artifact

from vivarium_gates_mncnh.compare import (
    Comparison,
    ComparisonError,
    _same_bytes,
    _same_frame,
    compare_artifacts,
    compare_notebooks,
    compare_results,
)


def shard(measure: str, draw: int, value: float = 0.1) -> pd.DataFrame:
    """A small results shard with the column types of a real psimulate shard."""
    return pd.DataFrame(
        {
            "measure": measure,
            "child_sex": pd.Categorical(["Female", "Male"], ordered=True),
            "input_draw": np.int64(draw),
            "value": [value, value * 3],
        }
    )


SHARDS = {
    "deaths/aaaa.parquet": shard("deaths", 0),
    "deaths/bbbb.parquet": shard("deaths", 1),
    "births/aaaa.parquet": shard("births", 0),
}


def write_files(root: Path, files: dict[str, bytes]) -> Path:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def write_shards(root: Path, shards: dict[str, pd.DataFrame]) -> Path:
    """Write shards the way psimulate does."""
    for name, frame in shards.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
    return root


def write_shards_other_metadata(root: Path, shards: dict[str, pd.DataFrame]) -> Path:
    """Write shards with the same content and different file metadata.

    The files differ in compression, row groups, statistics, string width,
    key-value metadata, and the pandas metadata, so their bytes differ too.
    """
    for name, frame in shards.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        table = table.cast(
            pa.schema(
                pa.field(f.name, pa.large_string()) if f.type == pa.string() else f
                for f in table.schema
            )
        ).replace_schema_metadata({"written_by": "another environment"})
        pq.write_table(
            table, path, compression="zstd", row_group_size=1, write_statistics=False
        )
    return root


def write_one(path: Path, frame: pd.DataFrame, **kwargs: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, **kwargs)
    return path


@pytest.fixture
def reference(tmp_path: Path) -> Path:
    return write_shards(tmp_path / "reference", SHARDS)


def test_results_same(reference: Path, tmp_path: Path) -> None:
    candidate = write_shards(tmp_path / "candidate", SHARDS)
    assert compare_results(reference, candidate) == Comparison()


def test_results_same_content_with_other_metadata(reference: Path, tmp_path: Path) -> None:
    candidate = write_shards_other_metadata(tmp_path / "candidate", SHARDS)
    name = "deaths/aaaa.parquet"
    assert not _same_bytes(reference / name, candidate / name)
    assert compare_results(reference, candidate) == Comparison()


def test_results_one_value_differs_with_other_metadata(
    reference: Path, tmp_path: Path
) -> None:
    changed = shard("deaths", 1, value=np.nextafter(0.1, 1.0))
    candidate = write_shards_other_metadata(
        tmp_path / "candidate", {**SHARDS, "deaths/bbbb.parquet": changed}
    )
    assert compare_results(reference, candidate) == Comparison(
        different=["deaths/bbbb.parquet"]
    )


def test_results_shard_on_one_side(reference: Path, tmp_path: Path) -> None:
    candidate_shards = {**SHARDS, "births/bbbb.parquet": shard("births", 1)}
    del candidate_shards["deaths/aaaa.parquet"]
    candidate = write_shards(tmp_path / "candidate", candidate_shards)
    assert compare_results(reference, candidate) == Comparison(
        only_in_reference=["deaths/aaaa.parquet"],
        only_in_candidate=["births/bbbb.parquet"],
    )


def test_results_ignore_files_outside_measure_shards(reference: Path, tmp_path: Path) -> None:
    candidate = write_shards(tmp_path / "candidate", SHARDS)
    write_files(
        candidate,
        {
            "flat.parquet": b"not a shard",
            "deaths/notes.txt": b"not parquet",
            "deaths/nested/cccc.parquet": b"too deep",
        },
    )
    assert compare_results(reference, candidate).same


def test_results_unreadable_shard_is_an_error(reference: Path, tmp_path: Path) -> None:
    candidate = write_shards(tmp_path / "candidate", SHARDS)
    write_files(candidate, {"deaths/aaaa.parquet": b"not parquet"})
    with pytest.raises(ComparisonError, match="cannot read .*deaths/aaaa.parquet"):
        compare_results(reference, candidate)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda f: f.assign(value=[np.nan, 0.3]), id="nan_in_same_place"),
        pytest.param(lambda f: f, id="identical"),
    ],
)
def test_same_frame_equal(tmp_path: Path, change: Any) -> None:
    frame = change(shard("deaths", 0))
    reference = write_one(tmp_path / "reference.parquet", frame)
    candidate = write_one(tmp_path / "candidate.parquet", frame, compression="gzip")
    assert _same_frame(reference, candidate)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(
            lambda f: f.assign(value=[np.nextafter(0.1, 1.0), 0.3]), id="value_one_ulp"
        ),
        pytest.param(lambda f: f.astype({"input_draw": "int32"}), id="int64_to_int32"),
        pytest.param(lambda f: f.rename(columns={"value": "count"}), id="column_renamed"),
        pytest.param(lambda f: f[["value", *f.columns[:-1]]], id="column_order"),
        pytest.param(lambda f: f.iloc[::-1].reset_index(drop=True), id="row_order"),
        pytest.param(lambda f: f.iloc[:1], id="row_removed"),
        pytest.param(
            lambda f: f.assign(child_sex=f["child_sex"].astype(str)), id="category_to_string"
        ),
    ],
)
def test_same_frame_different(tmp_path: Path, change: Any) -> None:
    frame = shard("deaths", 0)
    reference = write_one(tmp_path / "reference.parquet", frame)
    candidate = write_one(tmp_path / "candidate.parquet", change(frame))
    assert not _same_frame(reference, candidate)


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


def artifact_data() -> dict[str, Any]:
    """Artifact keys with the kinds of data a real artifact holds."""
    return {
        "population.structure": pd.DataFrame(
            {
                "sex": ["Female", "Male"],
                "year_start": [2023, 2023],
                "draw_0": [0.1, 0.3],
            }
        ).set_index(["sex", "year_start"]),
        "population.demographic_dimensions": ["sex", "year_start"],
        "cause.maternal_disorders.restrictions": {"yld_only": False, "age_start": 7.0},
    }


def write_artifact(path: Path, data: dict[str, Any]) -> Path:
    """Write keys with vivarium's Artifact, in the order given."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No artifact found")
        artifact = Artifact(path)
    for key, value in data.items():
        artifact.write(key, value)
    return path


def write_artifact_other_layout(path: Path, data: dict[str, Any]) -> Path:
    """Write the same keys in reverse order, with a key written and removed first.

    The data is the same, but the HDF file and the keyspace order differ.
    """
    artifact = Artifact(write_artifact(path, {"scratch.key": [1, 2, 3]}))
    for key, value in reversed(data.items()):
        artifact.write(key, value)
    artifact.remove("scratch.key")
    return path


def test_artifacts_same_with_other_name(tmp_path: Path) -> None:
    reference = write_artifact(tmp_path / "old/ethiopia.hdf", artifact_data())
    candidate = write_artifact(tmp_path / "new/renamed.hdf", artifact_data())
    assert compare_artifacts(reference, candidate) == Comparison()


def test_artifacts_same_data_with_other_layout(tmp_path: Path) -> None:
    reference = write_artifact(tmp_path / "reference.hdf", artifact_data())
    candidate = write_artifact_other_layout(tmp_path / "candidate.hdf", artifact_data())
    assert not _same_bytes(reference, candidate)
    assert compare_artifacts(reference, candidate) == Comparison()


def test_artifacts_one_value_differs_with_other_layout(tmp_path: Path) -> None:
    reference = write_artifact(tmp_path / "reference.hdf", artifact_data())
    data = artifact_data()
    data["population.structure"].iloc[0, 0] = np.nextafter(0.1, 1.0)
    candidate = write_artifact_other_layout(tmp_path / "candidate.hdf", data)
    assert compare_artifacts(reference, candidate) == Comparison(
        different=["population.structure"]
    )


def test_artifacts_key_on_one_side(tmp_path: Path) -> None:
    data = artifact_data()
    reference = write_artifact(tmp_path / "reference.hdf", data)
    del data["population.demographic_dimensions"]
    data["cause.other.restrictions"] = {"yld_only": True}
    candidate = write_artifact(tmp_path / "candidate.hdf", data)
    assert compare_artifacts(reference, candidate) == Comparison(
        only_in_reference=["population.demographic_dimensions"],
        only_in_candidate=["cause.other.restrictions"],
    )


@pytest.mark.parametrize(
    "key, value",
    [
        ("population.demographic_dimensions", ["sex", "year_end"]),
        ("cause.maternal_disorders.restrictions", {"yld_only": True, "age_start": 7.0}),
        (
            "population.structure",
            artifact_data()["population.structure"].astype({"draw_0": "float32"}),
        ),
    ],
    ids=["list", "dict", "float64_to_float32"],
)
def test_artifacts_key_data_differs(tmp_path: Path, key: str, value: Any) -> None:
    reference = write_artifact(tmp_path / "reference.hdf", artifact_data())
    candidate = write_artifact(tmp_path / "candidate.hdf", {**artifact_data(), key: value})
    assert compare_artifacts(reference, candidate) == Comparison(different=[key])


def test_artifact_with_no_keys_is_an_error_and_is_not_changed(tmp_path: Path) -> None:
    reference = write_artifact(tmp_path / "reference.hdf", artifact_data())
    empty = tmp_path / "empty.hdf"
    tables.open_file(str(empty), "w").close()
    before = empty.read_bytes()
    with pytest.raises(ComparisonError, match="candidate artifact has no keys"):
        compare_artifacts(reference, empty)
    assert empty.read_bytes() == before


def test_artifact_not_written_by_vivarium_is_an_error(tmp_path: Path) -> None:
    reference = write_artifact(tmp_path / "reference.hdf", artifact_data())
    not_hdf = write_files(tmp_path, {"not_hdf.hdf": b"data"}) / "not_hdf.hdf"
    pandas_hdf = tmp_path / "pandas.hdf"
    pd.DataFrame({"x": [1]}).to_hdf(pandas_hdf, key="x")
    for candidate in (not_hdf, pandas_hdf):
        with pytest.raises(ComparisonError, match="cannot read candidate artifact"):
            compare_artifacts(reference, candidate)


def test_missing_directory(reference: Path, tmp_path: Path) -> None:
    with pytest.raises(ComparisonError, match="candidate path is not a directory"):
        compare_results(reference, tmp_path / "missing")


def test_no_matching_files(reference: Path, tmp_path: Path) -> None:
    candidate = write_files(tmp_path / "candidate", {"metadata/aaaa.json": b"{}"})
    with pytest.raises(
        ComparisonError, match=r"no files match '\*/\*.parquet' in candidate path"
    ):
        compare_results(reference, candidate)


def test_artifact_must_be_a_file(tmp_path: Path) -> None:
    reference = write_artifact(tmp_path / "reference.hdf", artifact_data())
    with pytest.raises(ComparisonError, match="candidate artifact is not a file"):
        compare_artifacts(reference, tmp_path)


def test_unreadable_file_is_an_error(tmp_path: Path) -> None:
    candidate = write_files(tmp_path, {"a": b""}) / "a"
    with pytest.raises(ComparisonError, match="cannot read"):
        _same_bytes(tmp_path / "missing", candidate)
