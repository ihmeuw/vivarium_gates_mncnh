import filecmp
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from vivarium.artifact import Artifact

from vivarium_gates_mncnh.compare import (
    Comparison,
    ComparisonError,
    compare_artifacts,
    compare_notebooks,
    compare_results,
)
from vivarium_gates_mncnh.compare.utilities import check_equality


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


def same_bytes(reference: Path, candidate: Path) -> bool:
    return filecmp.cmp(reference, candidate, shallow=False)


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
    assert not same_bytes(reference / name, candidate / name)
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
    assert compare_results(reference, candidate).equal


def test_results_unreadable_shard_is_an_error(reference: Path, tmp_path: Path) -> None:
    candidate = write_shards(tmp_path / "candidate", SHARDS)
    write_files(candidate, {"deaths/aaaa.parquet": b"not parquet"})
    with pytest.raises(ComparisonError, match="cannot read .*deaths/aaaa.parquet"):
        compare_results(reference, candidate)


def equal_shards(reference: Path, candidate: Path) -> bool:
    """Whether two shards read into equal frames, as ``compare_results`` reads them."""
    return check_equality(pd.read_parquet(reference), pd.read_parquet(candidate))


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda f: f.assign(value=[np.nan, 0.3]), id="nan_in_same_place"),
        pytest.param(lambda f: f, id="identical"),
    ],
)
def test_equal_frames(tmp_path: Path, change: Any) -> None:
    frame = change(shard("deaths", 0))
    reference = write_one(tmp_path / "reference.parquet", frame)
    candidate = write_one(tmp_path / "candidate.parquet", frame, compression="gzip")
    assert equal_shards(reference, candidate)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(
            lambda f: f.assign(value=[np.nextafter(0.1, 1.0), 0.3]), id="value_one_ulp"
        ),
        pytest.param(lambda f: f.astype({"input_draw": "int32"}), id="int64_to_int32"),
        pytest.param(lambda f: f.rename(columns={"value": "count"}), id="column_renamed"),
        pytest.param(lambda f: f.rename_axis("row"), id="index_renamed"),
        pytest.param(lambda f: f[["value", *f.columns[:-1]]], id="column_order"),
        pytest.param(lambda f: f.iloc[::-1].reset_index(drop=True), id="row_order"),
        pytest.param(lambda f: f.iloc[:1], id="row_removed"),
        pytest.param(
            lambda f: f.assign(child_sex=f["child_sex"].astype(str)), id="category_to_string"
        ),
    ],
)
def test_unequal_frames(tmp_path: Path, change: Any) -> None:
    frame = shard("deaths", 0)
    reference = write_one(tmp_path / "reference.parquet", frame)
    candidate = write_one(tmp_path / "candidate.parquet", change(frame))
    assert not equal_shards(reference, candidate)


def test_index_level_renamed(tmp_path: Path) -> None:
    frame = shard("deaths", 0).set_index(["measure", "child_sex"])
    reference = write_one(tmp_path / "reference.parquet", frame)
    candidate = write_one(
        tmp_path / "candidate.parquet", frame.rename_axis(["measure", "sex"])
    )
    assert not equal_shards(reference, candidate)


def test_series_renamed() -> None:
    series = pd.Series([0.1, 0.3], name="value")
    assert check_equality(series, series.copy())
    assert not check_equality(series, series.rename("count"))


def notebook(cells: list[dict[str, Any]], **metadata: Any) -> dict[str, Any]:
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def code_cell(source: str, output: str, count: int, cell_id: str = "a1") -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": count,
        "id": cell_id,
        "metadata": {},
        "source": source,
        "outputs": [
            {"name": "stdout", "output_type": "stream", "text": output},
            {
                "output_type": "display_data",
                "data": {"image/png": "iVBORw0KGgo=", "text/plain": "<Figure>"},
                "metadata": {"image/png": {"width": 400}},
            },
        ],
    }


def notebook_cells() -> list[dict[str, Any]]:
    """The cells of a small executed notebook."""
    return [
        {"cell_type": "markdown", "id": "m1", "metadata": {}, "source": "# Deaths\n"},
        code_cell("deaths = 1.5\nprint(deaths)\n", "1.5\n", 1, "c1"),
        code_cell("print(deaths * 2)\n", "3.0\n", 2, "c2"),
    ]


def as_other_layout(nb: dict[str, Any]) -> dict[str, Any]:
    """The same notebook with other metadata and cell IDs."""
    other = json.loads(json.dumps(nb))
    other["metadata"] = {
        "kernelspec": {"name": "python3"},
        "language_info": {"version": "3.12"},
    }
    other["nbformat_minor"] = 4
    for index, cell in enumerate(other["cells"]):
        cell["id"] = f"other-{index}"
        cell["metadata"] = {"tags": ["new"], "execution": {"time": "2026-11-20"}}
        for output in cell.get("outputs", []):
            if "metadata" in output:
                output["metadata"] = {"image/png": {"width": 640}}
    return other


def write_notebooks(
    root: Path, notebooks: dict[str, dict[str, Any]], indent: int = 1
) -> Path:
    for name, nb in notebooks.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(nb, indent=indent, sort_keys=indent == 1))
    return root


NOTEBOOKS = {
    "results/executed/maternal.ipynb": notebook(notebook_cells()),
    "interactive/executed/sim.ipynb": notebook(notebook_cells()[:2]),
}


def changed_output(nb: dict[str, Any]) -> dict[str, Any]:
    changed = json.loads(json.dumps(nb))
    changed["cells"][2]["outputs"][0]["text"] = "3.1\n"
    return changed


def test_notebooks_same(tmp_path: Path) -> None:
    reference = write_notebooks(tmp_path / "reference", NOTEBOOKS)
    candidate = write_notebooks(tmp_path / "candidate", NOTEBOOKS)
    assert compare_notebooks(reference, candidate) == Comparison()


def test_notebooks_same_content_with_other_layout(tmp_path: Path) -> None:
    reference = write_notebooks(tmp_path / "reference", NOTEBOOKS)
    candidate = write_notebooks(
        tmp_path / "candidate",
        {name: as_other_layout(nb) for name, nb in NOTEBOOKS.items()},
        indent=2,
    )
    name = "results/executed/maternal.ipynb"
    assert not same_bytes(reference / name, candidate / name)
    assert compare_notebooks(reference, candidate) == Comparison()


def test_notebooks_one_output_differs_with_other_layout(tmp_path: Path) -> None:
    reference = write_notebooks(tmp_path / "reference", NOTEBOOKS)
    name = "results/executed/maternal.ipynb"
    candidate_notebooks = {nb_name: as_other_layout(nb) for nb_name, nb in NOTEBOOKS.items()}
    candidate_notebooks[name] = as_other_layout(changed_output(NOTEBOOKS[name]))
    candidate = write_notebooks(tmp_path / "candidate", candidate_notebooks, indent=2)
    assert compare_notebooks(reference, candidate) == Comparison(different=[name])


def change_source(nb: dict[str, Any]) -> None:
    nb["cells"][1]["source"] = "deaths = 1.6\nprint(deaths)\n"


def change_count(nb: dict[str, Any]) -> None:
    nb["cells"][1]["execution_count"] = 5


def change_image(nb: dict[str, Any]) -> None:
    nb["cells"][1]["outputs"][1]["data"]["image/png"] = "iVBORw0KGgp="


def change_cell_type(nb: dict[str, Any]) -> None:
    nb["cells"][0]["cell_type"] = "raw"


def swap_cells(nb: dict[str, Any]) -> None:
    nb["cells"][1], nb["cells"][2] = nb["cells"][2], nb["cells"][1]


def drop_cell(nb: dict[str, Any]) -> None:
    del nb["cells"][2]


def add_output(nb: dict[str, Any]) -> None:
    nb["cells"][2]["outputs"].append(
        {"name": "stderr", "output_type": "stream", "text": "warning\n"}
    )


@pytest.mark.parametrize(
    "change",
    [
        change_source,
        change_count,
        change_image,
        change_cell_type,
        swap_cells,
        drop_cell,
        add_output,
    ],
)
def test_notebooks_content_differs(tmp_path: Path, change: Any) -> None:
    name = "results/executed/maternal.ipynb"
    reference = write_notebooks(tmp_path / "reference", NOTEBOOKS)
    nb = json.loads(json.dumps(NOTEBOOKS[name]))
    change(nb)
    candidate = write_notebooks(tmp_path / "candidate", {**NOTEBOOKS, name: nb})
    assert compare_notebooks(reference, candidate) == Comparison(different=[name])


def test_notebooks_ignore_source_notebooks(tmp_path: Path) -> None:
    reference = write_notebooks(
        tmp_path / "reference", {**NOTEBOOKS, "results/maternal.ipynb": notebook([])}
    )
    candidate = write_notebooks(
        tmp_path / "candidate",
        {**NOTEBOOKS, "results/maternal.ipynb": notebook(notebook_cells())},
    )
    assert compare_notebooks(reference, candidate) == Comparison()


def test_notebook_on_one_side(tmp_path: Path) -> None:
    reference = write_notebooks(tmp_path / "reference", NOTEBOOKS)
    candidate = write_notebooks(
        tmp_path / "candidate",
        {"results/executed/maternal.ipynb": NOTEBOOKS["results/executed/maternal.ipynb"]},
    )
    assert compare_notebooks(reference, candidate) == Comparison(
        only_in_reference=["interactive/executed/sim.ipynb"]
    )


@pytest.mark.parametrize(
    "content, message",
    [
        (b"not json", "cannot read results/executed/maternal.ipynb"),
        (b'{"no": "cells"}', "cannot read results/executed/maternal.ipynb: not a notebook"),
    ],
)
def test_unreadable_notebook_is_an_error(
    tmp_path: Path, content: bytes, message: str
) -> None:
    reference = write_notebooks(tmp_path / "reference", NOTEBOOKS)
    candidate = write_notebooks(tmp_path / "candidate", NOTEBOOKS)
    write_files(candidate, {"results/executed/maternal.ipynb": content})
    with pytest.raises(ComparisonError, match=message):
        compare_notebooks(reference, candidate)


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
    assert not same_bytes(reference, candidate)
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


def test_artifact_with_no_keys_is_an_error(tmp_path: Path) -> None:
    reference = write_artifact(tmp_path / "reference.hdf", artifact_data())
    candidate = write_artifact(tmp_path / "candidate.hdf", {})
    with pytest.raises(ComparisonError, match="candidate artifact has no keys"):
        compare_artifacts(reference, candidate)


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
