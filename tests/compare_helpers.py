"""Builders for the compare-tool tests: results directories, artifacts and notebooks.

Every builder writes only under the directory it is given, so tests stay
xdist-safe when handed a ``tmp_path``.
"""

import copy
import itertools
import warnings
from pathlib import Path
from typing import Any, Optional

import nbformat
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from vivarium.artifact import Artifact

from vivarium_gates_mncnh.tools.compare import PLOT_DATA_MIMETYPE

##############
# Results    #
##############

RESULTS_ID_COLUMNS = [
    "measure",
    "entity_type",
    "entity",
    "sub_entity",
    "sex",
    "age_group",
    "input_draw",
    "random_seed",
    "scenario",
]


def results_frame(measure: str, seed: int = 0) -> pd.DataFrame:
    """A small vivarium-like results frame with positive float values."""
    rows = list(
        itertools.product(
            ["Female", "Male"],
            ["early_neonatal", "late_neonatal"],
            [0, 1],
            [0, 1],
            ["baseline", "intervention"],
        )
    )
    frame = pd.DataFrame(
        rows, columns=["sex", "age_group", "input_draw", "random_seed", "scenario"]
    )
    frame.insert(0, "measure", measure)
    frame.insert(1, "entity_type", "cause")
    frame.insert(2, "entity", "maternal_disorders")
    frame.insert(3, "sub_entity", "total")
    frame["input_draw"] = frame["input_draw"].astype("int64")
    frame["random_seed"] = frame["random_seed"].astype("int64")
    rng = np.random.default_rng(seed)
    frame["value"] = rng.uniform(1.0, 100.0, len(frame))
    return frame[RESULTS_ID_COLUMNS + ["value"]]


def write_measure(
    results_dir: Path,
    name: str,
    frame: pd.DataFrame,
    layout: str = "flat",
    n_shards: int = 3,
    **write_kwargs: Any,
) -> None:
    """Write one measure as ``<name>.parquet`` (flat) or ``<name>/`` shards (sharded)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    if layout == "flat":
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(table, results_dir / f"{name}.parquet", **write_kwargs)
    elif layout == "sharded":
        shard_dir = results_dir / name
        shard_dir.mkdir()
        for i, shard in enumerate(np.array_split(np.arange(len(frame)), n_shards)):
            table = pa.Table.from_pandas(frame.iloc[shard], preserve_index=False)
            pq.write_table(table, shard_dir / f"shard_{i}.parquet", **write_kwargs)
    else:
        raise ValueError(layout)


def make_results_dir(
    run_root: Path,
    measures: dict[str, pd.DataFrame],
    layout: str = "flat",
    requirements: Optional[list[str]] = None,
    **write_kwargs: Any,
) -> Path:
    """Create ``run_root/results`` holding the measures; return the results dir.

    If ``requirements`` is given, it is written as the run root's
    ``requirements.txt``, as psimulate does.
    """
    results_dir = run_root / "results"
    results_dir.mkdir(parents=True)
    for name, frame in measures.items():
        write_measure(results_dir, name, frame, layout=layout, **write_kwargs)
    if requirements is not None:
        (run_root / "requirements.txt").write_text("\n".join(requirements) + "\n")
    return results_dir


def standard_measures() -> dict[str, pd.DataFrame]:
    return {
        "deaths": results_frame("deaths", seed=1),
        "births": results_frame("births", seed=2),
    }


##############
# Artifacts  #
##############


def artifact_frame(seed: int = 0, n_draws: int = 3) -> pd.DataFrame:
    """A small artifact-style frame indexed by demographics with draw columns."""
    ages = [(0.0, 0.01917808), (0.01917808, 0.07671233)]
    rows = [
        (sex, age_start, age_end, year, year + 1)
        for sex in ["Female", "Male"]
        for age_start, age_end in ages
        for year in [2023, 2024]
    ]
    index = pd.MultiIndex.from_tuples(
        rows, names=["sex", "age_start", "age_end", "year_start", "year_end"]
    )
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.uniform(0.1, 1.0, (len(index), n_draws)),
        index=index,
        columns=[f"draw_{i}" for i in range(n_draws)],
    )


def standard_artifact_data() -> dict[str, Any]:
    return {
        "metadata.locations": ["Ethiopia"],
        "population.structure": artifact_frame(seed=1),
        "cause.some_cause.prevalence": artifact_frame(seed=2),
    }


def make_artifact(path: Path, data: dict[str, Any]) -> Path:
    """Write a new artifact at ``path`` holding ``data``, in the dict's key order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        # Artifact warns that it is building a new file; that is the point here.
        warnings.simplefilter("ignore")
        artifact = Artifact(path)
    for key, value in data.items():
        artifact.write(key, value)
    return path


##############
# Notebooks  #
##############


def plot_data(line_y=(1.0, 2.0, 3.0), bar_height=2.5, point=(1.0, 4.0)) -> dict:
    """A PLOT_DATA_MIMETYPE payload holding one line, one bar and one scatter point."""
    return {
        "figures": [
            {
                "axes": [
                    {
                        "title": "A plot",
                        "xlabel": "x",
                        "ylabel": "y",
                        "lines": [{"label": "line", "x": [0.0, 1.0, 2.0], "y": list(line_y)}],
                        "patches": [
                            {
                                "label": "bars",
                                "x": -0.4,
                                "y": 0.0,
                                "width": 0.8,
                                "height": bar_height,
                            }
                        ],
                        "collections": [
                            {"label": "points", "offsets": [list(point)], "segments": []}
                        ],
                    }
                ]
            }
        ]
    }


def stdout(text: str):
    return nbformat.v4.new_output("stream", name="stdout", text=text)


def stderr(text: str):
    return nbformat.v4.new_output("stream", name="stderr", text=text)


def execute_result(text_plain: str, text_html: Optional[str] = None, count: int = 1):
    data = {"text/plain": text_plain}
    if text_html is not None:
        data["text/html"] = text_html
    return nbformat.v4.new_output("execute_result", data=data, execution_count=count)


def image_output(png: str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"):
    return nbformat.v4.new_output(
        "display_data",
        data={"image/png": png, "text/plain": "<Figure size 640x480 with 1 Axes>"},
    )


def plot_output(payload: Optional[dict] = None):
    return nbformat.v4.new_output(
        "display_data", data={PLOT_DATA_MIMETYPE: payload or plot_data()}
    )


def error_output(ename: str = "ValueError", evalue: str = "boom"):
    return nbformat.v4.new_output(
        "error",
        ename=ename,
        evalue=evalue,
        traceback=[f"{ename}: {evalue}"],
    )


def code_cell(source: str, outputs=(), execution_count: Optional[int] = None):
    cell = nbformat.v4.new_code_cell(source, execution_count=execution_count)
    cell.outputs = list(outputs)
    return cell


def standard_cells(with_plot_data: bool = True) -> list:
    """Cells of a small executed notebook.

    Code cells, in order: a print (stdout), a table (text/plain and text/html)
    and a plot (an image, plus plot data when ``with_plot_data``).
    """
    plot_outputs = [image_output()]
    if with_plot_data:
        plot_outputs.append(plot_output())
    return [
        nbformat.v4.new_markdown_cell("# A notebook"),
        code_cell("print(42)", [stdout("42\n")], execution_count=1),
        code_cell(
            "df",
            [execute_result("   a\n0  1", "<table><tr><td>1</td></tr></table>", count=2)],
            execution_count=2,
        ),
        code_cell("plt.plot([0, 1, 2], [1, 2, 3])", plot_outputs, execution_count=3),
    ]


PRINT_CELL, TABLE_CELL, PLOT_CELL = 1, 2, 3
"""Positions of the standard cells within the notebook's cell list."""


def write_notebook(
    directory: Path, name: str, cells: list, metadata: Optional[dict] = None
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.v4.new_notebook(cells=copy.deepcopy(cells))
    notebook.metadata.update(
        metadata
        or {
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
                "language": "python",
            }
        }
    )
    path = directory / name
    nbformat.write(notebook, path)
    return path


def read_notebook(path: Path):
    return nbformat.read(path, as_version=4)


def code_cells(notebook) -> list:
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def plot_data_outputs(notebook) -> list[dict]:
    """Every PLOT_DATA_MIMETYPE payload in the notebook, in order."""
    return [
        output["data"][PLOT_DATA_MIMETYPE]
        for cell in code_cells(notebook)
        for output in cell.get("outputs", [])
        if PLOT_DATA_MIMETYPE in output.get("data", {})
    ]


def stdout_text(notebook) -> str:
    return "".join(
        output.get("text", "")
        for cell in code_cells(notebook)
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream" and output.get("name") == "stdout"
    )


def kernel_available(kernel_name: str = "python3") -> bool:
    try:
        from jupyter_client.kernelspec import KernelSpecManager

        return kernel_name in KernelSpecManager().find_kernel_specs()
    except Exception:
        return False


def require_kernel() -> None:
    """Skip the calling test unless papermill and the python3 kernel are available."""
    pytest.importorskip("papermill")
    if not kernel_available():
        pytest.skip("python3 Jupyter kernel not available")


PLOTTING_NOTEBOOK_SOURCE = [
    "%matplotlib inline\nimport matplotlib.pyplot as plt",
    "print('ready')",
    (
        "fig, ax = plt.subplots()\n"
        "ax.plot([0, 1, 2], [1.5, 2.5, 4.5], label='line')\n"
        "ax.bar([10, 11], [3.25, 7.75], label='bars')\n"
        "ax.scatter([20.5, 21.5], [8.125, 9.375], label='points')\n"
        "plt.show()"
    ),
]


def write_source_notebook(directory: Path, name: str, sources: list[str]) -> Path:
    """Write an unexecuted notebook whose code cells hold ``sources``."""
    return write_notebook(directory, name, [code_cell(source) for source in sources])
