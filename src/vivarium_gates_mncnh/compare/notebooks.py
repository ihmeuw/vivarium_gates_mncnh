"""Compare two notebook directories by the content of each executed notebook."""

import json
from pathlib import Path
from typing import Any

from vivarium_gates_mncnh.compare.base import Comparison, compare_items, find_files

NOTEBOOKS_PATTERN = "**/executed/*.ipynb"


def compare_notebooks(reference: Path, candidate: Path) -> Comparison:
    """Compare the executed notebooks in two notebook directories.

    Notebooks are paired by their path relative to each directory. Each pair must
    have the same cells in the same order, with the same type, source,
    execution count, and outputs. Notebook, cell, and output metadata are
    ignored, and so are cell IDs.

    Raises
    ------
    ComparisonError
        If either path is not a directory, has no executed notebooks, or has a
        notebook that cannot be read.
    """
    reference_files = find_files(reference, "reference", NOTEBOOKS_PATTERN)
    candidate_files = find_files(candidate, "candidate", NOTEBOOKS_PATTERN)
    return compare_items(
        reference_files,
        candidate_files,
        lambda name: _cells(reference_files[name]) == _cells(candidate_files[name]),
    )


def _cells(path: Path) -> list[dict[str, Any]]:
    """Return the content of each cell in a notebook file."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise ValueError("not a notebook: it has no list of cells")
    return [_cell_content(cell) for cell in notebook["cells"]]


def _cell_content(cell: dict[str, Any]) -> dict[str, Any]:
    """Return what a cell holds: its type, source, execution count, and outputs."""
    content = {key: value for key, value in cell.items() if key not in ("metadata", "id")}
    if "outputs" in cell:
        content["outputs"] = [_output_content(output) for output in cell["outputs"]]
    return content


def _output_content(output: dict[str, Any]) -> dict[str, Any]:
    """Return an output without its metadata."""
    return {key: value for key, value in output.items() if key != "metadata"}
