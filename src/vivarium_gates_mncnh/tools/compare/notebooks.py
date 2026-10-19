"""Execute notebooks and compare their outputs."""

from pathlib import Path
from typing import Any, Optional, Union

from loguru import logger

from vivarium_gates_mncnh.tools.compare.core import ComparisonResult

PLOT_DATA_MIMETYPE = "application/vnd.vivarium.plot-data+json"
"""MIME type of the output that records a cell's plotted data.

Its data is a JSON object of the form::

    {"figures": [{"axes": [{
        "title": str, "xlabel": str, "ylabel": str,
        "lines": [{"label": str, "x": [...], "y": [...]}],
        "patches": [{"label": str, "x": float, "y": float,
                     "width": float, "height": float}],
        "collections": [{"label": str, "offsets": [[x, y], ...],
                         "segments": [[[x, y], ...], ...]}],
    }]}]}

``lines`` covers line plots, horizontal and vertical reference lines and
errorbar markers and caps; ``patches`` covers bars and histogram bins;
``collections`` covers scatter points (``offsets``) and errorbar whiskers
(``segments``).
"""


def discover_notebook_paths(notebook_directory: Path) -> list[Path]:
    """
    Discover all Jupyter notebook files in the notebook directory.

    Searches only the top level of the directory (non-recursive) and filters
    out checkpoint files.

    Returns:
        List of Path objects for discovered notebooks
    """
    # Filter out checkpoint files and nbdime merge conflict artifacts (e.g. *_BACKUP_*, *_BASE_*, *_LOCAL_*, *_REMOTE_*)
    nbdime_suffixes = ("_BACKUP_", "_BASE_", "_LOCAL_", "_REMOTE_")
    notebooks = [
        nb
        for nb in notebook_directory.glob("*.ipynb")
        if ".ipynb_checkpoints" not in str(nb)
        and not any(suffix in nb.stem for suffix in nbdime_suffixes)
    ]

    logger.info(f"Found {len(notebooks)} notebook(s) in {notebook_directory}")

    return sorted(notebooks)


def execute_notebooks(
    notebook_directory: Union[str, Path],
    output_directory: Union[str, Path],
    kernel_name: str = "python3",
    parameters: Optional[dict[str, Any]] = None,
) -> dict[str, Optional[str]]:
    """Execute every notebook in a directory, recording what each one plots.

    Each notebook found by :func:`discover_notebook_paths` is executed with
    papermill and written under the same name to ``output_directory``, which
    is created if needed. The source notebooks are never modified. While a
    notebook runs, every code cell that displays one or more matplotlib
    figures gets one extra ``display_data`` output of type
    :data:`PLOT_DATA_MIMETYPE` recording the data plotted in those figures.

    A notebook that fails is still written, up to and including the failing
    cell, and the remaining notebooks still run.

    Args:
        notebook_directory: Directory of notebooks to execute.
        output_directory: Directory to write the executed notebooks to.
        kernel_name: Jupyter kernel to execute with.
        parameters: Parameters papermill injects into each notebook.

    Returns:
        A mapping from each notebook's file name to None if it ran to
        completion, or to a short description of the error if it failed.

    Raises:
        ComparisonInputError: If ``notebook_directory`` is not a directory or
            contains no notebooks.
    """
    raise NotImplementedError


def compare_notebooks(
    reference_dir: Union[str, Path],
    candidate_dir: Union[str, Path],
    rtol: float = 0.0,
    atol: float = 0.0,
) -> ComparisonResult:
    """Compare the outputs of two directories of executed notebooks.

    Notebooks are paired by file name and cells by position. For each code
    cell, its outputs are compared: stdout, ``text/plain``, ``text/html`` and
    error outputs must match exactly, and :data:`PLOT_DATA_MIMETYPE` outputs
    are compared by value, numbers within tolerance and everything else
    exactly. Stderr, execution counts, metadata, images and widget outputs are
    ignored.

    Reported as differences:

    - a notebook present in only one directory;
    - a notebook whose cell sources differ (schema change);
    - a notebook that raised an error on only one side, or a different error
      (execution failed), naming the cell;
    - a cell whose text outputs changed (output changed);
    - a cell whose plotted data changed (plot data changed).

    When one side of a cell has plot data and the other has none, the plots
    are not compared and a note says so; this is the case when the reference
    is a committed ``executed/`` notebook rather than one produced by
    :func:`execute_notebooks`.

    Args:
        reference_dir: Directory of reference executed notebooks.
        candidate_dir: Directory of candidate executed notebooks.
        rtol: Relative tolerance for plotted values; see
            :func:`~vivarium_gates_mncnh.tools.compare.core.compare_values`.
        atol: Absolute tolerance for plotted values.

    Returns:
        The comparison, with one item compared per notebook on either side.

    Raises:
        ComparisonInputError: If either path is not a directory, or either
            directory contains no notebooks.
    """
    raise NotImplementedError
