"""Execute notebooks and compare their outputs."""

import inspect
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from loguru import logger

from vivarium_gates_mncnh.tools.compare.core import (
    ComparisonInputError,
    ComparisonResult,
    Difference,
    DifferenceKind,
    ValueDiff,
    compare_values,
)

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

The coordinates, and how many figures, axes, lines, patches and collections
there are, are the plotted data that :func:`compare_notebooks` compares. The
presentation fields (``title``, ``xlabel``, ``ylabel`` and each ``label``)
are never compared; the axes ``title`` is used only to name a plot that
moved in the report.
"""

_PLOT_PRESENTATION_KEYS = frozenset({"title", "xlabel", "ylabel", "label"})
"""Keys of :data:`PLOT_DATA_MIMETYPE` objects that are presentation, not data."""


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
    is created if needed; any copy of a notebook already there is removed
    before it runs, so a failed run never leaves an earlier run's outputs in
    place. Notebooks run with ``output_directory`` as their working directory,
    so any files they write land beside their outputs; the source notebooks
    are never modified. While a
    notebook runs, every code cell that displays one or more matplotlib
    figures gets one extra ``display_data`` output of type
    :data:`PLOT_DATA_MIMETYPE` recording the data plotted in those figures.

    A notebook that fails is still written, up to and including the failing
    cell, and the remaining notebooks still run.

    When ``notebook_directory`` is inside a checkout of this repository, the
    kernel must import ``vivarium_gates_mncnh`` from that same checkout;
    otherwise the notebooks would run some other checkout's code and any
    difference found would mean nothing, so nothing is run.

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
            contains no notebooks, if ``output_directory`` is the notebook
            directory, or if the kernel imports ``vivarium_gates_mncnh`` from a
            different checkout than the one the notebooks are in.
    """
    notebook_directory, output_directory = Path(notebook_directory), Path(output_directory)
    notebooks = _find_notebooks(notebook_directory, "notebook")
    if output_directory.resolve() == notebook_directory.resolve():
        raise ComparisonInputError(
            f"Output directory must differ from the notebook directory: {output_directory}"
        )
    _check_kernel_package(notebook_directory, kernel_name)
    output_directory.mkdir(parents=True, exist_ok=True)

    outcomes = {}
    with tempfile.TemporaryDirectory() as staging_directory:
        for name, source in notebooks.items():
            outcomes[name] = _execute_notebook(
                source,
                Path(staging_directory) / name,
                output_directory / name,
                kernel_name,
                parameters,
            )
    return outcomes


_SETUP_CELL_TAG = "vivarium-plot-capture"
_WIDGET_MIMETYPE = "application/vnd.jupyter.widget-view+json"
_TEXT_MIMETYPES = ("text/plain", "text/html")
_MEMORY_ADDRESS = re.compile(r"\b0x[0-9a-fA-F]{6,}\b")


def _find_notebooks(directory: Path, side: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise ComparisonInputError(f"{side} directory is not a directory: {directory}")
    notebooks = {path.name: path for path in discover_notebook_paths(directory)}
    if not notebooks:
        raise ComparisonInputError(f"No notebooks found in {side} directory {directory}")
    return notebooks


def _check_kernel_package(notebook_directory: Path, kernel_name: str) -> None:
    """Fail if the kernel would import this package from a different checkout.

    The notebooks locate the model through ``vivarium_gates_mncnh.__file__``,
    so they run whichever checkout the kernel's environment imports. The check
    is skipped when the notebooks are not inside a checkout, or when the
    kernel cannot be found or cannot import the package at all (each notebook
    then reports that failure itself).
    """
    from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel

    package = Path("src") / "vivarium_gates_mncnh"
    repo = next(
        (
            parent
            for parent in notebook_directory.resolve().parents
            if (parent / package / "__init__.py").is_file()
        ),
        None,
    )
    if repo is None:
        return
    try:
        python = KernelSpecManager().get_kernel_spec(kernel_name).argv[0]
    except NoSuchKernel:
        return
    # Jupyter launches a bare "python" kernel with the current interpreter.
    if python in ("python", "python3"):
        python = sys.executable
    found = subprocess.run(
        [python, "-c", "import vivarium_gates_mncnh; print(vivarium_gates_mncnh.__file__)"],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if found.returncode != 0:
        return
    imported = Path(found.stdout.strip().splitlines()[-1]).resolve().parent
    expected = (repo / package).resolve()
    if imported != expected:
        raise ComparisonInputError(
            f"Kernel '{kernel_name}' imports vivarium_gates_mncnh from {imported}, "
            f"not from {expected}, the checkout the notebooks are in. Install the "
            "package from this checkout, or put its src directory on PYTHONPATH."
        )


def _execute_notebook(
    source: Path,
    staged: Path,
    output: Path,
    kernel_name: str,
    parameters: Optional[dict[str, Any]],
) -> Optional[str]:
    """Execute a copy of ``source`` with plot capture, writing it to ``output``.

    Returns None on success, or a short description of the error.
    """
    import nbformat
    import papermill as pm
    from papermill.exceptions import PapermillExecutionError

    # Never leave an earlier run's outputs in place if this run fails before writing.
    output.unlink(missing_ok=True)
    notebook = nbformat.read(source, as_version=4)
    setup_cell = nbformat.v4.new_code_cell(
        source=_plot_capture_source(), metadata={"tags": [_SETUP_CELL_TAG]}
    )
    if notebook.nbformat_minor < 5:
        setup_cell.pop("id", None)
    notebook.cells.insert(0, setup_cell)
    nbformat.write(notebook, staged)

    logger.info(f"Executing {source.name} with kernel {kernel_name}")
    error = None
    try:
        pm.execute_notebook(
            str(staged),
            str(output),
            parameters=parameters,
            kernel_name=kernel_name,
            cwd=str(output.parent),
            progress_bar=False,
        )
    except PapermillExecutionError as e:
        error = f"{e.ename}: {e.evalue}"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    if output.is_file():
        executed = nbformat.read(output, as_version=4)
        executed.cells = [
            cell
            for cell in executed.cells
            if _SETUP_CELL_TAG not in cell.get("metadata", {}).get("tags", [])
        ]
        nbformat.write(executed, output)

    if error is None:
        logger.info(f"{source.name} ran to completion")
        return None
    error = error.strip().splitlines()[0] if error.strip() else error
    error = error if len(error) <= 200 else error[:197] + "..."
    logger.error(f"{source.name} failed: {error}")
    return error


def _plot_capture_source() -> str:
    """Source of the setup cell that installs plot capture in the kernel."""
    return (
        inspect.getsource(_install_plot_capture)
        + f"\n\n_install_plot_capture({PLOT_DATA_MIMETYPE!r})\ndel _install_plot_capture\n"
    )


def _install_plot_capture(mimetype: str) -> None:
    """Record the data of every matplotlib figure a cell displays.

    Runs inside the notebook kernel: its source is injected as a setup cell, so
    it must be self-contained. A display formatter records each figure as it is
    displayed (by ``plt.show()``, by the inline backend at the end of a cell, or
    by ``display``), and after each cell the recorded figures are emitted as
    one output of type ``mimetype``. Non-finite numbers are recorded as null
    (NaN) or the strings ``"inf"`` and ``"-inf"``.
    """
    import math
    import sys

    import numpy as np
    from IPython import get_ipython
    from IPython.core.formatters import BaseFormatter
    from IPython.display import display

    captured = []

    def to_json(value):
        if isinstance(value, list):
            return [to_json(item) for item in value]
        if isinstance(value, float):
            if math.isnan(value):
                return None
            if math.isinf(value):
                return "inf" if value > 0 else "-inf"
        return value

    def to_number(value):
        try:
            return to_json(float(value))
        except (TypeError, ValueError):
            return str(value)

    def to_list(values):
        try:
            array = np.ma.filled(np.ma.asarray(values, dtype=float), np.nan)
        except (TypeError, ValueError):
            return [str(item) for item in np.ravel(np.asarray(values, dtype=object))]
        return to_json(array.tolist())

    def axes_data(ax):
        from matplotlib.patches import Rectangle

        return {
            "title": ax.get_title(),
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
            "lines": [
                {
                    "label": str(line.get_label()),
                    "x": to_list(line.get_xdata()),
                    "y": to_list(line.get_ydata()),
                }
                for line in ax.get_lines()
            ],
            "patches": [
                {
                    "label": str(patch.get_label()),
                    "x": to_number(patch.get_x()),
                    "y": to_number(patch.get_y()),
                    "width": to_number(patch.get_width()),
                    "height": to_number(patch.get_height()),
                }
                for patch in ax.patches
                if isinstance(patch, Rectangle)
            ],
            "collections": [
                {
                    "label": str(collection.get_label()),
                    "offsets": to_list(collection.get_offsets()),
                    "segments": [to_list(segment) for segment in collection.get_segments()]
                    if hasattr(collection, "get_segments")
                    else [],
                }
                for collection in ax.collections
            ],
        }

    class PlotDataFormatter(BaseFormatter):
        def __call__(self, obj):
            figure_module = sys.modules.get("matplotlib.figure")
            if figure_module is not None and isinstance(obj, figure_module.Figure):
                try:
                    captured.append({"axes": [axes_data(ax) for ax in obj.axes]})
                except Exception as error:
                    captured.append({"error": f"{type(error).__name__}: {error}"})
            return None

    def reset(*args):
        captured.clear()

    def emit(*args):
        if captured:
            data = {"figures": list(captured)}
            captured.clear()
            display({mimetype: data}, raw=True)

    shell = get_ipython()
    shell.display_formatter.formatters[mimetype] = PlotDataFormatter()
    shell.events.register("pre_run_cell", reset)
    shell.events.register("post_run_cell", emit)


def compare_notebooks(
    reference_dir: Union[str, Path],
    candidate_dir: Union[str, Path],
    rtol: float = 0.0,
    atol: float = 0.0,
) -> ComparisonResult:
    """Compare the outputs of two directories of executed notebooks.

    Notebooks are paired by file name and code cells by position, numbered
    from 1 and ignoring markdown cells. For each code cell, its outputs are
    compared: stdout, ``text/plain``, ``text/html`` and error outputs must
    match exactly once memory addresses (``0x`` followed by six or more hex
    digits), which change on every run, are masked; :data:`PLOT_DATA_MIMETYPE`
    outputs are compared by their plotted data only, numbers within tolerance
    and the number of figures, axes and plotted elements exactly. Plot
    presentation (titles, axis labels, legend labels), stderr, execution
    counts, metadata, images and widget outputs are ignored.

    Reported as differences:

    - a notebook present in only one directory;
    - a notebook whose code cell count or sources differ (schema change);
    - a notebook that raised an error on only one side, or a different error
      (execution failed), naming the cell; later cells of that notebook are
      not compared, since the side that stopped has no outputs for them;
    - a cell whose text outputs changed (output changed);
    - a cell whose plotted data changed: values beyond tolerance, or figures,
      axes or plotted elements added or removed (plot data changed). There is
      one such difference per cell; its detail names each set of axes that
      moved by its title (the reference's, or the candidate's if the reference
      has none), or as ``figure <i>, axes <j>`` (numbered from 1) when it is
      untitled, with what changed in it. When the number of plot outputs,
      figures, or axes in a figure differs, the detail gives both counts and
      the plots at that level are not paired further, since pairing by
      position would attribute changes to the wrong plots. Figures are
      numbered across the cell, advancing by the larger side's count.

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
        ComparisonInputError: If either path is not a directory, either
            directory contains no notebooks, or a notebook cannot be read.
    """
    reference_dir, candidate_dir = Path(reference_dir), Path(candidate_dir)
    ref_notebooks = _find_notebooks(reference_dir, "reference")
    cand_notebooks = _find_notebooks(candidate_dir, "candidate")
    names = sorted(ref_notebooks.keys() | cand_notebooks.keys())
    result = ComparisonResult(
        title="Notebooks",
        reference=str(reference_dir),
        candidate=str(candidate_dir),
        n_compared=len(names),
    )
    for name in names:
        if name not in cand_notebooks:
            result.differences.append(
                Difference(name, DifferenceKind.ONLY_IN_REFERENCE, "notebook")
            )
        elif name not in ref_notebooks:
            result.differences.append(
                Difference(name, DifferenceKind.ONLY_IN_CANDIDATE, "notebook")
            )
        else:
            differences, notes = _compare_notebook(
                name,
                _read_notebook(ref_notebooks[name], "reference"),
                _read_notebook(cand_notebooks[name], "candidate"),
                rtol,
                atol,
            )
            result.differences.extend(differences)
            result.notes.extend(notes)
    return result


def _read_notebook(path: Path, side: str) -> Any:
    """Read a notebook, raising ComparisonInputError if it is not a valid notebook."""
    import nbformat

    try:
        return nbformat.read(path, as_version=4)
    except Exception as e:
        raise ComparisonInputError(
            f"Could not read {side} notebook {path}: {type(e).__name__}: {e}"
        ) from e


def _compare_notebook(
    name: str, reference: Any, candidate: Any, rtol: float, atol: float
) -> tuple[list[Difference], list[str]]:
    """Compare two executed notebooks code cell by code cell."""
    ref_cells = [cell for cell in reference.cells if cell.cell_type == "code"]
    cand_cells = [cell for cell in candidate.cells if cell.cell_type == "code"]
    ref_sources = [_text(cell.source) for cell in ref_cells]
    cand_sources = [_text(cell.source) for cell in cand_cells]
    if len(ref_sources) != len(cand_sources):
        detail = (
            f"{len(ref_sources)} code cells in reference, {len(cand_sources)} in candidate"
        )
        return [Difference(name, DifferenceKind.SCHEMA, detail)], []
    changed_sources = [
        str(n) for n, (r, c) in enumerate(zip(ref_sources, cand_sources), start=1) if r != c
    ]
    if changed_sources:
        detail = f"source changed in code cell(s) {', '.join(changed_sources)}"
        return [Difference(name, DifferenceKind.SCHEMA, detail)], []

    differences = []
    no_plot_data = {"reference": 0, "candidate": 0}
    for n, (ref_cell, cand_cell) in enumerate(zip(ref_cells, cand_cells), start=1):
        item = f"{name}, cell {n}"
        ref_text, ref_errors, ref_plots = _summarize_outputs(ref_cell)
        cand_text, cand_errors, cand_plots = _summarize_outputs(cand_cell)
        if ref_errors != cand_errors:
            detail = (
                f"reference: {'; '.join(ref_errors) or 'no error'}; "
                f"candidate: {'; '.join(cand_errors) or 'no error'}; later cells not compared"
            )
            differences.append(Difference(item, DifferenceKind.EXECUTION_FAILED, detail))
            break
        changed = [label for label in ref_text if ref_text[label] != cand_text[label]]
        if changed:
            differences.append(
                Difference(item, DifferenceKind.OUTPUT, f"{', '.join(changed)} changed")
            )
        if ref_plots and cand_plots:
            plot_difference = _compare_plot_data(item, ref_plots, cand_plots, rtol, atol)
            if plot_difference is not None:
                differences.append(plot_difference)
        elif ref_plots or cand_plots:
            no_plot_data["candidate" if ref_plots else "reference"] += 1

    notes = [
        f"{name}: plots not comparable in {count} cell(s) ({side} has no plot data)"
        for side, count in no_plot_data.items()
        if count
    ]
    return differences, notes


def _text(value: Union[str, list[str]]) -> str:
    return value if isinstance(value, str) else "".join(value)


def _scrub(text: str) -> str:
    """Mask memory addresses, which change from run to run."""
    return _MEMORY_ADDRESS.sub("0x...", text)


def _summarize_outputs(cell: Any) -> tuple[dict[str, Any], list[str], list[Any]]:
    """Extract the compared parts of a code cell's outputs.

    Returns:
        The cell's text outputs by kind, its errors, and its plot data.
    """
    stdout, errors, plots = [], [], []
    texts = {mimetype: [] for mimetype in _TEXT_MIMETYPES}
    for output in cell.get("outputs", []):
        output_type = output.get("output_type")
        if output_type == "stream":
            if output.get("name") == "stdout":
                stdout.append(_text(output.get("text", "")))
        elif output_type == "error":
            errors.append(f"{output.get('ename')}: {_scrub(output.get('evalue', ''))}")
        elif output_type in ("display_data", "execute_result"):
            data = output.get("data", {})
            if _WIDGET_MIMETYPE in data:
                continue
            if PLOT_DATA_MIMETYPE in data:
                plots.append(data[PLOT_DATA_MIMETYPE])
            for mimetype in _TEXT_MIMETYPES:
                if mimetype in data:
                    texts[mimetype].append(_scrub(_text(data[mimetype])))
    return {"stdout": _scrub("".join(stdout)), **texts}, errors, plots


def _compare_plot_data(
    item: str, reference: Any, candidate: Any, rtol: float, atol: float
) -> Optional[Difference]:
    """Compare a cell's plot data outputs, numbers within tolerance.

    Plot outputs, their figures and each figure's axes are paired by position
    only when their counts agree; a count that differs is reported with both
    counts, and nothing at that level or below it is paired or compared.
    Figures are numbered across the cell, advancing by the larger side's
    count. Each set of axes whose plotted data moved is described under its
    title (the reference's, or the candidate's if the reference has none), or
    as ``figure <i>, axes <j>`` when it is untitled.

    Returns:
        One difference for the cell, with the value differences of everything
        that moved merged, or None if nothing moved.
    """
    ref_numbers, cand_numbers, details = [], [], []

    def compare_part(label: str, ref_part: Any, cand_part: Any) -> None:
        part_ref, part_cand = [], []
        same = _pair_plot_values(ref_part, cand_part, part_ref, part_cand)
        ref_numbers.extend(part_ref)
        cand_numbers.extend(part_cand)
        part_diff = _plot_value_diff(part_ref, part_cand, rtol, atol)
        changes = []
        if part_diff is not None:
            changes.append(f"{part_diff.n_changed} plotted value(s) changed")
        if not same:
            changes.append("plot structure changed")
        if changes:
            details.append(f"{label}: {', '.join(changes)}")

    if len(reference) != len(candidate):
        # Pairing by position would attribute changes to the wrong plots.
        details.append(
            f"plot structure changed ({len(reference)} plot outputs in reference, "
            f"{len(candidate)} in candidate)"
        )
        reference, candidate = [], []
    multiple_outputs = len(reference) > 1 or len(candidate) > 1
    n_figures = 0
    for k, (ref_output, cand_output) in enumerate(zip(reference, candidate), start=1):
        ref_figures = _plot_children(ref_output, "figures")
        cand_figures = _plot_children(cand_output, "figures")
        figures_start = n_figures + 1
        n_figures += max(len(ref_figures or []), len(cand_figures or []))
        if ref_figures is None or cand_figures is None:
            compare_part(f"plot output {k}", ref_output, cand_output)
            continue
        compare_part(
            f"plot output {k}",
            _without(ref_output, "figures"),
            _without(cand_output, "figures"),
        )
        if len(ref_figures) != len(cand_figures):
            where = f" in plot output {k}" if multiple_outputs else ""
            details.append(
                f"plot structure changed ({len(ref_figures)} figures in reference, "
                f"{len(cand_figures)} in candidate{where})"
            )
            continue
        for i, (ref_figure, cand_figure) in enumerate(
            zip(ref_figures, cand_figures), start=figures_start
        ):
            ref_axes = _plot_children(ref_figure, "axes")
            cand_axes = _plot_children(cand_figure, "axes")
            if ref_axes is None or cand_axes is None:
                compare_part(f"figure {i}", ref_figure, cand_figure)
                continue
            compare_part(
                f"figure {i}", _without(ref_figure, "axes"), _without(cand_figure, "axes")
            )
            if len(ref_axes) != len(cand_axes):
                details.append(
                    f"plot structure changed ({len(ref_axes)} axes in reference, "
                    f"{len(cand_axes)} in candidate in figure {i})"
                )
                continue
            for j, (ref_ax, cand_ax) in enumerate(zip(ref_axes, cand_axes), start=1):
                title = next(
                    (
                        ax["title"]
                        for ax in (ref_ax, cand_ax)
                        if isinstance(ax, dict) and ax.get("title")
                    ),
                    None,
                )
                compare_part(
                    f"'{title}'" if title else f"figure {i}, axes {j}", ref_ax, cand_ax
                )

    value_diff = _plot_value_diff(ref_numbers, cand_numbers, rtol, atol)
    if not details and value_diff is None:
        return None
    return Difference(item, DifferenceKind.PLOT, "; ".join(details), value_diff)


def _plot_value_diff(
    ref_numbers: list[float], cand_numbers: list[float], rtol: float, atol: float
) -> Optional[ValueDiff]:
    """Compare paired plotted numbers within tolerance."""
    return compare_values(
        pd.Series(ref_numbers, dtype=float),
        pd.Series(cand_numbers, dtype=float),
        rtol=rtol,
        atol=atol,
    )


def _plot_children(node: Any, key: str) -> Optional[list[Any]]:
    """The list under ``key`` of a plot data object, or None if there is none."""
    if isinstance(node, dict) and isinstance(node.get(key), list):
        return node[key]
    return None


def _without(node: dict[str, Any], key: str) -> dict[str, Any]:
    return {k: v for k, v in node.items() if k != key}


def _pair_plot_values(
    reference: Any, candidate: Any, ref_numbers: list[float], cand_numbers: list[float]
) -> bool:
    """Collect paired numbers from two plot data trees.

    Presentation fields (:data:`_PLOT_PRESENTATION_KEYS`) are skipped
    entirely, so neither their values nor their presence is compared.

    Returns:
        Whether everything other than the numbers and the presentation
        fields matches exactly.
    """
    ref_number, cand_number = _plot_number(reference), _plot_number(candidate)
    if ref_number is not None and cand_number is not None:
        ref_numbers.append(ref_number)
        cand_numbers.append(cand_number)
        return True
    if isinstance(reference, dict) and isinstance(candidate, dict):
        ref_keys = reference.keys() - _PLOT_PRESENTATION_KEYS
        cand_keys = candidate.keys() - _PLOT_PRESENTATION_KEYS
        same = ref_keys == cand_keys
        for key in ref_keys & cand_keys:
            same &= _pair_plot_values(
                reference[key], candidate[key], ref_numbers, cand_numbers
            )
        return same
    if isinstance(reference, list) and isinstance(candidate, list):
        same = len(reference) == len(candidate)
        for ref_item, cand_item in zip(reference, candidate):
            same &= _pair_plot_values(ref_item, cand_item, ref_numbers, cand_numbers)
        return same
    return reference == candidate


def _plot_number(value: Any) -> Optional[float]:
    """The number a plot data leaf stands for, or None if it is not a number."""
    if value is None:
        return math.nan
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if value in ("inf", "-inf"):
        return float(value)
    return None
