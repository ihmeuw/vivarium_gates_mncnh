"""Tests for notebook execution and comparison in vivarium_gates_mncnh.tools.compare.notebooks.

Comparison tests build executed notebooks with nbformat; only the execution tests
need a Jupyter kernel.
"""

import copy

import nbformat
import pytest

from vivarium_gates_mncnh.tools.compare import (
    PLOT_DATA_MIMETYPE,
    ComparisonInputError,
    DifferenceKind,
    compare_notebooks,
    discover_notebook_paths,
    execute_notebooks,
)

from .compare_helpers import (
    PLOT_CELL,
    PLOTTING_NOTEBOOK_SOURCE,
    PRINT_CELL,
    TABLE_CELL,
    code_cell,
    code_cells,
    error_output,
    execute_result,
    image_output,
    plot_data,
    plot_data_outputs,
    plot_output,
    read_notebook,
    require_kernel,
    standard_cells,
    stderr,
    stdout,
    stdout_text,
    write_notebook,
    write_source_notebook,
)

NOTEBOOK = "analysis.ipynb"


def _write_pair(tmp_path, reference_cells, candidate_cells, name=NOTEBOOK, **kwargs):
    """Write one notebook into each of reference/ and candidate/, plus an identical
    second notebook so a comparison spans more than one notebook."""
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    write_notebook(reference_dir, name, reference_cells, kwargs.get("reference_metadata"))
    write_notebook(candidate_dir, name, candidate_cells, kwargs.get("candidate_metadata"))
    for directory in (reference_dir, candidate_dir):
        write_notebook(directory, "other.ipynb", standard_cells())
    return reference_dir, candidate_dir


def _assert_names_notebook_and_cell(difference, name=NOTEBOOK):
    assert name.removesuffix(".ipynb") in difference.item
    assert "cell" in difference.item.lower()


def _plot_cell_with(payload):
    cells = standard_cells(with_plot_data=False)
    cells[PLOT_CELL].outputs.append(plot_output(payload))
    return cells


#########################
# Notebook discovery    #
#########################


def test_discover_skips_checkpoints_and_merge_files(tmp_path):
    """discover_notebook_paths returns top-level notebooks sorted, skipping
    checkpoints and nbdime _BACKUP_/_BASE_/_LOCAL_/_REMOTE_ files."""
    for name in [
        "b_second.ipynb",
        "a_first.ipynb",
        "analysis_BACKUP_1234.ipynb",
        "analysis_BASE_1234.ipynb",
        "analysis_LOCAL_1234.ipynb",
        "analysis_REMOTE_1234.ipynb",
    ]:
        write_notebook(tmp_path, name, standard_cells())
    write_notebook(tmp_path / ".ipynb_checkpoints", "a_first-checkpoint.ipynb", [])
    write_notebook(tmp_path / "nested", "c_nested.ipynb", [])
    (tmp_path / "notes.txt").write_text("not a notebook\n")

    found = discover_notebook_paths(tmp_path)

    assert found == [tmp_path / "a_first.ipynb", tmp_path / "b_second.ipynb"]


#########################
# Comparison            #
#########################


def test_identical_notebooks_compare_clean(tmp_path):
    """Two directories of identical executed notebooks compare clean."""
    reference_dir, candidate_dir = _write_pair(tmp_path, standard_cells(), standard_cells())

    result = compare_notebooks(reference_dir, candidate_dir)

    assert result.differences == []
    assert not result.moved
    assert result.n_compared == 2


def _add_stderr(reference, candidate):
    candidate[PRINT_CELL].outputs.append(stderr("UserWarning: something noisy\n"))


def _change_stderr(reference, candidate):
    reference[PRINT_CELL].outputs.append(stderr("took 1.2s\n"))
    candidate[PRINT_CELL].outputs.append(stderr("took 3.4s\n"))


def _change_execution_counts(reference, candidate):
    for cell in candidate:
        if cell.cell_type == "code":
            cell.execution_count = (cell.execution_count or 0) + 10
            for output in cell.outputs:
                if output.output_type == "execute_result":
                    output.execution_count = cell.execution_count


def _change_cell_metadata(reference, candidate):
    for cell in candidate:
        cell.metadata["papermill"] = {"duration": 1.5, "status": "completed"}
        cell.metadata["tags"] = ["changed"]
    candidate[PLOT_CELL].outputs[0].metadata["needs_background"] = "light"


def _change_image(reference, candidate):
    reference[PLOT_CELL].outputs[0] = image_output("AAAAAAAA")
    candidate[PLOT_CELL].outputs[0] = image_output("BBBBBBBB")


def _change_widget(reference, candidate):
    for cells, model_id in [(reference, "aaaa1111"), (candidate, "bbbb2222")]:
        cells[TABLE_CELL].outputs.append(
            nbformat.v4.new_output(
                "display_data",
                data={
                    "application/vnd.jupyter.widget-view+json": {
                        "model_id": model_id,
                        "version_major": 2,
                        "version_minor": 0,
                    },
                    "text/plain": "IntProgress(value=10, max=10)",
                },
            )
        )


IGNORED_CHANGES = {
    "stderr_added": _add_stderr,
    "stderr_changed": _change_stderr,
    "execution_counts": _change_execution_counts,
    "cell_metadata": _change_cell_metadata,
    "image": _change_image,
    "widget": _change_widget,
}


@pytest.mark.parametrize("change", list(IGNORED_CHANGES) + ["notebook_metadata"])
def test_ignored_outputs_do_not_register(tmp_path, change):
    """Differences only in stderr, execution counts, metadata, images or widget
    outputs compare clean."""
    reference_cells, candidate_cells = standard_cells(), standard_cells()
    metadata = {}
    if change == "notebook_metadata":
        metadata["candidate_metadata"] = {
            "kernelspec": {"name": "other", "display_name": "Other", "language": "python"},
            "papermill": {"duration": 12.5, "parameters": {}},
        }
    else:
        IGNORED_CHANGES[change](reference_cells, candidate_cells)
    reference_dir, candidate_dir = _write_pair(
        tmp_path, reference_cells, candidate_cells, **metadata
    )

    result = compare_notebooks(reference_dir, candidate_dir)

    assert result.differences == []


def test_changed_stdout_reported(tmp_path):
    """A changed stdout stream is reported as output changed, naming notebook and
    cell."""
    candidate_cells = standard_cells()
    candidate_cells[PRINT_CELL].outputs = [stdout("43\n")]
    reference_dir, candidate_dir = _write_pair(tmp_path, standard_cells(), candidate_cells)

    result = compare_notebooks(reference_dir, candidate_dir)

    assert len(result.differences) == 1
    difference = result.differences[0]
    assert difference.kind == DifferenceKind.OUTPUT
    _assert_names_notebook_and_cell(difference)


@pytest.mark.parametrize("mimetype", ["text/plain", "text/html"])
def test_changed_text_plain_and_html_reported(tmp_path, mimetype):
    """Changed text/plain and text/html outputs are reported as output changed."""
    candidate_cells = standard_cells()
    if mimetype == "text/plain":
        changed = execute_result("   a\n0  2", "<table><tr><td>1</td></tr></table>", count=2)
    else:
        changed = execute_result("   a\n0  1", "<table><tr><td>2</td></tr></table>", count=2)
    candidate_cells[TABLE_CELL].outputs = [changed]
    reference_dir, candidate_dir = _write_pair(tmp_path, standard_cells(), candidate_cells)

    result = compare_notebooks(reference_dir, candidate_dir)

    assert len(result.differences) == 1
    difference = result.differences[0]
    assert difference.kind == DifferenceKind.OUTPUT
    _assert_names_notebook_and_cell(difference)


@pytest.mark.parametrize("failing_side", ["reference", "candidate"])
def test_error_on_one_side_reported(tmp_path, failing_side):
    """A cell that errors on only one side is reported as execution failed."""
    failing = standard_cells()
    failing[PLOT_CELL].outputs = [error_output("ValueError", "boom")]
    if failing_side == "reference":
        reference_dir, candidate_dir = _write_pair(tmp_path, failing, standard_cells())
    else:
        reference_dir, candidate_dir = _write_pair(tmp_path, standard_cells(), failing)

    result = compare_notebooks(reference_dir, candidate_dir)

    assert result.moved
    failures = [d for d in result.differences if d.kind == DifferenceKind.EXECUTION_FAILED]
    assert len(failures) == 1
    _assert_names_notebook_and_cell(failures[0])
    assert all("other" not in d.item for d in result.differences)


@pytest.mark.parametrize(
    "changed_payload",
    [
        plot_data(line_y=(1.0, 2.0, 3.5)),
        plot_data(bar_height=2.75),
        plot_data(point=(1.0, 4.5)),
    ],
    ids=["line", "bar", "scatter"],
)
def test_changed_plot_data_reported(tmp_path, changed_payload):
    """A plot-data output with changed values is reported as plot data changed."""
    reference_dir, candidate_dir = _write_pair(
        tmp_path, _plot_cell_with(plot_data()), _plot_cell_with(changed_payload)
    )

    result = compare_notebooks(reference_dir, candidate_dir)

    assert len(result.differences) == 1
    difference = result.differences[0]
    assert difference.kind == DifferenceKind.PLOT
    _assert_names_notebook_and_cell(difference)


@pytest.mark.parametrize("tolerance", [{"rtol": 1e-6}, {"atol": 1e-6}], ids=["rtol", "atol"])
def test_plot_data_tolerance_is_applied(tmp_path, tolerance):
    """A small change in plotted values is reported by default and not within
    rtol/atol."""
    nudged = plot_data(line_y=(1.0, 2.0 * (1 + 1e-9), 3.0))
    reference_dir, candidate_dir = _write_pair(
        tmp_path, _plot_cell_with(plot_data()), _plot_cell_with(nudged)
    )

    by_default = compare_notebooks(reference_dir, candidate_dir)
    assert [d.kind for d in by_default.differences] == [DifferenceKind.PLOT]

    within_tolerance = compare_notebooks(reference_dir, candidate_dir, **tolerance)
    assert not within_tolerance.moved


@pytest.mark.parametrize("side_with_plot_data", ["reference", "candidate"])
def test_plot_data_missing_on_one_side_is_a_note(tmp_path, side_with_plot_data):
    """When only one side has plot data for a cell, nothing moves and a note says
    plots were not comparable."""
    with_plot = standard_cells(with_plot_data=True)
    without_plot = standard_cells(with_plot_data=False)
    if side_with_plot_data == "reference":
        reference_dir, candidate_dir = _write_pair(tmp_path, with_plot, without_plot)
    else:
        reference_dir, candidate_dir = _write_pair(tmp_path, without_plot, with_plot)

    result = compare_notebooks(reference_dir, candidate_dir)

    assert not result.moved
    assert result.notes
    assert any(NOTEBOOK.removesuffix(".ipynb") in note for note in result.notes)


def test_notebook_only_on_one_side(tmp_path):
    """A notebook present in only one directory is reported as only in reference or
    only in candidate."""
    reference_dir, candidate_dir = _write_pair(tmp_path, standard_cells(), standard_cells())
    write_notebook(reference_dir, "reference_only.ipynb", standard_cells())
    write_notebook(candidate_dir, "candidate_only.ipynb", standard_cells())

    result = compare_notebooks(reference_dir, candidate_dir)

    assert result.moved
    kinds = {
        name: [d.kind for d in result.differences if name in d.item]
        for name in ["reference_only", "candidate_only", NOTEBOOK.removesuffix(".ipynb")]
    }
    assert kinds["reference_only"] == [DifferenceKind.ONLY_IN_REFERENCE]
    assert kinds["candidate_only"] == [DifferenceKind.ONLY_IN_CANDIDATE]
    assert kinds["analysis"] == []
    assert result.n_compared == 4


@pytest.mark.parametrize("change", ["edited_source", "added_cell", "removed_cell"])
def test_changed_cell_sources_reported(tmp_path, change):
    """Notebooks whose cell sources differ are reported as a schema change."""
    candidate_cells = standard_cells()
    if change == "edited_source":
        candidate_cells[PRINT_CELL].source = "print(40 + 2)"
    elif change == "added_cell":
        candidate_cells.append(code_cell("x = 1", [], execution_count=4))
    else:
        del candidate_cells[TABLE_CELL]
    reference_dir, candidate_dir = _write_pair(tmp_path, standard_cells(), candidate_cells)

    result = compare_notebooks(reference_dir, candidate_dir)

    assert result.moved
    schema = [d for d in result.differences if d.kind == DifferenceKind.SCHEMA]
    assert schema
    assert all(NOTEBOOK.removesuffix(".ipynb") in d.item for d in schema)
    assert all("other" not in d.item for d in result.differences)


@pytest.mark.parametrize("bad_side", ["reference", "candidate"])
@pytest.mark.parametrize("problem", ["missing", "file", "empty", "only_checkpoints"])
def test_missing_directory_raises(tmp_path, bad_side, problem):
    """A path that is not a directory, or a directory with no notebooks, raises
    ComparisonInputError."""
    good = tmp_path / "good"
    write_notebook(good, NOTEBOOK, standard_cells())
    bad = tmp_path / "bad"
    if problem == "file":
        bad = good / NOTEBOOK
    elif problem == "empty":
        bad.mkdir()
    elif problem == "only_checkpoints":
        write_notebook(bad / ".ipynb_checkpoints", "analysis-checkpoint.ipynb", [])
        (bad / "notes.txt").write_text("not a notebook\n")
    paths = (bad, good) if bad_side == "reference" else (good, bad)

    with pytest.raises(ComparisonInputError):
        compare_notebooks(*paths)


#########################
# Execution             #
#########################


def _all(payloads, collection):
    return [
        item
        for payload in payloads
        for figure in payload["figures"]
        for axes in figure["axes"]
        for item in axes[collection]
    ]


def test_execute_writes_outputs_and_plot_data(tmp_path):
    """Executing a small notebook that plots a line, bars and a scatter writes it to
    the output directory with a plot-data output holding the plotted values."""
    require_kernel()
    source_dir = tmp_path / "source"
    source = write_source_notebook(source_dir, "plots.ipynb", PLOTTING_NOTEBOOK_SOURCE)
    output_dir = tmp_path / "out" / "executed"  # does not exist yet

    statuses = execute_notebooks(source_dir, output_dir)

    assert statuses == {"plots.ipynb": None}
    executed_path = output_dir / "plots.ipynb"
    assert executed_path.is_file()
    executed = read_notebook(executed_path)
    assert [c.source for c in code_cells(executed)] == [
        c.source for c in code_cells(read_notebook(source))
    ]
    assert "ready" in stdout_text(executed)

    # Only the plotting cell displays a figure, so only it gets plot data.
    cells_with_plot_data = [
        i
        for i, cell in enumerate(code_cells(executed))
        if any(PLOT_DATA_MIMETYPE in o.get("data", {}) for o in cell.outputs)
    ]
    assert cells_with_plot_data == [2]
    plot_outputs = [
        o for o in code_cells(executed)[2].outputs if PLOT_DATA_MIMETYPE in o.get("data", {})
    ]
    assert len(plot_outputs) == 1
    assert plot_outputs[0]["output_type"] == "display_data"

    payloads = plot_data_outputs(executed)
    lines = _all(payloads, "lines")
    assert any(
        list(line["x"]) == pytest.approx([0, 1, 2])
        and list(line["y"]) == pytest.approx([1.5, 2.5, 4.5])
        for line in lines
        if len(line["y"]) == 3
    )
    heights = sorted(patch["height"] for patch in _all(payloads, "patches"))
    assert 3.25 in heights and 7.75 in heights
    offsets = [list(point) for c in _all(payloads, "collections") for point in c["offsets"]]
    assert [20.5, 8.125] in offsets
    assert [21.5, 9.375] in offsets


def test_execute_does_not_modify_sources(tmp_path):
    """Executing leaves the source notebooks unchanged."""
    require_kernel()
    source_dir = tmp_path / "source"
    write_source_notebook(source_dir, "plots.ipynb", PLOTTING_NOTEBOOK_SOURCE)
    write_source_notebook(source_dir, "prints.ipynb", ["print('hello')"])
    before = {path.name: path.read_bytes() for path in source_dir.iterdir()}

    execute_notebooks(source_dir, tmp_path / "executed")

    after = {path.name: path.read_bytes() for path in source_dir.iterdir()}
    assert after == before


def test_execute_continues_after_failure(tmp_path):
    """A failing notebook is reported with its error and still written; the other
    notebooks still run and map to None."""
    require_kernel()
    source_dir = tmp_path / "source"
    write_source_notebook(source_dir, "a_ok.ipynb", ["print('a ran')"])
    write_source_notebook(
        source_dir,
        "b_fails.ipynb",
        ["print('before')", "raise ValueError('boom')", "print('after')"],
    )
    write_source_notebook(source_dir, "c_ok.ipynb", ["print('c ran')"])
    output_dir = tmp_path / "executed"

    statuses = execute_notebooks(source_dir, output_dir)

    assert set(statuses) == {"a_ok.ipynb", "b_fails.ipynb", "c_ok.ipynb"}
    assert statuses["a_ok.ipynb"] is None
    assert statuses["c_ok.ipynb"] is None
    error = statuses["b_fails.ipynb"]
    assert isinstance(error, str) and error
    assert "ValueError" in error or "boom" in error

    for name in statuses:
        assert (output_dir / name).is_file()
    assert "a ran" in stdout_text(read_notebook(output_dir / "a_ok.ipynb"))
    assert "c ran" in stdout_text(read_notebook(output_dir / "c_ok.ipynb"))

    failed = read_notebook(output_dir / "b_fails.ipynb")
    assert "before" in stdout_text(failed)
    assert "after" not in stdout_text(failed)
    errors = [
        output
        for cell in code_cells(failed)
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert errors
    assert errors[0]["ename"] == "ValueError"


def test_executed_runs_compare_clean(tmp_path):
    """Two executions of the same notebooks into separate directories compare clean,
    plot data included."""
    require_kernel()
    source_dir = tmp_path / "source"
    write_source_notebook(source_dir, "plots.ipynb", PLOTTING_NOTEBOOK_SOURCE)
    write_source_notebook(source_dir, "prints.ipynb", ["print(sum(range(10)))"])
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    assert set(execute_notebooks(source_dir, first_dir).values()) == {None}
    assert set(execute_notebooks(source_dir, second_dir).values()) == {None}

    first_plots = plot_data_outputs(read_notebook(first_dir / "plots.ipynb"))
    second_plots = plot_data_outputs(read_notebook(second_dir / "plots.ipynb"))
    assert first_plots and second_plots

    result = compare_notebooks(first_dir, second_dir)

    assert result.differences == []
    assert result.n_compared == 2

    # The plot data took part in the comparison: nudging one plotted value in a copy
    # of the second run makes the plot move.
    nudged_dir = tmp_path / "nudged"
    nudged_dir.mkdir()
    for path in second_dir.glob("*.ipynb"):
        notebook = read_notebook(path)
        for cell in code_cells(notebook):
            for output in cell.outputs:
                payload = output.get("data", {}).get(PLOT_DATA_MIMETYPE)
                if payload is not None:
                    payload = copy.deepcopy(payload)
                    line = next(
                        line
                        for figure in payload["figures"]
                        for axes in figure["axes"]
                        for line in axes["lines"]
                        if len(line["y"]) == 3
                    )
                    line["y"][1] += 1.0
                    output["data"][PLOT_DATA_MIMETYPE] = payload
        nbformat.write(notebook, nudged_dir / path.name)

    nudged = compare_notebooks(first_dir, nudged_dir)
    assert [d.kind for d in nudged.differences] == [DifferenceKind.PLOT]
    assert "plots" in nudged.differences[0].item
