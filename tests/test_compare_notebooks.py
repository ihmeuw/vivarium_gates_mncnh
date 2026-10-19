"""Tests for notebook execution and comparison in vivarium_gates_mncnh.tools.compare.notebooks.

Comparison tests build executed notebooks with nbformat; only the execution tests
need a Jupyter kernel.
"""

import copy
import json

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
    # The text/plain fallback differs too, so a comparer that read widget outputs'
    # text would catch it.
    for cells, model_id, text in [
        (reference, "aaaa1111", "IntProgress(value=3, max=10)"),
        (candidate, "bbbb2222", "IntProgress(value=5, max=10)"),
    ]:
        cells[TABLE_CELL].outputs.append(
            nbformat.v4.new_output(
                "display_data",
                data={
                    "application/vnd.jupyter.widget-view+json": {
                        "model_id": model_id,
                        "version_major": 2,
                        "version_minor": 0,
                    },
                    "text/plain": text,
                },
            )
        )


REFERENCE_ADDRESS, CANDIDATE_ADDRESS = "0x7f12ab34cd56", "0x7f99aa00bb11"


def _change_stdout_address(reference, candidate):
    for cells, address in [(reference, REFERENCE_ADDRESS), (candidate, CANDIDATE_ADDRESS)]:
        cells[PRINT_CELL].outputs = [stdout(f"<object at {address}>\n42\n")]


def _change_text_address(reference, candidate):
    for cells, address in [(reference, REFERENCE_ADDRESS), (candidate, CANDIDATE_ADDRESS)]:
        cells[TABLE_CELL].outputs = [
            execute_result(
                f"<Thing at {address}>",
                f"<div>&lt;Thing at {address}&gt;</div>",
                count=2,
            )
        ]


def _change_error_address(reference, candidate):
    # The same error on both sides, in the last code cell, differing only by address.
    for cells, address in [(reference, REFERENCE_ADDRESS), (candidate, CANDIDATE_ADDRESS)]:
        cells[PLOT_CELL].outputs = [
            error_output("TypeError", f"cannot use <Thing object at {address}>")
        ]


IGNORED_CHANGES = {
    "stderr_added": _add_stderr,
    "stderr_changed": _change_stderr,
    "execution_counts": _change_execution_counts,
    "cell_metadata": _change_cell_metadata,
    "image": _change_image,
    "widget": _change_widget,
}

MEMORY_ADDRESS_CHANGES = {
    "stdout": _change_stdout_address,
    "text": _change_text_address,
    "error": _change_error_address,
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


@pytest.mark.parametrize("change", list(MEMORY_ADDRESS_CHANGES))
def test_memory_addresses_are_masked(tmp_path, change):
    """stdout, text/plain, text/html and error values that differ only by a memory
    address compare clean."""
    reference_cells, candidate_cells = standard_cells(), standard_cells()
    MEMORY_ADDRESS_CHANGES[change](reference_cells, candidate_cells)
    reference_dir, candidate_dir = _write_pair(tmp_path, reference_cells, candidate_cells)

    result = compare_notebooks(reference_dir, candidate_dir)

    assert result.differences == []


@pytest.mark.parametrize(
    "reference_text, candidate_text",
    [
        ("value 0x1f\n", "value 0x2f\n"),
        ("total apples\n", "total pears\n"),
        (
            f"<object at {REFERENCE_ADDRESS}> apples\n",
            f"<object at {REFERENCE_ADDRESS}> pears\n",
        ),
    ],
    ids=["short_hex", "word", "word_beside_address"],
)
def test_non_address_changes_are_not_masked(tmp_path, reference_text, candidate_text):
    """Text that differs by something other than a memory address (a short hex value,
    a word) is still reported as output changed."""
    reference_cells, candidate_cells = standard_cells(), standard_cells()
    reference_cells[PRINT_CELL].outputs = [stdout(reference_text)]
    candidate_cells[PRINT_CELL].outputs = [stdout(candidate_text)]
    reference_dir, candidate_dir = _write_pair(tmp_path, reference_cells, candidate_cells)

    result = compare_notebooks(reference_dir, candidate_dir)

    assert len(result.differences) == 1
    difference = result.differences[0]
    assert difference.kind == DifferenceKind.OUTPUT
    _assert_names_notebook_and_cell(difference)


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


def test_failed_execution_does_not_leave_stale_output(tmp_path, monkeypatch):
    """When a notebook cannot run at all (it fails before anything is written) and
    the output directory holds that notebook from an earlier run, the earlier copy is
    not left behind looking like a fresh result."""
    papermill = pytest.importorskip("papermill")
    source_dir = tmp_path / "source"
    write_source_notebook(source_dir, NOTEBOOK, ["print('fresh run')"])
    output_dir = tmp_path / "executed"
    # An earlier run's executed copy of the same notebook.
    stale_cell = code_cell(
        "print('fresh run')", [stdout("STALE-RUN-OUTPUT\n")], execution_count=1
    )
    write_notebook(output_dir, NOTEBOOK, [stale_cell])
    assert "STALE-RUN-OUTPUT" in (output_dir / NOTEBOOK).read_text()

    # Fail before papermill writes anything, so only the up-front removal of the
    # earlier copy can keep it from being left behind.
    def fail_immediately(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(papermill, "execute_notebook", fail_immediately)

    statuses = execute_notebooks(source_dir, output_dir)

    assert set(statuses) == {NOTEBOOK}
    error = statuses[NOTEBOOK]
    assert isinstance(error, str) and error
    assert "boom" in error or "RuntimeError" in error
    assert not (output_dir / NOTEBOOK).exists()


def test_plot_capture_source_runs_without_a_kernel():
    """The plot capture setup cell's source executes in a bare namespace against a
    stand-in IPython shell and records a figure's plotted values, with no Jupyter
    kernel."""
    pytest.importorskip("IPython")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from IPython.core.interactiveshell import InteractiveShell

    from vivarium_gates_mncnh.tools.compare import notebooks as notebooks_module

    source = notebooks_module._plot_capture_source()
    assert isinstance(source, str) and source.strip()

    published = []

    def record(*args, **kwargs):
        data = kwargs.get("data", args[0] if args else None)
        published.append(data)

    created_shell = not InteractiveShell.initialized()
    shell = InteractiveShell.instance()
    original_publish = shell.display_pub.publish
    try:
        # Record before the setup runs, so anything it wraps still reaches the recorder.
        shell.display_pub.publish = record

        namespace = {"__name__": "__main__", "get_ipython": lambda: shell}
        exec(compile(source, "<plot-capture-setup>", "exec"), namespace)

        run = shell.run_cell(
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import display\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot([1, 2, 3], [4.5, 5.5, 6.5])\n"
            "display(fig)\n"
        )
        run.raise_error()
    finally:
        plt.close("all")
        shell.display_pub.publish = original_publish
        if created_shell:
            InteractiveShell.clear_instance()

    payloads = []
    for data in published:
        if isinstance(data, dict) and PLOT_DATA_MIMETYPE in data:
            payload = data[PLOT_DATA_MIMETYPE]
            payloads.append(json.loads(payload) if isinstance(payload, str) else payload)
    assert payloads, f"no {PLOT_DATA_MIMETYPE} output was published: {published!r}"
    lines = _all(payloads, "lines")
    assert any(list(line["y"]) == pytest.approx([4.5, 5.5, 6.5]) for line in lines), lines


def _first_axes(payload):
    return payload["figures"][0]["axes"][0]


def _set_axes_field(field, value):
    def edit(payload):
        _first_axes(payload)[field] = value

    return edit


def _set_element_label(collection, value):
    def edit(payload):
        _first_axes(payload)[collection][0]["label"] = value

    return edit


PRESENTATION_CHANGES = {
    "title": _set_axes_field("title", "A renamed plot"),
    "xlabel": _set_axes_field("xlabel", "age (days)"),
    "ylabel": _set_axes_field("ylabel", "deaths per 1,000"),
    "line_label": _set_element_label("lines", "renamed line"),
    "patch_label": _set_element_label("patches", "renamed bars"),
    "collection_label": _set_element_label("collections", "renamed points"),
}


def _edited_plot_data(edit, **plot_data_kwargs):
    payload = copy.deepcopy(plot_data(**plot_data_kwargs))
    edit(payload)
    return payload


@pytest.mark.parametrize("change", list(PRESENTATION_CHANGES))
def test_plot_presentation_changes_are_ignored(tmp_path, change):
    """Plot data differing only in title, axis labels or legend labels compares clean,
    while the same presentation change alongside a changed value is still reported as
    plot data changed."""
    edit = PRESENTATION_CHANGES[change]

    # Presentation alone: clean.
    clean_dir = tmp_path / "presentation_only"
    reference_dir, candidate_dir = _write_pair(
        clean_dir, _plot_cell_with(plot_data()), _plot_cell_with(_edited_plot_data(edit))
    )

    result = compare_notebooks(reference_dir, candidate_dir)

    assert result.differences == []
    assert not result.moved

    # Presentation plus one changed y value: exactly one PLOT difference counting
    # only the moved value.
    moved_dir = tmp_path / "presentation_and_value"
    reference_dir, candidate_dir = _write_pair(
        moved_dir,
        _plot_cell_with(plot_data()),
        _plot_cell_with(_edited_plot_data(edit, line_y=(1.0, 2.0, 3.5))),
    )

    result = compare_notebooks(reference_dir, candidate_dir)

    assert len(result.differences) == 1
    difference = result.differences[0]
    assert difference.kind == DifferenceKind.PLOT
    _assert_names_notebook_and_cell(difference)
    assert difference.value_diff is not None
    assert difference.value_diff.n_changed == 1


def _add_line(payload):
    _first_axes(payload)["lines"].append(
        {"label": "second line", "x": [0.0, 1.0, 2.0], "y": [5.0, 6.0, 7.0]}
    )


def _remove_line(payload):
    _first_axes(payload)["lines"].clear()


def _add_patch(payload):
    _first_axes(payload)["patches"].append(
        {"label": "bars", "x": 0.6, "y": 0.0, "width": 0.8, "height": 1.5}
    )


def _remove_patch(payload):
    _first_axes(payload)["patches"].clear()


def _add_collection(payload):
    _first_axes(payload)["collections"].append(
        {"label": "more points", "offsets": [[2.0, 5.0]], "segments": []}
    )


def _remove_collection(payload):
    _first_axes(payload)["collections"].clear()


STRUCTURE_CHANGES = {
    "line_added": _add_line,
    "line_removed": _remove_line,
    "bar_added": _add_patch,
    "bar_removed": _remove_patch,
    "scatter_added": _add_collection,
    "scatter_removed": _remove_collection,
}


@pytest.mark.parametrize("change", list(STRUCTURE_CHANGES))
def test_plot_structure_change_reported(tmp_path, change):
    """Plot data with a line, bar or scatter series added or removed is reported as
    plot data changed."""
    candidate_payload = _edited_plot_data(STRUCTURE_CHANGES[change])
    reference_dir, candidate_dir = _write_pair(
        tmp_path, _plot_cell_with(plot_data()), _plot_cell_with(candidate_payload)
    )

    result = compare_notebooks(reference_dir, candidate_dir)

    assert result.moved
    assert len(result.differences) == 1
    difference = result.differences[0]
    assert difference.kind == DifferenceKind.PLOT
    _assert_names_notebook_and_cell(difference)


def _two_axes_plot_data(titles, second_line_y=(1.0, 2.0, 3.0)):
    """Plot data with one figure holding two copies of the standard axes, titled
    ``titles``; the second axes' line plots ``second_line_y``."""
    payload = copy.deepcopy(plot_data())
    first = _first_axes(payload)
    second = copy.deepcopy(first)
    first["title"], second["title"] = titles
    second["lines"][0]["y"] = list(second_line_y)
    payload["figures"][0]["axes"].append(second)
    return payload


def test_plot_difference_names_moved_axes_by_title(tmp_path):
    """In a cell with two titled subplots where only one moved, the plot data changed
    detail names that subplot's title and not the other's; an untitled subplot that
    moved is named by figure and axes number."""
    # Titled: only "Hemoglobin" moves.
    titles = ("Anemia", "Hemoglobin")
    reference_dir, candidate_dir = _write_pair(
        tmp_path / "titled",
        _plot_cell_with(_two_axes_plot_data(titles)),
        _plot_cell_with(_two_axes_plot_data(titles, second_line_y=(1.0, 2.0, 3.5))),
    )

    result = compare_notebooks(reference_dir, candidate_dir)

    assert len(result.differences) == 1
    difference = result.differences[0]
    assert difference.kind == DifferenceKind.PLOT
    _assert_names_notebook_and_cell(difference)
    assert "Hemoglobin" in difference.detail
    assert "Anemia" not in difference.detail
    assert difference.value_diff is not None
    assert difference.value_diff.n_changed == 1

    # Untitled: the moved second axes is named by figure and axes number.
    untitled = ("", "")
    reference_dir, candidate_dir = _write_pair(
        tmp_path / "untitled",
        _plot_cell_with(_two_axes_plot_data(untitled)),
        _plot_cell_with(_two_axes_plot_data(untitled, second_line_y=(1.0, 2.0, 3.5))),
    )

    result = compare_notebooks(reference_dir, candidate_dir)

    assert len(result.differences) == 1
    difference = result.differences[0]
    assert difference.kind == DifferenceKind.PLOT
    _assert_names_notebook_and_cell(difference)
    detail = difference.detail.lower()
    assert "figure 1" in detail
    assert "axes 2" in detail
    assert "axes 1" not in detail
    assert difference.value_diff is not None
    assert difference.value_diff.n_changed == 1


def test_axes_count_change_is_not_paired_by_position(tmp_path):
    """When a subplot is inserted before others in a figure, the plot data changed
    detail gives both axes counts and does not name the shifted subplots as moved."""
    titles = ("Anemia", "Hemoglobin")
    reference_payload = _two_axes_plot_data(titles)
    candidate_payload = _two_axes_plot_data(titles)
    inserted = copy.deepcopy(_first_axes(candidate_payload))
    inserted["title"] = "New"
    candidate_payload["figures"][0]["axes"].insert(0, inserted)
    reference_dir, candidate_dir = _write_pair(
        tmp_path, _plot_cell_with(reference_payload), _plot_cell_with(candidate_payload)
    )

    result = compare_notebooks(reference_dir, candidate_dir)

    assert len(result.differences) == 1
    difference = result.differences[0]
    assert difference.kind == DifferenceKind.PLOT
    _assert_names_notebook_and_cell(difference)
    assert "2" in difference.detail
    assert "3" in difference.detail
    assert "axes" in difference.detail.lower()
    assert "Anemia" not in difference.detail
    assert "Hemoglobin" not in difference.detail
