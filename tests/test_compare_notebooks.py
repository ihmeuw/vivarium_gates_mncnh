"""Tests for notebook execution and comparison in vivarium_gates_mncnh.tools.compare.notebooks.

Comparison tests build executed notebooks with nbformat; only the execution tests
need a Jupyter kernel.
"""

import copy
import html
import json
import math
import re
from pathlib import Path
from typing import NamedTuple, Optional

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


def test_output_change_names_code_cell_and_first_changed_line(tmp_path):
    """An output change names the code cell with the In [n] Jupyter shows, and shows
    the first changed line on each side and how many lines changed."""
    reference_cells, candidate_cells = standard_cells(), standard_cells()
    reference_cells[PRINT_CELL].outputs = [stdout("header\n42\n7\n")]
    candidate_cells[PRINT_CELL].outputs = [stdout("header\n43\n8\n")]
    reference_dir, candidate_dir = _write_pair(tmp_path, reference_cells, candidate_cells)

    result = compare_notebooks(reference_dir, candidate_dir)

    (difference,) = result.differences
    assert difference.kind == DifferenceKind.OUTPUT
    assert difference.item == f"{NOTEBOOK}, code cell 1 (In [1])"
    assert difference.detail == "stdout: '42' -> '43' (2 line(s))"


def test_output_change_shows_where_a_long_line_differs(tmp_path):
    """A change late in a long line is shown, not cut off with the rest of the line."""
    prefix = "WARNING | simulation_1-results_manager:_warn_check_stratifications:"
    reference_cells, candidate_cells = standard_cells(), standard_cells()
    reference_cells[PRINT_CELL].outputs = [stdout(f"{prefix}462 - Specified excluded\n")]
    candidate_cells[PRINT_CELL].outputs = [stdout(f"{prefix}500 - Specified excluded\n")]
    reference_dir, candidate_dir = _write_pair(tmp_path, reference_cells, candidate_cells)

    result = compare_notebooks(reference_dir, candidate_dir)

    (difference,) = result.differences
    reference_shown, candidate_shown = difference.detail.split(" -> ")
    assert "462" in reference_shown
    assert "500" in candidate_shown


def _log_and_timing_output(time, elapsed, cpu, wall, colored, message="Conflicting"):
    if colored:
        time, elapsed = f"\x1b[32m{time}\x1b[0m", f"\x1b[32m{elapsed}\x1b[0m"
    return stdout(
        f"{time} | {elapsed} | WARNING  | sim-values_manager:_warn:488 - {message}\n"
        f"CPU times: user {cpu} s, sys: 1.52 s, total: 15 s\n"
        f"Wall time: {wall} s\n"
    )


@pytest.mark.parametrize("colored", [False, True], ids=["plain", "colored"])
def test_log_times_and_cell_timings_are_masked(tmp_path, colored):
    """Output that differs only by the time and elapsed time that start a loguru line,
    or by the CPU and wall times %%time prints, compares clean."""
    reference_cells, candidate_cells = standard_cells(), standard_cells()
    reference_cells[PRINT_CELL].outputs = [
        _log_and_timing_output(
            "2026-10-27 00:53:45.328", "0:00:24.030555", 13.5, 17.1, colored
        )
    ]
    candidate_cells[PRINT_CELL].outputs = [
        _log_and_timing_output(
            "2026-10-27 01:27:40.912", "0:00:26.123218", 14.5, 18.6, colored
        )
    ]
    reference_dir, candidate_dir = _write_pair(tmp_path, reference_cells, candidate_cells)

    result = compare_notebooks(reference_dir, candidate_dir)

    assert result.differences == []


@pytest.mark.parametrize(
    "reference_date, candidate_date",
    [
        ("Wed 30 Sep 2026 10:29:29 AM PDT", "Tue 27 Oct 2026 01:27:40 AM PDT"),
        ("Wed Sep 30 10:29:29 PDT 2026", "Tue Oct  7 01:27:40 PDT 2026"),
    ],
    ids=["en_US", "C"],
)
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_date_lines_are_masked(tmp_path, reference_date, candidate_date, newline):
    """Output that differs only by a whole line of `date` output compares clean,
    including the \\r\\n line endings that `!date` in a notebook produces."""
    reference_cells, candidate_cells = standard_cells(), standard_cells()
    reference_cells[PRINT_CELL].outputs = [
        stdout(f"hjafari{newline}{reference_date}{newline}")
    ]
    candidate_cells[PRINT_CELL].outputs = [
        stdout(f"hjafari{newline}{candidate_date}{newline}")
    ]
    reference_dir, candidate_dir = _write_pair(tmp_path, reference_cells, candidate_cells)

    result = compare_notebooks(reference_dir, candidate_dir)

    assert result.differences == []


def test_dates_within_other_text_are_not_masked(tmp_path):
    """A date that is part of a longer line is real output and is still compared."""
    reference_cells, candidate_cells = standard_cells(), standard_cells()
    reference_cells[PRINT_CELL].outputs = [stdout("Ran on Wed 30 Sep 2026 10:29:29 AM PDT\n")]
    candidate_cells[PRINT_CELL].outputs = [stdout("Ran on Tue 27 Oct 2026 01:27:40 AM PDT\n")]
    reference_dir, candidate_dir = _write_pair(tmp_path, reference_cells, candidate_cells)

    result = compare_notebooks(reference_dir, candidate_dir)

    assert [d.kind for d in result.differences] == [DifferenceKind.OUTPUT]


def test_log_message_changes_are_not_masked(tmp_path):
    """A log line whose message changed is still reported, even as its time is masked."""
    reference_cells, candidate_cells = standard_cells(), standard_cells()
    reference_cells[PRINT_CELL].outputs = [
        _log_and_timing_output("2026-10-27 00:53:45.328", "0:00:24.030555", 1, 1, False)
    ]
    candidate_cells[PRINT_CELL].outputs = [
        _log_and_timing_output(
            "2026-10-27 01:27:40.912", "0:00:26.123218", 1, 1, False, message="Other"
        )
    ]
    reference_dir, candidate_dir = _write_pair(tmp_path, reference_cells, candidate_cells)

    result = compare_notebooks(reference_dir, candidate_dir)

    (difference,) = result.differences
    assert difference.kind == DifferenceKind.OUTPUT
    assert "Other" in difference.detail


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


def test_invalid_notebook_file_raises(tmp_path):
    """A notebook file that is not valid notebook JSON is an input error naming it."""
    reference_dir, candidate_dir = _write_pair(tmp_path, standard_cells(), standard_cells())
    (candidate_dir / NOTEBOOK).write_text("not a notebook\n")

    with pytest.raises(ComparisonInputError, match="Could not read candidate notebook"):
        compare_notebooks(reference_dir, candidate_dir)


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
# Diff pages            #
#########################


def _page_text(path):
    """A diff page's visible text: tags removed, so highlighted characters join up."""
    return html.unescape(re.sub(r"<[^>]+>", "", path.read_text())).replace("\xa0", " ")


def test_diff_page_written_for_moved_notebook_only(tmp_path):
    """With diff_dir, a moved notebook gets a page showing its moved cells' source
    and changed lines; a notebook that did not move gets none."""
    candidate = standard_cells()
    candidate[PRINT_CELL].outputs = [stdout("43\n")]
    reference_dir, candidate_dir = _write_pair(tmp_path, standard_cells(), candidate)
    diff_dir = tmp_path / "diffs"

    result = compare_notebooks(reference_dir, candidate_dir, diff_dir=diff_dir)

    page = diff_dir / "analysis.html"
    assert result.diff_files == {NOTEBOOK: page}
    assert sorted(p.name for p in diff_dir.iterdir()) == ["analysis.html"]
    text = _page_text(page)
    assert "code cell 1" in text
    assert "print(42)" in text
    assert "43" in text
    # Cells that did not move are not shown.
    assert "code cell 2" not in text


def test_diff_page_shows_plotted_data_and_images(tmp_path):
    """A plot change shows the plotted data side by side and both sides' images."""
    reference_dir, candidate_dir = _write_pair(
        tmp_path,
        _plot_cell_with(plot_data(bar_height=2.0)),
        _plot_cell_with(plot_data(bar_height=2.75)),
    )

    result = compare_notebooks(reference_dir, candidate_dir, diff_dir=tmp_path / "diffs")

    page = result.diff_files[NOTEBOOK]
    assert "plotted data" in _page_text(page)
    assert '"height": 2.75' in _page_text(page)
    assert page.read_text().count("data:image/png;base64,") == 2


def test_diff_page_for_schema_change_shows_sources(tmp_path):
    """When cell sources differ, the page shows the sources side by side."""
    candidate = standard_cells()
    candidate[PRINT_CELL].source = "print(43)"
    reference_dir, candidate_dir = _write_pair(tmp_path, standard_cells(), candidate)

    result = compare_notebooks(reference_dir, candidate_dir, diff_dir=tmp_path / "diffs")

    text = _page_text(result.diff_files[NOTEBOOK])
    assert "print(42)" in text and "print(43)" in text


def test_stale_diff_page_is_removed(tmp_path):
    """A page from an earlier comparison of a notebook that no longer moved is
    removed; other files in the directory are kept."""
    reference_dir, candidate_dir = _write_pair(tmp_path, standard_cells(), standard_cells())
    diff_dir = tmp_path / "diffs"
    diff_dir.mkdir()
    (diff_dir / "analysis.html").write_text("old")
    (diff_dir / "keep.txt").write_text("mine")

    result = compare_notebooks(reference_dir, candidate_dir, diff_dir=diff_dir)

    assert result.diff_files == {}
    assert sorted(p.name for p in diff_dir.iterdir()) == ["keep.txt"]


def test_no_diff_pages_without_diff_dir(tmp_path):
    """A moved notebook gets no page when no diff_dir is given."""
    candidate = standard_cells()
    candidate[PRINT_CELL].outputs = [stdout("43\n")]
    reference_dir, candidate_dir = _write_pair(tmp_path, standard_cells(), candidate)

    result = compare_notebooks(reference_dir, candidate_dir)

    assert result.moved
    assert result.diff_files == {}


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


def test_execute_refuses_the_notebook_directory_as_output(tmp_path):
    """Executing into the notebook directory itself is an input error, and nothing
    is run or overwritten."""
    source_dir = tmp_path / "source"
    source = write_source_notebook(source_dir, "prints.ipynb", ["print('hello')"])
    before = source.read_bytes()

    with pytest.raises(ComparisonInputError, match="must differ"):
        execute_notebooks(source_dir, source_dir)

    assert source.read_bytes() == before


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


# One notebook per kind of output the model notebooks produce. Each cell reads its
# value from an environment variable, so a run can change one cell's data without
# changing any source (a source change would be a schema change instead). Every
# value is 3.0 unless changed.
KINDS_SETUP_SOURCE = (
    "%matplotlib inline\n"
    "import os\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "from IPython.display import display\n"
    "def value(name):\n"
    "    return float(os.environ.get(f'COMPARE_TEST_{name}', '3.0'))"
)


class Kind(NamedTuple):
    source: str
    kind: DifferenceKind
    detail: str
    """Text expected in the difference's detail when the value changes to 3.5."""
    max_abs_diff: Optional[float] = None
    """For plots, the expected largest change in a plotted value."""


OUTPUT_KINDS = {
    "bar": Kind(
        "plt.bar(['a', 'b', 'c'], [1.0, 2.0, value('BAR')])\nplt.show()",
        DifferenceKind.PLOT,
        "1 plotted value(s) changed",
        0.5,
    ),
    "scatter": Kind(
        "plt.scatter([1.0, 2.0, 3.0], [1.0, value('SCATTER'), 3.0])\nplt.show()",
        DifferenceKind.PLOT,
        "1 plotted value(s) changed",
        0.5,
    ),
    "dataframe": Kind(
        "pd.DataFrame({'x': [1.5, value('DATAFRAME')], 'y': ['p', 'q']})",
        DifferenceKind.OUTPUT,
        "3.5",
    ),
    "print": Kind(
        "print(f\"mean: {value('PRINT')}\")",
        DifferenceKind.OUTPUT,
        "'mean: 3.0' -> 'mean: 3.5'",
    ),
    # display() writes display_data outputs rather than an execute_result.
    "displayed_dataframe": Kind(
        "display(pd.DataFrame({'incidence': [0.5, value('DISPLAYED_DATAFRAME')]}))\n"
        "print('shown')",
        DifferenceKind.OUTPUT,
        "3.5",
    ),
    # A Series has a text/plain output only, no HTML table.
    "series": Kind(
        "frame = pd.DataFrame({'sex': ['F', 'M', 'F'], 'v': [1.0, 2.0, value('SERIES')]})\n"
        "display(frame.groupby('sex').v.mean())\n"
        "frame.v.describe()",
        DifferenceKind.OUTPUT,
        "2.25",
    ),
    "scalar": Kind(
        "np.float64(value('SCALAR')) / 8",
        DifferenceKind.OUTPUT,
        "'np.float64(0.375)' -> 'np.float64(0.4375)'",
    ),
    "pandas_bar": Kind(
        "frame = pd.DataFrame({'group': ['a', 'b'], 'sim': [1.0, value('PANDAS_BAR')],"
        " 'artifact': [1.1, 2.9]})\n"
        "frame.plot(kind='bar', x='group', y=['sim', 'artifact'])\n"
        "plt.show()",
        DifferenceKind.PLOT,
        "1 plotted value(s) changed",
        0.5,
    ),
    "line": Kind(
        "plt.plot([1.0, 2.0, 3.0], [1.0, value('LINE'), 3.0], marker='o', label='sim')\n"
        "plt.legend()\n"
        "plt.show()",
        DifferenceKind.PLOT,
        "1 plotted value(s) changed",
        0.5,
    ),
    # Only the error bar moves: both ends of one whisker, by a tenth of the change.
    "errorbar": Kind(
        "plt.errorbar([1.0, 2.0], [1.0, 2.0], yerr=[0.1, value('ERRORBAR') / 10], fmt='o')\n"
        "plt.show()",
        DifferenceKind.PLOT,
        "2 plotted value(s) changed",
        0.05,
    ),
    # Both ends of the reference line move.
    "axhline": Kind(
        "plt.plot([0.0, 1.0], [0.0, 1.0])\nplt.axhline(value('AXHLINE'))\nplt.show()",
        DifferenceKind.PLOT,
        "2 plotted value(s) changed",
        0.5,
    ),
    # One figure per loop pass; the change must be named by its own figure's title.
    "figures_in_a_loop": Kind(
        "for k in range(2):\n"
        "    plt.plot([0.0, 1.0], [0.0, value('FIGURES_IN_A_LOOP') if k else 1.0])\n"
        "    plt.title(f'outcome {k}')\n"
        "    plt.show()",
        DifferenceKind.PLOT,
        "'outcome 1': 1 plotted value(s) changed",
        0.5,
    ),
    "subplots": Kind(
        "fig, axs = plt.subplots(1, 2)\n"
        "axs[0].bar(['a'], [1.0])\n"
        "axs[0].set_title('left')\n"
        "axs[1].bar(['a'], [value('SUBPLOTS')])\n"
        "axs[1].set_title('right')\n"
        "plt.show()",
        DifferenceKind.PLOT,
        "'right': 1 plotted value(s) changed",
        0.5,
    ),
    # Last but one: when it fails, the cell after it does not run.
    "failed_assert": Kind(
        "assert value('FAILED_ASSERT') < 3.2, 'ratio outside the acceptable range'\n"
        "print('checks passed')",
        DifferenceKind.EXECUTION_FAILED,
        "AssertionError: ratio outside the acceptable range; later cells not compared",
    ),
}

KINDS_NOTEBOOK_SOURCE = [
    KINDS_SETUP_SOURCE,
    *(kind.source for kind in OUTPUT_KINDS.values()),
    "print('after the checks')",
]


def _kind_cell(name):
    """The code cell number of an output kind, counting the setup cell as 1."""
    return list(OUTPUT_KINDS).index(name) + 2


def _execute_kinds_notebook(directory, **values):
    """Execute the kinds notebook into ``directory`` with the given cell values.

    Returns:
        The directory, and the notebook's error or None.
    """
    source_dir = directory.parent / f"{directory.name}_source"
    write_source_notebook(source_dir, "kinds.ipynb", KINDS_NOTEBOOK_SOURCE)
    with pytest.MonkeyPatch.context() as patch:
        for name, value in values.items():
            patch.setenv(f"COMPARE_TEST_{name.upper()}", repr(value))
        return directory, execute_notebooks(source_dir, directory)["kinds.ipynb"]


@pytest.fixture(scope="module")
def kinds_reference(tmp_path_factory):
    require_kernel()
    reference, error = _execute_kinds_notebook(tmp_path_factory.mktemp("kinds") / "reference")
    assert error is None
    return reference


def test_kinds_notebook_runs_compare_clean(kinds_reference, tmp_path):
    """Two runs of the kinds notebook with the same values compare clean, and every
    plotting cell recorded plot data."""
    candidate, error = _execute_kinds_notebook(tmp_path / "candidate")

    result = compare_notebooks(kinds_reference, candidate)

    assert error is None
    assert result.differences == []
    assert result.notes == []
    n_plot_cells = sum(kind.kind == DifferenceKind.PLOT for kind in OUTPUT_KINDS.values())
    assert len(plot_data_outputs(read_notebook(candidate / "kinds.ipynb"))) == n_plot_cells


@pytest.mark.parametrize("name", OUTPUT_KINDS)
def test_changed_output_kind_is_reported(kinds_reference, tmp_path, name):
    """Changing the value behind one kind of output is reported once, against that
    cell, with the expected kind and detail."""
    kind = OUTPUT_KINDS[name]
    candidate, error = _execute_kinds_notebook(tmp_path / "candidate", **{name: 3.5})

    result = compare_notebooks(kinds_reference, candidate)

    assert (error is not None) == (kind.kind == DifferenceKind.EXECUTION_FAILED)
    assert len(result.differences) == 1, result.differences
    difference = result.differences[0]
    assert difference.kind == kind.kind
    assert f"code cell {_kind_cell(name)} " in difference.item
    assert kind.detail in difference.detail
    if kind.max_abs_diff is not None:
        assert difference.value_diff.max_abs_diff == pytest.approx(kind.max_abs_diff)


@pytest.mark.parametrize("name", ["bar", "scatter"])
def test_plotted_change_within_tolerance_is_clean(kinds_reference, tmp_path, name):
    """A plotted value that moves by one float step is within the default rtol."""
    candidate, _ = _execute_kinds_notebook(
        tmp_path / "candidate", **{name: math.nextafter(3.0, 4.0)}
    )

    assert compare_notebooks(kinds_reference, candidate).differences == []
    assert compare_notebooks(kinds_reference, candidate, rtol=0).moved


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


def test_execute_writes_side_effect_files_beside_outputs(tmp_path):
    """Files a notebook writes to its working directory land in the output directory,
    not beside the source notebooks."""
    require_kernel()
    source_dir, output_dir = tmp_path / "source", tmp_path / "executed"
    write_source_notebook(
        source_dir, "writes.ipynb", ["open('side_effect.txt', 'w').write('x')"]
    )

    statuses = execute_notebooks(source_dir, output_dir)

    assert statuses == {"writes.ipynb": None}
    assert (output_dir / "side_effect.txt").is_file()
    assert not (source_dir / "side_effect.txt").exists()


def test_execute_rejects_kernel_importing_another_checkout(tmp_path):
    """Notebooks inside a checkout whose package the kernel does not import are not
    run, and the error names both locations."""
    require_kernel()
    checkout = tmp_path / "other_checkout"
    (checkout / "src" / "vivarium_gates_mncnh").mkdir(parents=True)
    (checkout / "src" / "vivarium_gates_mncnh" / "__init__.py").write_text("")
    source_dir, output_dir = checkout / "tests" / "notebooks", tmp_path / "executed"
    write_source_notebook(source_dir, "prints.ipynb", ["print('hello')"])

    with pytest.raises(ComparisonInputError, match="vivarium_gates_mncnh") as error:
        execute_notebooks(source_dir, output_dir)

    assert str(checkout / "src" / "vivarium_gates_mncnh") in str(error.value)
    assert not (output_dir / "prints.ipynb").exists()


def test_execute_accepts_kernel_importing_the_same_checkout(tmp_path):
    """Notebooks inside the checkout the kernel imports from pass the check."""
    require_kernel()
    import vivarium_gates_mncnh

    checkout = Path(vivarium_gates_mncnh.__file__).resolve().parents[2]
    source_dir = checkout / "tests" / "model_notebooks"
    if not (checkout / "src" / "vivarium_gates_mncnh" / "__init__.py").is_file():
        pytest.skip("package is not imported from a source checkout")
    from vivarium_gates_mncnh.tools.compare.notebooks import _check_kernel_package

    _check_kernel_package(source_dir, "python3")


@pytest.mark.parametrize(
    "pattern",
    ["b_analysis", "b_analysis.ipynb", "b_*", "model_notebooks/interactive/b_analysis.ipynb"],
    ids=["stem", "file name", "glob", "path"],
)
def test_notebooks_selects_by_name_or_path(tmp_path, pattern):
    """Only the selected notebook is compared, however it is named."""
    changed_cells = standard_cells()
    changed_cells[PRINT_CELL].outputs = [stdout("43\n")]
    for name in ["a_analysis.ipynb", "b_analysis.ipynb"]:
        write_notebook(tmp_path / "reference", name, standard_cells())
        write_notebook(tmp_path / "candidate", name, changed_cells)

    result = compare_notebooks(
        tmp_path / "reference", tmp_path / "candidate", notebooks=[pattern]
    )

    assert result.n_compared == 1
    assert result.differences
    assert all(d.item.startswith("b_analysis.ipynb") for d in result.differences)
    assert result.notes == [f"Compared only the notebooks matching {pattern!r}: 1 of 2."]


def test_notebooks_pattern_matching_nothing_raises(tmp_path):
    """A pattern that matches no notebook is an input error, not a clean comparison."""
    write_notebook(tmp_path / "reference", "analysis.ipynb", standard_cells())
    write_notebook(tmp_path / "candidate", "analysis.ipynb", standard_cells())

    with pytest.raises(ComparisonInputError, match="missing"):
        compare_notebooks(
            tmp_path / "reference", tmp_path / "candidate", notebooks=["missing"]
        )


def test_execute_runs_only_selected_notebooks(tmp_path):
    """Only the notebooks that match a pattern are executed and written."""
    require_kernel()
    source_dir = tmp_path / "source"
    write_source_notebook(source_dir, "a_ok.ipynb", ["print('a ran')"])
    write_source_notebook(source_dir, "b_ok.ipynb", ["print('b ran')"])
    output_dir = tmp_path / "executed"

    statuses = execute_notebooks(source_dir, output_dir, notebooks=["b_ok"])

    assert statuses == {"b_ok.ipynb": None}
    assert not (output_dir / "a_ok.ipynb").exists()
    assert (output_dir / "b_ok.ipynb").is_file()
