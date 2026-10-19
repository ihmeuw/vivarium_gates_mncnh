"""Tests for notebook execution and comparison in vivarium_gates_mncnh.tools.compare.notebooks.

Comparison tests build executed notebooks with nbformat; only the execution tests
need a Jupyter kernel.
"""


def test_discover_skips_checkpoints_and_merge_files():
    """discover_notebook_paths returns top-level notebooks sorted, skipping
    checkpoints and nbdime _BACKUP_/_BASE_/_LOCAL_/_REMOTE_ files."""


def test_identical_notebooks_compare_clean():
    """Two directories of identical executed notebooks compare clean."""


def test_ignored_outputs_do_not_register():
    """Differences only in stderr, execution counts, metadata, images or widget
    outputs compare clean."""


def test_changed_stdout_reported():
    """A changed stdout stream is reported as output changed, naming notebook and
    cell."""


def test_changed_text_plain_and_html_reported():
    """Changed text/plain and text/html outputs are reported as output changed."""


def test_error_on_one_side_reported():
    """A cell that errors on only one side is reported as execution failed."""


def test_changed_plot_data_reported():
    """A plot-data output with changed values is reported as plot data changed."""


def test_plot_data_tolerance_is_applied():
    """A small change in plotted values is reported by default and not within
    rtol/atol."""


def test_plot_data_missing_on_one_side_is_a_note():
    """When only one side has plot data for a cell, nothing moves and a note says
    plots were not comparable."""


def test_notebook_only_on_one_side():
    """A notebook present in only one directory is reported as only in reference or
    only in candidate."""


def test_changed_cell_sources_reported():
    """Notebooks whose cell sources differ are reported as a schema change."""


def test_missing_directory_raises():
    """A path that is not a directory, or a directory with no notebooks, raises
    ComparisonInputError."""


def test_execute_writes_outputs_and_plot_data():
    """Executing a small notebook that plots a line, bars and a scatter writes it to
    the output directory with a plot-data output holding the plotted values."""


def test_execute_does_not_modify_sources():
    """Executing leaves the source notebooks unchanged."""


def test_execute_continues_after_failure():
    """A failing notebook is reported with its error and still written; the other
    notebooks still run and map to None."""


def test_executed_runs_compare_clean():
    """Two executions of the same notebooks into separate directories compare clean,
    plot data included."""
