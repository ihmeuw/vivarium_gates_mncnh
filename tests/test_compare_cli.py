"""End-to-end tests of the compare_outputs command line interface."""

import pytest
from click.testing import CliRunner

from vivarium_gates_mncnh.tools import cli
from vivarium_gates_mncnh.tools.cli import compare_outputs
from vivarium_gates_mncnh.tools.compare import ComparisonResult

from .compare_helpers import (
    PLOTTING_NOTEBOOK_SOURCE,
    PRINT_CELL,
    make_artifact,
    make_results_dir,
    plot_data_outputs,
    read_notebook,
    require_kernel,
    standard_artifact_data,
    standard_cells,
    standard_measures,
    stdout,
    write_notebook,
    write_source_notebook,
)


def _invoke(*args):
    return CliRunner().invoke(compare_outputs, [str(arg) for arg in args])


def _changed_measures(factor: float = 1.5):
    measures = standard_measures()
    deaths = measures["deaths"].copy()
    deaths.loc[4, "value"] *= factor
    return {**measures, "deaths": deaths}


def test_results_clean_exits_zero(tmp_path):
    """`compare_outputs results` on identical directories prints a clean report and
    exits 0."""
    reference = make_results_dir(tmp_path / "reference", standard_measures())
    candidate = make_results_dir(tmp_path / "candidate", standard_measures())

    result = _invoke("results", reference, candidate)

    assert result.exit_code == 0, result.output
    assert "clean" in result.output.lower()


def test_results_moved_exits_one(tmp_path):
    """`compare_outputs results` on differing directories prints the moved measure and
    exits 1."""
    reference = make_results_dir(tmp_path / "reference", standard_measures())
    candidate = make_results_dir(tmp_path / "candidate", _changed_measures())

    result = _invoke("results", reference, candidate)

    assert result.exit_code == 1, result.output
    # Exit 1 means "moved", not an unhandled error.
    assert isinstance(result.exception, SystemExit)
    assert "deaths" in result.output
    assert "moved" in result.output.lower()


@pytest.mark.parametrize(
    "options", [["--rtol", "1e-6"], ["--atol", "1e-3"]], ids=["rtol", "atol"]
)
def test_tolerance_options_passed_through(tmp_path, options):
    """--rtol/--atol suppress a small change, turning exit 1 into exit 0."""
    reference = make_results_dir(tmp_path / "reference", standard_measures())
    candidate = make_results_dir(tmp_path / "candidate", _changed_measures(1 + 1e-9))

    by_default = _invoke("results", reference, candidate)
    assert by_default.exit_code == 1, by_default.output

    with_tolerance = _invoke("results", reference, candidate, *options)
    assert with_tolerance.exit_code == 0, with_tolerance.output


@pytest.mark.parametrize(
    "options, expected",
    [([], "rtol=1e-12, atol=0"), (["--rtol", "0", "--atol", "0.5"], "rtol=0, atol=0.5")],
    ids=["default", "given"],
)
def test_report_states_tolerance(tmp_path, options, expected):
    """The report says which tolerance it compared with, so a clean result can be read
    as "nothing moved beyond this"."""
    reference = make_results_dir(tmp_path / "reference", standard_measures())
    candidate = make_results_dir(tmp_path / "candidate", standard_measures())

    result = _invoke("results", reference, candidate, *options)

    assert result.exit_code == 0, result.output
    assert f"Tolerance: {expected}" in result.stdout


@pytest.mark.parametrize(
    "command", ["results", "artifacts", "notebooks", "execute-notebooks"]
)
def test_bad_input_exits_two(tmp_path, command):
    """A missing path exits 2 with a one-line message rather than a traceback."""
    missing = tmp_path / "does_not_exist"
    if command == "results":
        existing = make_results_dir(tmp_path / "reference", standard_measures())
        args = ["results", existing, missing]
    elif command == "artifacts":
        existing = make_artifact(tmp_path / "reference.hdf", standard_artifact_data())
        args = ["artifacts", existing, missing / "candidate.hdf"]
    elif command == "notebooks":
        existing = tmp_path / "reference"
        write_notebook(existing, "analysis.ipynb", standard_cells())
        args = ["notebooks", existing, missing]
    else:
        args = ["execute-notebooks", tmp_path / "executed", "--notebook-dir", missing]

    result = _invoke(*args)

    assert result.exit_code == 2, result.output
    assert isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output
    message_lines = [line for line in result.output.splitlines() if line.strip()]
    assert len(message_lines) == 1, result.output
    if command == "artifacts":
        assert not (missing / "candidate.hdf").exists()


def test_unexpected_error_exits_two_naming_its_type(tmp_path, monkeypatch):
    """A failure other than an input error still exits 2, not 1, with a one-line
    message that names the exception type."""

    def fail(*args, **kwargs):
        raise RuntimeError("disk vanished")

    monkeypatch.setattr(cli, "compare_results", fail)

    result = _invoke("results", tmp_path, tmp_path)

    assert result.exit_code == 2, result.output
    assert result.stderr.strip() == "Error: RuntimeError: disk vanished"
    assert result.stdout == ""


def test_artifacts_command(tmp_path):
    """`compare_outputs artifacts` exits 0 on identical artifacts and 1 on changed
    ones."""
    data = standard_artifact_data()
    reference = make_artifact(tmp_path / "reference.hdf", data)
    same = make_artifact(tmp_path / "same.hdf", data)
    changed_frame = data["population.structure"] * 1.25
    changed = make_artifact(
        tmp_path / "changed.hdf", {**data, "population.structure": changed_frame}
    )

    clean = _invoke("artifacts", reference, same)
    assert clean.exit_code == 0, clean.output
    assert "clean" in clean.output.lower()

    moved = _invoke("artifacts", reference, changed)
    assert moved.exit_code == 1, moved.output
    assert isinstance(moved.exception, SystemExit)
    assert "population.structure" in moved.output


def test_notebooks_command(tmp_path):
    """`compare_outputs notebooks` exits 0 on identical directories and 1 on changed
    ones."""
    reference = tmp_path / "reference"
    same = tmp_path / "same"
    changed = tmp_path / "changed"
    write_notebook(reference, "analysis.ipynb", standard_cells())
    write_notebook(same, "analysis.ipynb", standard_cells())
    changed_cells = standard_cells()
    changed_cells[PRINT_CELL].outputs = [stdout("43\n")]
    write_notebook(changed, "analysis.ipynb", changed_cells)

    clean = _invoke("notebooks", reference, same)
    assert clean.exit_code == 0, clean.output
    assert "clean" in clean.output.lower()

    moved = _invoke("notebooks", reference, changed)
    assert moved.exit_code == 1, moved.output
    assert isinstance(moved.exception, SystemExit)
    assert "analysis" in moved.output


def _moved_notebook_dirs(tmp_path, name="pair"):
    reference, candidate = tmp_path / name / "reference", tmp_path / name / "candidate"
    write_notebook(reference, "analysis.ipynb", standard_cells())
    changed_cells = standard_cells()
    changed_cells[PRINT_CELL].outputs = [stdout("43\n")]
    write_notebook(candidate, "analysis.ipynb", changed_cells)
    return reference, candidate


def test_notebooks_report_links_diff_pages(tmp_path):
    """With --report, `compare_outputs notebooks` writes a diff page beside the report,
    links it, and prints its absolute path to stderr; without --report it writes none."""
    reference, candidate = _moved_notebook_dirs(tmp_path)
    report_path = tmp_path / "out" / "notebooks.md"

    moved = _invoke("notebooks", reference, candidate, "--report", report_path)

    assert moved.exit_code == 1, moved.output
    assert (tmp_path / "out" / "notebooks_diffs" / "analysis.html").is_file()
    assert "[analysis.ipynb](notebooks_diffs/analysis.html)" in report_path.read_text()
    page = (tmp_path / "out" / "notebooks_diffs" / "analysis.html").resolve()
    assert f"Diff page for analysis.ipynb: {page}" in moved.stderr
    assert str(page) not in moved.stdout

    no_report = _invoke("notebooks", reference, candidate)
    assert "Diff page" not in no_report.output


def test_all_writes_diff_pages_per_notebooks_pair(tmp_path):
    """`compare_outputs all` gives each --notebooks pair its own diff page directory."""
    first = _moved_notebook_dirs(tmp_path, "first")
    second = _moved_notebook_dirs(tmp_path, "second")
    report_path = tmp_path / "out" / "all.md"

    result = _invoke(
        "all",
        *("--notebooks", *first),
        *("--notebooks", *second),
        *("--report", report_path),
    )

    assert result.exit_code == 1, result.output
    report = report_path.read_text()
    for k in (1, 2):
        assert (tmp_path / "out" / "all_diffs" / str(k) / "analysis.html").is_file()
        assert f"(all_diffs/{k}/analysis.html)" in report
        page = (tmp_path / "out" / "all_diffs" / str(k) / "analysis.html").resolve()
        assert str(page) in result.stderr


def test_execute_notebooks_command(tmp_path):
    """`compare_outputs execute-notebooks` writes executed notebooks to OUTPUT_DIR,
    exiting 0 when all run and 1 when any fails."""
    require_kernel()
    passing_dir = tmp_path / "passing"
    write_source_notebook(passing_dir, "plots.ipynb", PLOTTING_NOTEBOOK_SOURCE)
    failing_dir = tmp_path / "failing"
    write_source_notebook(failing_dir, "a_ok.ipynb", ["print('ok')"])
    write_source_notebook(failing_dir, "b_fails.ipynb", ["raise ValueError('boom')"])

    all_ran = _invoke(
        "execute-notebooks", tmp_path / "out_passing", "--notebook-dir", passing_dir
    )
    assert all_ran.exit_code == 0, all_ran.output
    executed = read_notebook(tmp_path / "out_passing" / "plots.ipynb")
    assert plot_data_outputs(executed)

    some_failed = _invoke(
        "execute-notebooks", tmp_path / "out_failing", "--notebook-dir", failing_dir
    )
    assert some_failed.exit_code == 1, some_failed.output
    assert isinstance(some_failed.exception, SystemExit)
    assert (tmp_path / "out_failing" / "a_ok.ipynb").is_file()
    assert (tmp_path / "out_failing" / "b_fails.ipynb").is_file()
    assert "b_fails" in some_failed.output


@pytest.mark.parametrize("bad_side", ["reference", "candidate"])
def test_unreadable_input_exits_two(tmp_path, bad_side):
    """A results measure that is not a readable parquet file exits 2 with a one-line
    message, not 1 and not a traceback."""
    good = make_results_dir(tmp_path / "good", standard_measures())
    bad = make_results_dir(tmp_path / "bad", standard_measures())
    (bad / "deaths.parquet").write_bytes(b"this is not a parquet file\n" * 8)
    paths = (bad, good) if bad_side == "reference" else (good, bad)

    result = _invoke("results", *paths)

    assert result.exit_code == 2, result.output
    assert isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output
    message_lines = [line for line in result.output.splitlines() if line.strip()]
    assert len(message_lines) == 1, result.output


@pytest.mark.parametrize("problem", ["no_matching_notebook", "output_is_source"])
def test_execute_notebooks_input_errors_exit_two(tmp_path, problem):
    """`execute-notebooks` exits 2 without running anything when --notebook matches
    nothing or OUTPUT_DIR is the notebook directory."""
    source_dir = tmp_path / "source"
    source = write_source_notebook(source_dir, "a_ok.ipynb", ["print('ok')"])
    before = source.read_bytes()
    if problem == "no_matching_notebook":
        output_dir = tmp_path / "executed"
        args = ["--notebook-dir", source_dir, "--notebook", "no_such_notebook"]
        expected = "no_such_notebook"
    else:
        output_dir = source_dir
        args = ["--notebook-dir", source_dir]
        expected = "must differ"

    result = _invoke("execute-notebooks", output_dir, *args)

    assert result.exit_code == 2, result.output
    assert expected in result.stderr
    assert "Traceback" not in result.output
    assert source.read_bytes() == before
    assert not (tmp_path / "executed").exists()


@pytest.mark.parametrize(
    "kernel_args, expected", [((), "python3"), (("--kernel", "mncnh"), "mncnh")]
)
def test_execute_notebooks_runs_the_named_kernel(
    tmp_path, monkeypatch, kernel_args, expected
):
    """`execute-notebooks` runs the notebooks with --kernel, or python3 by default."""
    used = {}

    def record_kernel(notebook_dir, output_dir, kernel_name, notebooks):
        used["kernel"] = kernel_name
        return {"a_ok.ipynb": None}

    monkeypatch.setattr(cli, "execute_notebooks", record_kernel)

    result = _invoke(
        "execute-notebooks", tmp_path / "out", "--notebook-dir", tmp_path, *kernel_args
    )

    assert result.exit_code == 0, result.output
    assert used == {"kernel": expected}


def test_report_option_writes_report(tmp_path):
    """--report writes the printed report to a file, creating its directory."""
    reference = make_results_dir(tmp_path / "reference", standard_measures())
    candidate = make_results_dir(tmp_path / "candidate", _changed_measures())
    report_path = tmp_path / "reports" / "results.md"

    result = _invoke("results", reference, candidate, "--report", report_path)

    assert result.exit_code == 1, result.output
    assert report_path.read_text() == result.stdout


def test_report_option_records_error(tmp_path):
    """--report still writes a file when the comparison fails, holding the error."""
    reference = make_results_dir(tmp_path / "reference", standard_measures())
    report_path = tmp_path / "report.md"

    result = _invoke("results", reference, tmp_path / "missing", "--report", report_path)

    assert result.exit_code == 2, result.output
    report = report_path.read_text()
    assert report.startswith("Error: ")
    # The file also names what was being compared, since it may be read on its own.
    assert f"Comparison: results {reference} vs {tmp_path / 'missing'}" in report


SELECTIONS = {
    "results": ("--measure", "births"),
    "artifacts": ("--key", "metadata.*"),
    "notebooks": ("--notebook", "other"),
}


def _pair_changed_outside_selection(directory, command):
    """A reference and candidate for ``command`` that differ only outside
    SELECTIONS[command]."""
    if command == "results":
        reference = make_results_dir(directory / "reference", standard_measures())
        candidate = make_results_dir(directory / "candidate", _changed_measures())
    elif command == "artifacts":
        data = standard_artifact_data()
        reference = make_artifact(directory / "reference.hdf", data)
        changed = {**data, "population.structure": data["population.structure"] * 2}
        candidate = make_artifact(directory / "candidate.hdf", changed)
    else:
        reference, candidate = directory / "reference", directory / "candidate"
        changed_cells = standard_cells()
        changed_cells[PRINT_CELL].outputs = [stdout("43\n")]
        for side, cells in [(reference, standard_cells()), (candidate, changed_cells)]:
            write_notebook(side, "analysis.ipynb", cells)
            write_notebook(side, "other.ipynb", standard_cells())
    return reference, candidate


@pytest.mark.parametrize(
    "command, option, pattern",
    [(command, *selection) for command, selection in SELECTIONS.items()],
)
def test_selection_options(tmp_path, command, option, pattern):
    """A selection option limits a comparison to what it matches, so a change
    elsewhere does not register; a pattern that matches nothing exits 2."""
    reference, candidate = _pair_changed_outside_selection(tmp_path, command)

    unselected = _invoke(command, reference, candidate)
    assert unselected.exit_code == 1, unselected.output

    selected = _invoke(command, reference, candidate, option, pattern)
    assert selected.exit_code == 0, selected.output

    no_match = _invoke(command, reference, candidate, option, "no_such_thing")
    assert no_match.exit_code == 2, no_match.output
    assert "no_such_thing" in no_match.output


def test_all_applies_each_selection_to_its_comparisons(tmp_path):
    """In `all`, --measure, --key and --notebook limit every results, artifacts and
    notebooks comparison, so changes outside them do not register."""
    args = ["all"]
    for command in SELECTIONS:
        args += [
            f"--{command}",
            *_pair_changed_outside_selection(tmp_path / command, command),
        ]

    unselected = _invoke(*args)
    assert unselected.exit_code == 1, unselected.output

    selected = _invoke(
        *args, *(arg for selection in SELECTIONS.values() for arg in selection)
    )
    assert selected.exit_code == 0, selected.output
    assert "3 comparisons, 3 clean, 0 moved, 0 failed" in selected.stdout


def test_all_combines_comparisons(tmp_path):
    """`compare_outputs all` reports every comparison under one summary and exits
    with the worst status: 0 clean, 1 moved."""
    reference = make_results_dir(tmp_path / "reference", standard_measures())
    same = make_results_dir(tmp_path / "same", standard_measures())
    changed = make_results_dir(tmp_path / "changed", _changed_measures())
    data = standard_artifact_data()
    reference_artifact = make_artifact(tmp_path / "reference.hdf", data)
    same_artifact = make_artifact(tmp_path / "same.hdf", data)
    report_path = tmp_path / "report.md"

    clean = _invoke(
        "all",
        *("--results", reference, same),
        *("--artifacts", reference_artifact, same_artifact),
        *("--report", report_path),
    )
    assert clean.exit_code == 0, clean.output
    assert "2 comparisons, 2 clean, 0 moved, 0 failed" in clean.output
    assert report_path.read_text() == clean.stdout

    moved = _invoke(
        "all",
        *("--results", reference, same),
        *("--results", reference, changed),
        *("--artifacts", reference_artifact, same_artifact),
    )
    assert moved.exit_code == 1, moved.output
    assert "3 comparisons, 2 clean, 1 moved, 0 failed" in moved.output


def test_all_continues_after_a_failed_comparison(tmp_path):
    """A failed comparison is listed in the report with its inputs, the others still
    run, and the exit status is 2."""
    reference = make_results_dir(tmp_path / "reference", standard_measures())
    changed = make_results_dir(tmp_path / "changed", _changed_measures())
    missing = tmp_path / "missing"
    report_path = tmp_path / "report.md"

    result = _invoke(
        "all",
        *("--results", reference, missing),
        *("--results", reference, changed),
        *("--report", report_path),
    )

    assert result.exit_code == 2, result.output
    report = report_path.read_text()
    assert "2 comparisons, 0 clean, 1 moved, 1 failed" in report
    assert "## Errors" in report
    assert f"results {reference} vs {missing}: Error:" in report
    assert "deaths" in report


def test_all_requires_a_comparison():
    """`compare_outputs all` with nothing to compare is a usage error (exit 2)."""
    result = _invoke("all")

    assert result.exit_code == 2
    assert "--results" in result.output


def test_execute_notebooks_selection(tmp_path):
    """`execute-notebooks --notebook` executes only the matching notebooks."""
    require_kernel()
    source_dir = tmp_path / "source"
    write_source_notebook(source_dir, "a_ok.ipynb", ["print('a')"])
    write_source_notebook(source_dir, "b_ok.ipynb", ["print('b')"])

    result = _invoke(
        "execute-notebooks",
        tmp_path / "out",
        "--notebook-dir",
        source_dir,
        "--notebook",
        "a_ok",
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "a_ok.ipynb").is_file()
    assert not (tmp_path / "out" / "b_ok.ipynb").exists()
