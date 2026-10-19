"""End-to-end tests of the compare_outputs command line interface."""

import pytest
from click.testing import CliRunner

from vivarium_gates_mncnh.tools.cli import compare_outputs

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
