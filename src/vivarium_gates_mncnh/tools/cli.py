import os
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple

import click
from loguru import logger
from vivarium.engine.framework.utilities import handle_exceptions

from vivarium_gates_mncnh.constants import metadata, paths
from vivarium_gates_mncnh.constants.paths import MODEL_NOTEBOOKS_DIR
from vivarium_gates_mncnh.tools import build_artifacts, configure_logging_to_terminal
from vivarium_gates_mncnh.tools.compare import (
    DEFAULT_RTOL,
    ComparisonInputError,
    ComparisonResult,
    compare_artifacts,
    compare_notebooks,
    compare_results,
    execute_notebooks,
    render_report,
)


@click.command()
@click.option(
    "-l",
    "--location",
    default="all",
    show_default=True,
    type=click.Choice(metadata.LOCATIONS + ["all"]),
    help=(
        "Location for which to make an artifact. Note: prefer building archives on the cluster.\n"
        'If you specify location "all" you must be on a cluster node.'
    ),
)
@click.option(
    "--years",
    default=None,
    help=(
        "Years for which to make an artifact. Can be a single year or 'all'. \n"
        "If not specified, make for most recent year."
    ),
)
@click.option(
    "-o",
    "--output-dir",
    default=str(paths.ARTIFACT_ROOT),
    show_default=True,
    type=click.Path(),
    help="Specify an output directory. Directory must exist.",
)
@click.option(
    "-a", "--append", is_flag=True, help="Append to the artifact instead of overwriting."
)
@click.option("-r", "--replace-keys", multiple=True, help="Specify keys to overwrite")
@click.option(
    "--resume",
    is_flag=True,
    help=(
        "Resume the previous '-l all' build in the output directory, rerunning only "
        "the locations that did not finish. Applies to on-cluster '-l all' builds only."
    ),
)
@click.option("-v", "verbose", count=True, help="Configure logging verbosity.")
@click.option(
    "--pdb",
    "with_debugger",
    is_flag=True,
    help="Drop into python debugger if an error occurs.",
)
def make_artifacts(
    location: str,
    years: Optional[str],
    output_dir: str,
    append: bool,
    replace_keys: Tuple[str, ...],
    resume: bool,
    verbose: int,
    with_debugger: bool,
) -> None:
    configure_logging_to_terminal(verbose)
    main = handle_exceptions(build_artifacts, logger, with_debugger=with_debugger)
    main(location, years, output_dir, append or replace_keys, replace_keys, verbose, resume)


_COMPARISON_OPTIONS = [
    click.option(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
        show_default=True,
        help="Relative tolerance; values within it are not reported as moved.",
    ),
    click.option(
        "--atol",
        type=float,
        default=0.0,
        show_default=True,
        help="Absolute tolerance; values within it are not reported as moved.",
    ),
    click.option(
        "--report",
        "report_path",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help=(
            "Also write the report to this file, creating its directory. The file "
            "is written even when a comparison fails, with the error in it."
        ),
    ),
]


def _comparison_options(command):
    for option in reversed(_COMPARISON_OPTIONS):
        command = option(command)
    return command


_MEASURE_OPTION = click.option(
    "--measure",
    "measures",
    multiple=True,
    metavar="PATTERN",
    help="Compare only the measures that match this glob pattern. Repeatable.",
)


def _default_workers() -> int:
    """The CPUs Slurm gave this job, or 1 outside Slurm so a shared node is not swamped."""
    return int(os.environ.get("SLURM_CPUS_PER_TASK", 1))


_WORKERS_OPTION = click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=_default_workers,
    show_default="SLURM_CPUS_PER_TASK, or 1",
    help="Processes to compare results with. The report is the same for any number.",
)
_KEY_OPTION = click.option(
    "--key",
    "keys",
    multiple=True,
    metavar="PATTERN",
    help="Compare only the artifact keys that match this glob pattern. Repeatable.",
)
_NOTEBOOK_OPTION = click.option(
    "--notebook",
    "notebooks",
    multiple=True,
    metavar="PATTERN",
    help=(
        "Use only the notebooks whose file name matches this glob pattern. A name "
        "without .ipynb or a path also works. Repeatable."
    ),
)


@click.group()
def compare_outputs() -> None:
    """Compare two runs' outputs value by value.

    Each comparison prints a Markdown report to stdout and exits 0 when the
    two sides compare clean, 1 when anything moved, and 2 when the inputs
    cannot be compared (for example, a path that does not exist, a file that
    cannot be read, or a selection pattern that matches nothing) or the
    comparison fails for any other reason, with a one-line ``Error: ...``
    message on stderr.
    """


@compare_outputs.command("results")
@click.argument("reference_dir", type=click.Path(path_type=Path))
@click.argument("candidate_dir", type=click.Path(path_type=Path))
@_MEASURE_OPTION
@_WORKERS_OPTION
@_comparison_options
def compare_results_command(
    reference_dir: Path,
    candidate_dir: Path,
    measures: Tuple[str, ...],
    workers: int,
    rtol: float,
    atol: float,
    report_path: Optional[Path],
) -> None:
    """Compare two results directories (one location's results/ each)."""
    _run_comparisons(
        [
            _results_comparison(reference_dir, candidate_dir, measures, rtol, atol, workers),
        ],
        report_path,
    )


@compare_outputs.command("artifacts")
@click.argument("reference_path", type=click.Path(path_type=Path))
@click.argument("candidate_path", type=click.Path(path_type=Path))
@_KEY_OPTION
@_comparison_options
def compare_artifacts_command(
    reference_path: Path,
    candidate_path: Path,
    keys: Tuple[str, ...],
    rtol: float,
    atol: float,
    report_path: Optional[Path],
) -> None:
    """Compare two artifacts key by key."""
    _run_comparisons(
        [_artifacts_comparison(reference_path, candidate_path, keys, rtol, atol)],
        report_path,
    )


@compare_outputs.command("notebooks")
@click.argument("reference_dir", type=click.Path(path_type=Path))
@click.argument("candidate_dir", type=click.Path(path_type=Path))
@_NOTEBOOK_OPTION
@_comparison_options
def compare_notebooks_command(
    reference_dir: Path,
    candidate_dir: Path,
    notebooks: Tuple[str, ...],
    rtol: float,
    atol: float,
    report_path: Optional[Path],
) -> None:
    """Compare two directories of executed notebooks.

    With --report, a diff page for each notebook that moved is written to
    the directory REPORT_diffs beside the report, the report links to it, and
    its absolute path is printed to stderr.
    """
    _run_comparisons(
        [
            _notebooks_comparison(
                reference_dir,
                candidate_dir,
                notebooks,
                rtol,
                atol,
                _diff_dir(report_path),
            )
        ],
        report_path,
    )


@compare_outputs.command("all")
@click.option(
    "--results",
    "results_pairs",
    type=click.Path(path_type=Path),
    nargs=2,
    multiple=True,
    metavar="REFERENCE_DIR CANDIDATE_DIR",
    help="Two results directories to compare. Repeat for each location.",
)
@click.option(
    "--artifacts",
    "artifact_pairs",
    type=click.Path(path_type=Path),
    nargs=2,
    multiple=True,
    metavar="REFERENCE_PATH CANDIDATE_PATH",
    help="Two artifacts to compare. Repeat for each location.",
)
@click.option(
    "--notebooks",
    "notebook_pairs",
    type=click.Path(path_type=Path),
    nargs=2,
    multiple=True,
    metavar="REFERENCE_DIR CANDIDATE_DIR",
    help="Two directories of executed notebooks to compare. Repeatable.",
)
@_MEASURE_OPTION
@_KEY_OPTION
@_NOTEBOOK_OPTION
@_WORKERS_OPTION
@_comparison_options
def compare_all_command(
    results_pairs: Tuple[Tuple[Path, Path], ...],
    artifact_pairs: Tuple[Tuple[Path, Path], ...],
    notebook_pairs: Tuple[Tuple[Path, Path], ...],
    measures: Tuple[str, ...],
    keys: Tuple[str, ...],
    notebooks: Tuple[str, ...],
    workers: int,
    rtol: float,
    atol: float,
    report_path: Optional[Path],
) -> None:
    """Run several comparisons into one report with one exit status.

    The report starts with a summary line and has one section per comparison.
    A comparison that fails does not stop the others: its error is listed in
    the report and on stderr. The exit status is 2 if any comparison failed,
    otherwise 1 if anything moved, otherwise 0. --measure, --key and
    --notebook apply to every results, artifacts and notebooks comparison.
    With --report, notebook diff pages are written as for the notebooks
    command, into one numbered subdirectory per --notebooks pair.
    """
    comparisons = [
        *(
            _results_comparison(*pair, measures, rtol, atol, workers)
            for pair in results_pairs
        ),
        *(_artifacts_comparison(*pair, keys, rtol, atol) for pair in artifact_pairs),
        *(
            _notebooks_comparison(
                *pair, notebooks, rtol, atol, _diff_dir(report_path, k, len(notebook_pairs))
            )
            for k, pair in enumerate(notebook_pairs, start=1)
        ),
    ]
    if not comparisons:
        raise click.UsageError("Give at least one of --results, --artifacts or --notebooks.")
    _run_comparisons(comparisons, report_path)


@compare_outputs.command("execute-notebooks")
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--notebook-dir",
    type=click.Path(path_type=Path),
    default=MODEL_NOTEBOOKS_DIR / "interactive",
    show_default=True,
    help="Directory of notebooks to execute.",
)
@_NOTEBOOK_OPTION
@click.option(
    "--kernel",
    "kernel_name",
    default="python3",
    show_default=True,
    help="Jupyter kernel to execute the notebooks with.",
)
def execute_notebooks_command(
    output_dir: Path, notebook_dir: Path, notebooks: Tuple[str, ...], kernel_name: str
) -> None:
    """Execute notebooks into OUTPUT_DIR, recording their plotted data.

    Prints which notebooks failed, if any. Exits 0 when every notebook ran,
    1 when any failed (all are still written), and 2 when the notebook
    directory is missing, has no notebooks, has no notebook that matches a
    --notebook pattern, or is OUTPUT_DIR itself.
    """
    ctx = click.get_current_context()
    try:
        outcomes = execute_notebooks(
            notebook_dir, output_dir, kernel_name, notebooks=notebooks
        )
    except Exception as e:
        click.echo(_error_message(e), err=True)
        ctx.exit(2)
    failures = {name: error for name, error in outcomes.items() if error is not None}
    click.echo(
        f"Executed {len(outcomes)} notebook(s) into {output_dir}; {len(failures)} failed."
    )
    for name, error in failures.items():
        click.echo(f"  {name}: {error}")
    ctx.exit(1 if failures else 0)


_Comparison = Tuple[str, Callable[[], ComparisonResult]]
"""A label naming a comparison's inputs, and a call that runs the comparison."""


def _results_comparison(
    reference: Path,
    candidate: Path,
    measures: Tuple[str, ...],
    rtol: float,
    atol: float,
    workers: int,
) -> _Comparison:
    return (
        f"results {reference} vs {candidate}",
        partial(
            compare_results,
            reference,
            candidate,
            rtol,
            atol,
            measures=measures,
            workers=workers,
        ),
    )


def _artifacts_comparison(
    reference: Path, candidate: Path, keys: Tuple[str, ...], rtol: float, atol: float
) -> _Comparison:
    return (
        f"artifacts {reference} vs {candidate}",
        partial(compare_artifacts, reference, candidate, rtol, atol, keys=keys),
    )


def _notebooks_comparison(
    reference: Path,
    candidate: Path,
    notebooks: Tuple[str, ...],
    rtol: float,
    atol: float,
    diff_dir: Optional[Path],
) -> _Comparison:
    return (
        f"notebooks {reference} vs {candidate}",
        partial(
            compare_notebooks,
            reference,
            candidate,
            rtol,
            atol,
            notebooks=notebooks,
            diff_dir=diff_dir,
        ),
    )


def _diff_dir(report_path: Optional[Path], k: int = 1, n: int = 1) -> Optional[Path]:
    """Where the k-th of n notebook comparisons writes its diff pages, if anywhere."""
    if report_path is None:
        return None
    diff_dir = report_path.parent / f"{report_path.stem}_diffs"
    return diff_dir / str(k) if n > 1 else diff_dir


def _run_comparisons(comparisons: list[_Comparison], report_path: Optional[Path]) -> None:
    """Run comparisons, print and save their report, and exit with the status they imply.

    With a single comparison, a failure prints only the one-line error. With
    several, each failure is labelled with its inputs and listed in the
    report, and a summary line leads the report.
    """
    ctx = click.get_current_context()
    results, errors = [], []
    for label, compare in comparisons:
        try:
            results.append(compare())
        except Exception as e:
            message = _error_message(e)
            errors.append(message if len(comparisons) == 1 else f"{label}: {message}")
    try:
        report_dir = report_path.parent if report_path is not None else None
        report = render_report(results, report_dir) if results else ""
    except Exception as e:
        errors.append(_error_message(e))
        report = ""
    if len(comparisons) > 1:
        n_moved = sum(result.moved for result in results)
        summary = (
            f"**Summary:** {len(comparisons)} comparisons, "
            f"{len(results) - n_moved} clean, {n_moved} moved, {len(errors)} failed."
        )
        report = f"{summary}\n\n{report}"
        if errors:
            report += "\n## Errors\n\n" + "".join(f"- {error}\n" for error in errors)

    if report:
        click.echo(report, nl=False)
    for error in errors:
        click.echo(error, err=True)
    if report_path is not None:
        if not report:
            # One comparison that failed: name its inputs, not just the error.
            report = "".join(f"{error}\n" for error in errors)
            report += f"\nComparison: {comparisons[0][0]}\n"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
    # The report links diff pages relative to itself; say where they are on disk.
    for result in results:
        for item, page in result.diff_files.items():
            click.echo(f"Diff page for {item}: {page.resolve()}", err=True)
    if errors:
        ctx.exit(2)
    ctx.exit(1 if any(result.moved for result in results) else 0)


def _error_message(error: Exception) -> str:
    """A one-line ``Error: ...`` message for an exception, without a traceback.

    Exit code 1 means "moved", so any failure to compare must exit 2 instead;
    exceptions other than ComparisonInputError are named by type.
    """
    if isinstance(error, ComparisonInputError):
        message = f"Error: {error}"
    else:
        message = f"Error: {type(error).__name__}: {error}"
    return " ".join(message.split())
