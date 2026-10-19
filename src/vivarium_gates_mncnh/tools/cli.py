from pathlib import Path
from typing import Optional, Tuple

import click
from loguru import logger
from vivarium.engine.framework.utilities import handle_exceptions

from vivarium_gates_mncnh.constants import metadata, paths
from vivarium_gates_mncnh.constants.paths import MODEL_NOTEBOOKS_DIR
from vivarium_gates_mncnh.tools import build_artifacts, configure_logging_to_terminal


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


_TOLERANCE_OPTIONS = [
    click.option(
        "--rtol",
        type=float,
        default=0.0,
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
]


def _tolerance_options(command):
    for option in reversed(_TOLERANCE_OPTIONS):
        command = option(command)
    return command


@click.group()
def compare_outputs() -> None:
    """Compare two runs' outputs value by value.

    Each comparison prints a Markdown report to stdout and exits 0 when the
    two sides compare clean, 1 when anything moved, and 2 when the inputs
    cannot be compared (for example, a path that does not exist).
    """


@compare_outputs.command("results")
@click.argument("reference_dir", type=click.Path(path_type=Path))
@click.argument("candidate_dir", type=click.Path(path_type=Path))
@_tolerance_options
def compare_results_command(
    reference_dir: Path, candidate_dir: Path, rtol: float, atol: float
) -> None:
    """Compare two results directories (one location's results/ each)."""
    raise NotImplementedError


@compare_outputs.command("artifacts")
@click.argument("reference_path", type=click.Path(path_type=Path))
@click.argument("candidate_path", type=click.Path(path_type=Path))
@_tolerance_options
def compare_artifacts_command(
    reference_path: Path, candidate_path: Path, rtol: float, atol: float
) -> None:
    """Compare two artifacts key by key."""
    raise NotImplementedError


@compare_outputs.command("notebooks")
@click.argument("reference_dir", type=click.Path(path_type=Path))
@click.argument("candidate_dir", type=click.Path(path_type=Path))
@_tolerance_options
def compare_notebooks_command(
    reference_dir: Path, candidate_dir: Path, rtol: float, atol: float
) -> None:
    """Compare two directories of executed notebooks."""
    raise NotImplementedError


@compare_outputs.command("execute-notebooks")
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--notebook-dir",
    type=click.Path(path_type=Path),
    default=MODEL_NOTEBOOKS_DIR / "interactive",
    show_default=True,
    help="Directory of notebooks to execute.",
)
@click.option(
    "--kernel",
    "kernel_name",
    default="python3",
    show_default=True,
    help="Jupyter kernel to execute the notebooks with.",
)
def execute_notebooks_command(output_dir: Path, notebook_dir: Path, kernel_name: str) -> None:
    """Execute notebooks into OUTPUT_DIR, recording their plotted data.

    Prints which notebooks failed, if any. Exits 0 when every notebook ran,
    1 when any failed (all are still written), and 2 when the notebook
    directory has no notebooks.
    """
    raise NotImplementedError
