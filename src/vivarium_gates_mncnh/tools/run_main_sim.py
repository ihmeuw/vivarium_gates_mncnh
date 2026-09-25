"""
Run the main simulation for all locations.

Usage
-----
Run from anywhere; the script locates the repository itself:

    python -m vivarium_gates_mncnh.tools.run_main_sim \\
        --queue all.q --project proj_simscience_prod --model-number 29.0.2

psimulate runs in the environment this process is already in. Under
``source environment.sh -s`` that is a venv overlay whose editable install is
the code under test, so nothing is dispatched elsewhere.

Options
-------
``--baseline-only``
    Run only the baseline scenario.
``--locations``
    Run a subset of locations (repeatable). Defaults to all of them.
``--artifact-dir``
    Directory holding the per-location ``.hdf`` artifacts, overriding the model
    spec's production path.
``--output-root``
    Write results under this root instead of the team results mount.
``--skip-tag``
    Do not create or push the ``v{model_number}`` git tag. Requires
    ``--output-root`` set to something other than the production results root,
    so an untagged run cannot land beside real model results.
``--skip-tree-check``
    Skip the clean-tree guard.
"""

import argparse
import re
import subprocess
from pathlib import Path
from typing import List, Optional

import yaml

from vivarium_gates_mncnh.constants.metadata import LOCATIONS
from vivarium_gates_mncnh.tools.utilities import (
    check_clean_tree,
    check_psimulate_finished,
    create_and_push_tag,
    extract_results_dir,
    require_environment,
    run_command,
)

RESULTS_ROOT = Path("/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/results")
MODEL_SPEC_DIR = Path(__file__).resolve().parent.parent / "model_specifications"
MODEL_SPEC_PATH = MODEL_SPEC_DIR / "model_spec.yaml"
PATHS_MODULE = Path(__file__).resolve().parent.parent / "constants" / "paths.py"


def _update_model_results_dir(model_number: str) -> None:
    """Update ``MODEL_RESULTS_DIR`` in ``constants/paths.py`` to match *model_number*.

    Parameters
    ----------
    model_number
        The model version number (e.g. "29.0.2").
    """
    new_value = f"model{model_number}"
    content = PATHS_MODULE.read_text()

    pattern = re.compile(r'^(MODEL_RESULTS_DIR\s*=\s*)"[^"]*"', re.MULTILINE)
    if not pattern.search(content):
        raise RuntimeError(f"Could not find MODEL_RESULTS_DIR assignment in {PATHS_MODULE}")

    new_content = pattern.sub(rf'\1"{new_value}"', content)
    if new_content == content:
        print(f'MODEL_RESULTS_DIR already set to "{new_value}". No update needed.')
        return

    PATHS_MODULE.write_text(new_content)
    print(f'Updated MODEL_RESULTS_DIR to "{new_value}" in {PATHS_MODULE.name}')


def run_sim(
    queue: str,
    project: str,
    model_number: str,
    baseline_only: bool = False,
    locations: Optional[List[str]] = None,
    artifact_dir: Optional[Path] = None,
    output_root: Optional[Path] = None,
    tag: bool = True,
    check_tree: bool = True,
) -> None:
    """Run the main simulation for the requested locations.

    Verifies that the tree is clean and that ``psimulate`` is actually
    available, optionally creates and pushes a git tag ``v{model_number}``,
    updates the model results directory, then launches psimulate per location.

    psimulate runs in whichever environment this process is already in. The
    shared-environment setup activates a venv overlay whose editable install is
    the code under test, so dispatching elsewhere would run something other
    than the revision being checked.

    Parameters
    ----------
    queue
        The cluster queue to submit simulation jobs to.
    project
        The cluster project to submit simulation jobs to.
    model_number
        The model version number (e.g. "29.0.2").
    baseline_only
        If True, run only the baseline scenario.
    locations
        Locations to run. Defaults to every location in
        :data:`~vivarium_gates_mncnh.constants.metadata.LOCATIONS`. A framework
        check wants one location, not all of them.
    artifact_dir
        Directory holding the per-location ``.hdf`` artifacts. Defaults to the
        parent of the ``artifact_path`` in the model spec, which is pinned to a
        production directory -- a check run must be able to point elsewhere.
    output_root
        Root to write results under, in place of :data:`RESULTS_ROOT`. Results
        land in ``<output_root>/model{model_number}``.
    tag
        If False, skip creating and pushing the ``v{model_number}`` git tag
        entirely. A Jenkins checkout has no push credentials, and a framework
        check should not be tagging the repository.
    check_tree
        If False, skip the clean-tree guard.

    Raises
    ------
    RuntimeError
        If a precondition fails, or if psimulate does not complete every job
        for some location.
    """
    requested_locations = list(locations) if locations else list(LOCATIONS)
    unknown = [loc for loc in requested_locations if loc not in LOCATIONS]
    if unknown:
        raise RuntimeError(
            f"Unknown location(s): {', '.join(unknown)}. Expected some of {list(LOCATIONS)}."
        )

    results_root = output_root if output_root is not None else RESULTS_ROOT
    output_path = (results_root / f"model{model_number}").resolve()

    with open(MODEL_SPEC_PATH, "r") as f:
        model_spec = yaml.safe_load(f)
        default_artifact_path = Path(
            model_spec["configuration"]["input_data"]["artifact_path"]
        )

    if artifact_dir is None:
        if default_artifact_path.stem.lower() not in [loc.lower() for loc in LOCATIONS]:
            raise RuntimeError(
                f"Default artifact path {default_artifact_path} does not match any "
                "known location. Pass --artifact-dir to point at the artifacts to use."
            )
        artifact_directory = default_artifact_path.parent
    else:
        artifact_directory = Path(artifact_dir)

    artifact_paths = {
        location: artifact_directory / f"{location.lower()}.hdf"
        for location in requested_locations
    }
    # Every requested location up front, before anything is launched: a missing
    # artifact means every job for that location fails on the far side of a
    # cluster submission, which is hours rather than seconds.
    missing = [str(path) for path in artifact_paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(
            "Expected artifact(s) do not exist:\n"
            + "".join(f"  {path}\n" for path in missing)
            + "Build them with 'make_artifacts', or pass --artifact-dir to point "
            "at a directory that already holds them."
        )

    branches_file = (
        MODEL_SPEC_DIR / "branches" / "baseline_only.yaml"
        if baseline_only
        else MODEL_SPEC_DIR / "branches" / "scenarios.yaml"
    )

    print("\n" + "=" * 80)
    print("Main Simulation Workflow")
    print("=" * 80)
    print(f"Model number: {model_number}")
    print(f"Locations: {', '.join(requested_locations)}")
    print(f"Queue: {queue}")
    print(f"Project: {project}")
    print(f"Artifacts: {artifact_directory}")
    print(f"Output: {output_path}")
    print(f"Baseline only: {baseline_only}")
    print(f"Tagging: {tag}")
    print(f"Clean-tree check: {check_tree}")
    print("=" * 80)

    # Before anything slow or irreversible: confirm this really is a simulation
    # environment. The venv overlay only patches PATH on a sourced activation,
    # so a runner that launches this by absolute interpreter path leaves the
    # shared environment's bin/ off PATH and psimulate unresolvable. That is a
    # misconfigured step, and this reports it as one rather than quietly
    # reconstructing a PATH to work around it.
    require_environment("simulation")

    if check_tree:
        check_clean_tree()
    if tag:
        create_and_push_tag(model_number)
    _update_model_results_dir(model_number)

    for location in requested_locations:
        print(f"\n{'='*80}")
        print(f"Running simulation for {location}...")
        print(f"{'='*80}")

        psimulate_output = run_command(
            [
                "psimulate",
                "run",
                "-vvv",
                "-P",
                project,
                "-q",
                queue,
                "-i",
                str(artifact_paths[location]),
                "-m",
                "1",
                "-r",
                "00:15:00",
                "-o",
                str(output_path),
                str(MODEL_SPEC_PATH),
                str(branches_file),
            ],
            f"psimulate run for {location}",
            capture_full_output=True,
        )

        if not check_psimulate_finished(psimulate_output):
            raise RuntimeError(
                f"Simulation for {location}: Not all jobs finished successfully"
            )

        results_dir = extract_results_dir(psimulate_output)
        if results_dir:
            print(f"Simulation for {location} completed successfully.")
            print(f"Results directory: {results_dir}")
        else:
            print(
                f"WARNING: Simulation for {location} completed but could not "
                "determine results directory."
            )

    print("\n" + "=" * 80)
    print("All simulations completed.")
    print(f"Results are located in: {output_path}")
    print("=" * 80)


def _validate_cli_args(args: argparse.Namespace) -> None:
    """Reject flag combinations that are individually fine but jointly unsafe.

    Currently one rule: ``--skip-tag`` requires ``--output-root`` set to
    something other than :data:`RESULTS_ROOT`. The V&V process relies on a
    run's results directory corresponding to a git tag, so an untagged run must
    not be able to write into the production results tree. Skipping the tag is
    legitimate for a check run; skipping it *and* writing to production is not.

    Raises
    ------
    SystemExit
        Via ``parser.error`` semantics -- the caller reports the message and
        exits non-zero rather than raising a traceback at a user.
    """
    if not args.skip_tag:
        return

    output_root = getattr(args, "output_root", None)
    if output_root is None or Path(output_root).resolve() == RESULTS_ROOT.resolve():
        raise SystemExit(
            "argument --skip-tag: requires --output-root set to something other "
            f"than the production results root ({RESULTS_ROOT}). The V&V process "
            "relies on a results directory corresponding to a git tag, so an "
            "untagged run must not write into the production results tree."
        )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Factored out of ``__main__`` so the flag surface -- and the rules in
    :func:`_validate_cli_args` -- can be exercised without running a
    simulation.
    """
    parser = argparse.ArgumentParser(
        description="Run the main simulation for the requested locations."
    )
    parser.add_argument(
        "-q",
        "--queue",
        type=str,
        required=True,
        help="The queue to submit the simulation jobs to.",
    )
    parser.add_argument(
        "-P",
        "--project",
        type=str,
        required=True,
        help="The project to submit the simulation jobs to.",
    )
    parser.add_argument(
        "-m",
        "--model-number",
        type=str,
        required=True,
        help=(
            "Model version number (e.g. '29.0.2'). "
            f"Results are written to {RESULTS_ROOT}/model{{model_number}} "
            "and a git tag v{model_number} is created."
        ),
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Whether to run only the baseline scenario.",
    )
    parser.add_argument(
        "--locations",
        action="append",
        choices=list(LOCATIONS),
        metavar="LOCATION",
        help=(
            "A location to run. May be given more than once. "
            f"Defaults to all of them: {', '.join(LOCATIONS)}."
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help=(
            "Directory holding the per-location '.hdf' artifacts, overriding the "
            "production directory pinned in the model spec."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            f"Write results under this root instead of {RESULTS_ROOT}. "
            "Results land in <output-root>/model{model_number}."
        ),
    )
    parser.add_argument(
        "--skip-tag",
        action="store_true",
        help=(
            "Do not create or push the v{model_number} git tag. Requires "
            "--output-root set to something other than the production results root."
        ),
    )
    parser.add_argument(
        "--skip-tree-check",
        action="store_true",
        help="Skip the clean-tree guard.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    _validate_cli_args(args)

    run_sim(
        queue=args.queue,
        project=args.project,
        model_number=args.model_number,
        baseline_only=args.baseline_only,
        locations=args.locations,
        artifact_dir=args.artifact_dir,
        output_root=args.output_root,
        tag=not args.skip_tag,
        check_tree=not args.skip_tree_check,
    )
