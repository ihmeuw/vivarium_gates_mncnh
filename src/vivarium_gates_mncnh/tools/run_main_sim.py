"""
Run the main simulation for all locations.

Usage
-----
Run from the repository root:

    python -m vivarium_gates_mncnh.tools.run_main_sim \\
        --queue all.q --project proj_simscience_prod --model_number 29.0.2

psimulate is run in the simulation environment -- by default the active one when
it is already a simulation environment, otherwise its sibling
``.venv/<name>_simulation`` overlay. ``--env`` names one explicitly.

The simulation and artifact environments have incompatible dependencies and
cannot be merged. A script that builds an artifact and then launches models is
therefore always running in one of them while needing the other, which is why
every command here takes an explicit environment target rather than inheriting
the caller's.

Other options:

``--baseline-only``
    Run only the baseline scenario.
``--no-tag`` / ``--no-push``
    Skip the ``v{model_number}`` git tag, or create it locally without pushing.
``--parallel``
    Run all locations concurrently instead of one at a time.
"""
import argparse
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from vivarium_gates_mncnh.constants.metadata import LOCATIONS
from vivarium_gates_mncnh.tools.utilities import (
    check_clean_tree,
    check_environment,
    check_psimulate_finished,
    create_and_push_tag,
    default_env_for_type,
    extract_results_dir,
    run_command,
)

RESULTS_ROOT = Path("/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/results")
MODEL_SPEC_DIR = Path(__file__).resolve().parent.parent / "model_specifications"
MODEL_SPEC_PATH = MODEL_SPEC_DIR / "model_spec.yaml"
PATHS_MODULE = Path(__file__).resolve().parent.parent / "constants" / "paths.py"


def _update_model_results_dir(model_number: str) -> bool:
    """Update ``MODEL_RESULTS_DIR`` in ``constants/paths.py`` to match *model_number*.

    Parameters
    ----------
    model_number
        The model version number (e.g. "29.0.2").

    Returns
    -------
    bool
        True if the file was rewritten, i.e. the working tree is now dirty.
    """
    new_value = f"model{model_number}"
    content = PATHS_MODULE.read_text()

    pattern = re.compile(r'^(MODEL_RESULTS_DIR\s*=\s*)"[^"]*"', re.MULTILINE)
    if not pattern.search(content):
        raise RuntimeError(f"Could not find MODEL_RESULTS_DIR assignment in {PATHS_MODULE}")

    new_content = pattern.sub(rf'\1"{new_value}"', content)
    if new_content == content:
        print(f'MODEL_RESULTS_DIR already set to "{new_value}". No update needed.')
        return False

    PATHS_MODULE.write_text(new_content)
    print(f'Updated MODEL_RESULTS_DIR to "{new_value}" in {PATHS_MODULE.name}')
    print("\n" + "!" * 80)
    print(
        f"WARNING: this script has left an uncommitted change in {PATHS_MODULE.name}.\n"
        "         The results V&V notebooks read MODEL_RESULTS_DIR, so the change is\n"
        "         needed -- but it is yours to commit. The tag created for this run\n"
        "         does NOT include it."
    )
    print("!" * 80)
    return True


def run_sim(
    queue: str,
    project: str,
    model_number: str,
    baseline_only: bool = False,
    output_dir: str | None = None,
    branches: str | None = None,
    env: str | None = None,
    tag: bool = True,
    push: bool = True,
    parallel: bool = False,
    allow_env_mismatch: bool = False,
) -> None:
    """Run the main simulation for all locations.

    Checks that the tree is clean and that the environment is usable, creates a git
    tag ``v{model_number}`` (pushing it to origin unless told not to), updates the
    model results directory, then launches psimulate for every location.

    Parameters
    ----------
    queue
        The cluster queue to submit simulation jobs to.
    project
        The cluster project to submit simulation jobs to.
    model_number
        The model version number (e.g. "29.0.2").  Results are written to
        ``/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/results/model{model_number}``.
    baseline_only
        If True, run only the baseline scenario.
    output_dir
        Directory to write results under, in place of the team results mount. For
        scratch and smoke-test runs that should not land beside real model results.
    branches
        Path to a branches file, overriding the standard scenarios/baseline choice.
        For scratch runs over a reduced set of scenarios, draws or seeds.
    env
        Environment to run psimulate in: a conda env name, a venv name, or a path
        to either prefix. If None (the default), the *simulation* environment is
        derived from the active one -- which is not the same as "run here". The
        simulation and artifact environments have incompatible dependencies and
        cannot be merged, so a script that builds an artifact and then launches
        models is necessarily running in one of them while needing the other. When
        this is called from the artifact environment, the default resolves to the
        sibling simulation overlay rather than launching psimulate from an
        environment that cannot support it.
    tag
        If False, skip creating the ``v{model_number}`` git tag entirely.
    push
        If False, create the tag locally but do not push it to origin.  Pushing a
        tag on an unpushed branch publishes every commit on that branch.
    allow_env_mismatch
        If True, downgrade the environment revision check to a warning. Needed when
        the environment is deliberately not built from HEAD.
    parallel
        If True, run all locations concurrently instead of one after another.
        Each location is a separate Jobmon workflow writing to its own output
        subdirectory, so their output is interleaved and prefixed by location.
    """
    # Resolve before printing, so the banner names the environment that will
    # actually run rather than the one this process happens to be in.
    if env is None:
        env = default_env_for_type("simulation")

    results_root = Path(output_dir).expanduser() if output_dir else RESULTS_ROOT
    output_path = (results_root / f"model{model_number}").resolve()

    with open(MODEL_SPEC_PATH, "r") as f:
        model_spec = yaml.safe_load(f)
        default_artifact_path = Path(
            model_spec["configuration"]["input_data"]["artifact_path"]
        )

    assert default_artifact_path.stem.lower() in [
        loc.lower() for loc in LOCATIONS
    ], f"Default artifact path {default_artifact_path} does not match any known location."
    assert set(loc.lower() for loc in LOCATIONS) <= set(
        p.stem.lower() for p in default_artifact_path.parent.iterdir()
    ), f"Default artifact path {default_artifact_path.parent} does not contain all known locations."

    if branches is not None:
        branches_file = Path(branches).expanduser().resolve()
        if not branches_file.exists():
            raise RuntimeError(f"Branches file {branches_file} does not exist.")
    else:
        branches_file = (
            MODEL_SPEC_DIR / "branches" / "baseline_only.yaml"
            if baseline_only
            else MODEL_SPEC_DIR / "branches" / "scenarios.yaml"
        )

    print("\n" + "=" * 80)
    print("Main Simulation Workflow")
    print("=" * 80)
    print(f"Model number: {model_number}")
    print(f"Locations: {', '.join(LOCATIONS)}")
    print(f"Queue: {queue}")
    print(f"Project: {project}")
    print(f"Output: {output_path}")
    print(f"Baseline only: {baseline_only}")
    print(f"Branches: {branches_file}")
    print(f"Environment: {env if env else 'active environment'}")
    print(f"Tag: {f'v{model_number}' if tag else 'none'}{'' if push else ' (local only)'}")
    print(f"Parallel: {parallel}")
    print("=" * 80)

    check_clean_tree()
    check_environment(env, allow_env_mismatch=allow_env_mismatch)
    if tag:
        create_and_push_tag(model_number, push=push)
    else:
        print(f"\nSkipping git tag 'v{model_number}' (--no-tag)")
    _update_model_results_dir(model_number)

    def run_location(location: str, log_prefix: str = "") -> None:
        print(f"\n{log_prefix}{'='*80}")
        print(f"{log_prefix}Running simulation for {location}...")
        print(f"{log_prefix}{'='*80}")

        artifact_path = default_artifact_path.parent / f"{location.lower()}.hdf"
        assert (
            artifact_path.exists()
        ), f"Expected artifact path {artifact_path} does not exist."

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
                str(artifact_path),
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
            env=env,
            capture_full_output=True,
            log_prefix=log_prefix,
        )

        if not check_psimulate_finished(psimulate_output):
            raise RuntimeError(
                f"Simulation for {location}: Not all jobs finished successfully"
            )

        results_dir = extract_results_dir(psimulate_output)
        if results_dir:
            print(f"{log_prefix}Simulation for {location} completed successfully.")
            print(f"{log_prefix}Results directory: {results_dir}")
        else:
            print(
                f"{log_prefix}WARNING: Simulation for {location} completed but could "
                "not determine results directory."
            )

    if parallel:
        # Each location is an independent Jobmon workflow writing to its own
        # output subdirectory, so there is no contention between them.
        with ThreadPoolExecutor(max_workers=len(LOCATIONS)) as executor:
            futures = {
                executor.submit(run_location, location, f"[{location}] "): location
                for location in LOCATIONS
            }
            failures = {}
            for future, location in futures.items():
                try:
                    future.result()
                except Exception as e:  # noqa: BLE001 - report every failure
                    failures[location] = e
        if failures:
            raise RuntimeError(
                "Simulations failed for: "
                + "; ".join(f"{loc} ({e})" for loc, e in failures.items())
            )
    else:
        for location in LOCATIONS:
            run_location(location)

    print("\n" + "=" * 80)
    print("All simulations completed.")
    print(f"Results are located in: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main simulation for all locations.")
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
            "Results are written to /mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/results/model{model_number} "
            "and a git tag v{model_number} is created."
        ),
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Whether to run only the baseline scenario.",
    )
    parser.add_argument(
        "--no-tag",
        action="store_true",
        help="Do not create the v{model_number} git tag for this run.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help=(
            "Create the git tag locally but do not push it to origin. Pushing a tag "
            "on an unpushed branch publishes every commit on that branch."
        ),
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "Run all locations concurrently rather than one at a time. Each location "
            "is a separate Jobmon workflow with its own output subdirectory; their "
            "output is interleaved and prefixed by location."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to write results under, in place of the team results mount. "
            "For scratch and smoke-test runs."
        ),
    )
    parser.add_argument(
        "--branches",
        type=str,
        default=None,
        help=(
            "Path to a branches file, overriding the standard scenarios/baseline "
            "choice. For scratch runs over a reduced set of scenarios, draws or seeds."
        ),
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help=(
            "Environment to run psimulate in: a conda env name, a venv name, or a "
            "path to either one's directory. Defaults to the simulation "
            "environment derived from the active one -- the sibling "
            "'.venv/<name>_simulation' overlay when run from the artifact "
            "environment, which cannot run psimulate itself."
        ),
    )
    parser.add_argument(
        "--allow-env-mismatch",
        action="store_true",
        help=(
            "Warn instead of failing when the environment's installed revision does "
            "not match HEAD. For deliberate mismatches only -- the check exists to "
            "stop the cluster silently running a different checkout's code."
        ),
    )
    args = parser.parse_args()

    run_sim(
        queue=args.queue,
        project=args.project,
        model_number=args.model_number,
        baseline_only=args.baseline_only,
        output_dir=args.output_dir,
        branches=args.branches,
        env=args.env,
        tag=not args.no_tag,
        push=not args.no_push,
        parallel=args.parallel,
        allow_env_mismatch=args.allow_env_mismatch,
    )
