"""
Run the main simulation for all locations.

Usage
-----
Run from the repository root:

    python -m vivarium_gates_mncnh.tools.run_main_sim \\
        --queue all.q --project proj_simscience_prod --model_number 29.0.2

Add ``--baseline_only`` to run only the baseline scenario.
"""
import argparse
import re
import subprocess
from pathlib import Path

import yaml

from vivarium_gates_mncnh.constants.metadata import LOCATIONS
from vivarium_gates_mncnh.tools.utilities import (
    check_clean_tree,
    check_conda_environments,
    check_psimulate_finished,
    create_and_push_tag,
    extract_results_dir,
    run_command,
)

RESULTS_ROOT = Path("/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/results")
MODEL_SPEC_DIR = Path(__file__).resolve().parent.parent / "model_specifications"
MODEL_SPEC_PATH = MODEL_SPEC_DIR / "model_spec.yaml"
PATHS_MODULE = Path(__file__).resolve().parent.parent / "constants" / "paths.py"
CONDA_ENV = "vivarium_gates_mncnh_simulation"


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
) -> None:
    """Run the main simulation for all locations.

    Checks that the tree is clean, creates a git tag ``v{model_number}``, pushes it to origin, updates the model results directory, then launches
    psimulate for every location.

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
    """
    output_path = (RESULTS_ROOT / f"model{model_number}").resolve()

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
    print("=" * 80)

    check_clean_tree()
    create_and_push_tag(model_number)
    _update_model_results_dir(model_number)
    check_conda_environments()

    for location in LOCATIONS:
        print(f"\n{'='*80}")
        print(f"Running simulation for {location}...")
        print(f"{'='*80}")

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
                "15:00",
                "-o",
                str(output_path),
                str(MODEL_SPEC_PATH),
                str(branches_file),
            ],
            f"psimulate run for {location}",
            conda_env=CONDA_ENV,
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
    args = parser.parse_args()

    run_sim(
        queue=args.queue,
        project=args.project,
        model_number=args.model_number,
        baseline_only=args.baseline_only,
    )
