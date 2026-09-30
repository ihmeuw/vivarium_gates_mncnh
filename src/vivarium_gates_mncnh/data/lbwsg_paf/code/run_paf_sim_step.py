#!/usr/bin/env python
"""
Run one step of the LBWSG PAF workflow.

Usage
-----
Run the five steps in order, each in the environment named beside it:

    python run_paf_sim_step.py --step initial-artifact -a NAME -l Ethiopia   # artifact
    python run_paf_sim_step.py --step enn-paf          -a NAME -l Ethiopia   # simulation
    python run_paf_sim_step.py --step enn-artifact     -a NAME -l Ethiopia   # artifact
    python run_paf_sim_step.py --step lnn-paf          -a NAME -l Ethiopia   # simulation
    python run_paf_sim_step.py --step final-artifact   -a NAME -l Ethiopia   # artifact
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from vivarium_gates_mncnh.constants.metadata import LOCATIONS, PAF_PHASE_ENVIRONMENTS
from vivarium_gates_mncnh.constants.paths import CLUSTER_DATA_DIR
from vivarium_gates_mncnh.tools.utilities import (
    check_psimulate_finished,
    extract_results_dir,
    move_results,
    require_environment,
    run_command,
)

PAF_MEASURES = [
    "risk_factor.low_birth_weight_and_short_gestation.population_attributable_fraction",
    "cause.neonatal_preterm_birth.population_attributable_fraction",
]

TEAM_ARTIFACT_ROOT = Path(
    "/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/artifacts"
)


def _reuse_existing(artifact_file: Path) -> bool:
    """Ask whether to reuse an artifact that already exists."""
    print(f"\nArtifact already exists: {artifact_file}")
    try:
        response = input("Use existing artifact? [Y/n]: ").strip().lower()
    except EOFError:
        print("No interactive input available; defaulting to use existing artifact.")
        return True
    if response in ("", "y", "yes"):
        return True
    print("Will overwrite existing artifact.")
    return False


def _artifact_dir(artifact_name: Optional[str], output_dir: Optional[str]) -> Path:
    """Resolve the directory holding ``<location>.hdf``, creating it if needed.

    Exactly one of *artifact_name* and *output_dir* is set; argparse enforces it.
    """
    path = (
        Path(output_dir).expanduser()
        if output_dir
        else TEAM_ARTIFACT_ROOT / str(artifact_name)
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_artifact(
    location: str,
    artifact_dir: Path,
    description: str,
    measures: Optional[List[str]] = None,
) -> None:
    """Run ``make_artifacts`` for *location*, optionally restricted to *measures*."""
    cmd = ["make_artifacts", "-vvv", "-l", location.capitalize(), "-o", str(artifact_dir)]
    for measure in measures or []:
        cmd += ["-r", measure]
    run_command(cmd, description, auto_confirm=True)


def _run_paf_simulation(
    location: str,
    artifact_dir: Path,
    model_spec: str,
    description: str,
    results_to_move: List[Tuple[str, str, str]],
) -> None:
    """Run psimulate, then move its outputs into the tracked ``outputs/`` tree.

    *results_to_move* is a list of ``(source glob, destination subdirectory,
    description)`` applied to the results directory psimulate reports. The move
    happens here, in the same step, so no later step has to be told where this
    run's scratch directory was. The scratch directory is removed on success
    and preserved on failure, for debugging.
    """
    script_dir = Path(__file__).parent.parent
    working_dir = (
        Path(CLUSTER_DATA_DIR) / "paf_sim_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    working_dir.mkdir(parents=True, exist_ok=True)

    try:
        psimulate_output = run_command(
            [
                "psimulate",
                "run",
                "-vvv",
                "-P",
                "proj_simscience_prod",
                "-i",
                str(artifact_dir / f"{location}.hdf"),
                "-o",
                str(working_dir),
                str(script_dir / "code" / model_spec),
                str(script_dir / "code" / "lbwsg_paf_branches.yaml"),
            ],
            description,
            capture_full_output=True,
        )

        if not check_psimulate_finished(psimulate_output):
            raise RuntimeError(f"{description}: not all jobs finished successfully")

        results_dir = extract_results_dir(psimulate_output)
        if results_dir is None:
            raise RuntimeError(
                f"{description}: psimulate reported success but its results "
                "directory could not be determined, so the outputs cannot be "
                "collected."
            )

        for pattern, destination, what in results_to_move:
            move_results(
                f"{results_dir}/{pattern}",
                f"{script_dir}/outputs/{destination}/{location}",
                what,
            )
    except Exception:
        print(
            f"\nWorking directory preserved for debugging: {working_dir}",
            file=sys.stderr,
        )
        raise

    shutil.rmtree(working_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one step of the LBWSG PAF workflow.")
    parser.add_argument(
        "--step",
        required=True,
        # Ordered as constants.metadata.PAF_PHASES is, so --help lists the
        # phases in the order they run rather than alphabetically.
        choices=list(PAF_PHASE_ENVIRONMENTS),
        help="Which step to run. It runs in the environment the caller activated.",
    )
    parser.add_argument(
        "-l",
        "--location",
        action="append",
        dest="locations",
        default=None,
        metavar="LOCATION",
        help=(
            "Location to run, repeatable. Defaults to every location in "
            "constants.metadata.LOCATIONS, which is where a new location should "
            "be added -- the workflow does not name them."
        ),
    )
    # Two ways of naming one directory, so exactly one of them is required.
    # -o used to be an override that silently ignored a still-required -a.
    artifact_dir_group = parser.add_mutually_exclusive_group(required=True)
    artifact_dir_group.add_argument(
        "-a",
        "--artifact_name",
        type=str,
        dest="artifact_name",
        default=None,
        help=(
            f"Name of the artifact directory, under {TEAM_ARTIFACT_ROOT}. "
            "Use --output-dir instead to write somewhere else entirely."
        ),
    )
    artifact_dir_group.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        help=(
            "Full path to the artifact directory where {location}.hdf is written "
            "(supports ~) -- e.g. a personal scratch dir for isolated test runs. "
            "Use --artifact_name instead to write under the team mount."
        ),
    )
    args = parser.parse_args()

    step = args.step
    requested = list(args.locations) if args.locations else list(LOCATIONS)
    unknown = [loc for loc in requested if loc not in LOCATIONS]
    if unknown:
        parser.error(
            f"Unknown location(s): {', '.join(unknown)}. Expected some of {list(LOCATIONS)}."
        )
    artifact_dir = _artifact_dir(args.artifact_name, args.output_dir)

    print("\n" + "=" * 80)
    print(f"LBWSG PAF workflow -- step '{step}'")
    print("=" * 80)
    print(f"Locations: {', '.join(requested)}")
    print(f"Artifact: {artifact_dir}")
    print("=" * 80)

    require_environment(PAF_PHASE_ENVIRONMENTS[step])

    for location in (loc.lower() for loc in requested):
        _run_one_location(step, location, artifact_dir)

    print("\n" + "=" * 80)
    print(f"Step '{step}' completed successfully for {len(requested)} location(s).")
    print("=" * 80 + "\n")


def _run_one_location(step: str, location: str, artifact_dir: Path) -> None:
    """Run *step* for a single location."""
    if step == "initial-artifact":
        artifact_file = artifact_dir / f"{location}.hdf"
        if artifact_file.exists() and _reuse_existing(artifact_file):
            print("Using existing artifact.")
        else:
            _build_artifact(
                location, artifact_dir, f"initial artifact generation for {location}"
            )

    elif step == "enn-paf":
        _run_paf_simulation(
            location,
            artifact_dir,
            "lbwsg_paf_enn.yaml",
            "psimulate run (early neonatal PAFs)",
            [("calculated_lbwsg_paf*", "paf_outputs", "early neonatal PAF output files")],
        )

    elif step == "enn-artifact":
        _build_artifact(
            location,
            artifact_dir,
            f"early neonatal artifact generation for {location}",
            PAF_MEASURES,
        )

    elif step == "lnn-paf":
        _run_paf_simulation(
            location,
            artifact_dir,
            "lbwsg_paf.yaml",
            "psimulate run (late neonatal PAFs and preterm prevalence)",
            [
                ("calculated_lbwsg_paf*", "paf_outputs", "late neonatal PAF output files"),
                (
                    "calculated_late_neonatal_preterm*",
                    "preterm_prevalence_outputs",
                    "preterm prevalence output files",
                ),
            ],
        )

    elif step == "final-artifact":
        _build_artifact(
            location,
            artifact_dir,
            f"final artifact generation for {location}",
            PAF_MEASURES,
        )


if __name__ == "__main__":
    main()
