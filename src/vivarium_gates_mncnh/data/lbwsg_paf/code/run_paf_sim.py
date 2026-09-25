#!/usr/bin/env python
"""
Run one step of the LBWSG PAF workflow.

The workflow alternates between two environments whose dependency sets
conflict and cannot be merged: ``make_artifacts`` needs the *artifact*
environment, ``psimulate`` needs the *simulation* environment. Rather than
have this script dispatch commands into whichever one it is not currently in,
each step is a separate invocation that runs wholly inside the environment the
caller activated. The workflow runner selects that environment per step; see
``model_specifications/artifact_workflow.yaml``.

Usage
-----
Run the five steps in order, each in the environment named beside it:

    python run_paf_sim.py --step initial-artifact -a NAME -l Ethiopia   # artifact
    python run_paf_sim.py --step enn-paf          -a NAME -l Ethiopia   # simulation
    python run_paf_sim.py --step enn-artifact     -a NAME -l Ethiopia   # artifact
    python run_paf_sim.py --step lnn-paf          -a NAME -l Ethiopia   # simulation
    python run_paf_sim.py --step final-artifact   -a NAME -l Ethiopia   # artifact

Each step checks up front that the active environment is the right kind and
aborts with an actionable message if not, so a mis-set ``environment`` key in
the workflow fails in seconds rather than partway through a job.

State passes between steps on disk, exactly as it did when this was a single
script: the artifact at ``<artifact dir>/<location>.hdf``, and the PAF parquet
files under ``data/lbwsg_paf/outputs/``. No step needs to be told where an
earlier step's scratch output went -- each PAF step harvests its own results
before it finishes.
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from vivarium_gates_mncnh.constants.metadata import LOCATIONS
from vivarium_gates_mncnh.constants.paths import CLUSTER_DATA_DIR
from vivarium_gates_mncnh.tools.utilities import (
    check_psimulate_finished,
    extract_results_dir,
    move_results,
    require_environment,
    run_command,
    warn_if_dirty,
)

#: The measures regenerated from the PAF outputs by every artifact step after
#: the first. Named once because both later artifact steps request the same set.
PAF_MEASURES = [
    "risk_factor.low_birth_weight_and_short_gestation.population_attributable_fraction",
    "cause.neonatal_preterm_birth.population_attributable_fraction",
]

#: Each step, and the environment flavour it must run in. The workflow's
#: ``environment`` key for a step has to agree with this.
STEP_ENVIRONMENTS = {
    "initial-artifact": "artifact",
    "enn-paf": "simulation",
    "enn-artifact": "artifact",
    "lnn-paf": "simulation",
    "final-artifact": "artifact",
}


def _artifact_dir(artifact_name: str, output_dir: Optional[str]) -> Path:
    """Resolve the directory holding ``<location>.hdf``, creating it if needed."""
    if output_dir:
        path = Path(output_dir).expanduser()
    else:
        path = (
            Path("/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/artifacts")
            / artifact_name
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
    harvests: List[Tuple[str, str, str]],
) -> None:
    """Run psimulate, then move its outputs into the tracked ``outputs/`` tree.

    *harvests* is a list of ``(source glob, destination subdirectory,
    description)`` applied to the results directory psimulate reports. The
    harvest happens here, in the same step, so no later step has to be told
    where this run's scratch directory was. The scratch directory is removed on
    success and preserved on failure, for debugging.
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

        for pattern, destination, what in harvests:
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
        choices=sorted(STEP_ENVIRONMENTS),
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
    parser.add_argument(
        "-a",
        "--artifact_name",
        type=str,
        required=True,
        dest="artifact_name",
        help=(
            "Name of the artifact directory, under the team artifacts mount "
            "unless --output-dir is given."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        help=(
            "Full path to the artifact directory where {location}.hdf is written "
            "(supports ~). Overrides the default team-mount location -- e.g. a "
            "personal scratch dir for isolated test runs."
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

    # Fail in seconds, not partway through a job, if this step was launched in
    # the wrong environment.
    require_environment(STEP_ENVIRONMENTS[step])

    for location in (loc.lower() for loc in requested):
        _run_one(step, location, artifact_dir)

    print("\n" + "=" * 80)
    print(f"Step '{step}' completed successfully for {len(requested)} location(s).")
    print("=" * 80 + "\n")


def _run_one(step: str, location: str, artifact_dir: Path) -> None:
    """Run *step* for a single location.

    Locations are independent of one another -- separate artifacts, separate
    output directories -- so a step simply does each in turn. The workflow runs
    steps strictly sequentially anyway, so there is nothing to gain by
    splitting them into a task per location, and a good deal of YAML to lose.
    """
    if step == "initial-artifact":
        artifact_file = artifact_dir / f"{location}.hdf"
        if artifact_file.exists():
            print(f"\nArtifact already exists and will be reused: {artifact_file}")
            print("Delete it first if you want it rebuilt from scratch.")
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
        warn_if_dirty(["data/lbwsg_paf/outputs"], "The PAF workflow")


if __name__ == "__main__":
    main()
