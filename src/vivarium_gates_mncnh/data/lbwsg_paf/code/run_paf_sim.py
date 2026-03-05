#!/usr/bin/env python
"""
Run PAF simulations and generate artifacts.

Usage
-----
Run from this directory with a target artifact directory name and optional location:

    python run_paf_sim.py -a test_automation -l Pakistan

This would write the artifact to:

    /mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/artifacts/test_automation/pakistan.csv

If ``-l`` is omitted, the location defaults to Ethiopia.
"""
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

from vivarium_gates_mncnh.constants.paths import CLUSTER_DATA_DIR
from vivarium_gates_mncnh.tools.utilities import (
    check_conda_environments,
    check_psimulate_finished,
    extract_results_dir,
    move_results,
    run_command,
)


def generate_artifact():
    parser = argparse.ArgumentParser(description="Run PAF simulation workflow")
    parser.add_argument(
        "-l",
        "--location",
        type=str,
        default="Ethiopia",
        help="Location for the simulation (default: 'Ethiopia')",
    )
    parser.add_argument(
        "-a",
        "--artifact_name",
        type=str,
        required=True,
        dest="artifact_name",
        help="Name of the artifact directory. Will write to /mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/artifacts/{artifact_name}",
    )

    args = parser.parse_args()

    location = args.location.lower()
    artifact_name = args.artifact_name

    # Define artifact path
    artifact_path = f"/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/artifacts/{artifact_name}"

    # Create timestamped working directory on shared filesystem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    working_dir = Path(CLUSTER_DATA_DIR) / "paf_sim_results" / timestamp
    working_dir.mkdir(parents=True, exist_ok=True)
    working_dir_str = str(working_dir)

    # Get data/lbwsg_paf
    script_dir = Path(__file__).parent.parent

    print("\n" + "=" * 80)
    print("PAF Simulation Workflow")
    print("=" * 80)
    print(f"Location: {location}")
    print(f"Artifact: {artifact_path}")
    print(f"Working directory: {working_dir_str}")
    print("=" * 80)

    try:
        # Check for required conda environments
        check_conda_environments()

        # Step 1: Initial artifact generation
        run_command(
            ["make_artifacts", "-vvv", "-l", location.capitalize(), "-o", artifact_path],
            f"initial artifact generation for {location.capitalize()}",
            conda_env="vivarium_gates_mncnh_artifact",
            auto_confirm=True,
        )

        # Step 2: Run first psimulate with early neonatal spec (only one time step)
        psimulate_output = run_command(
            [
                "psimulate",
                "run",
                "-vvv",
                "-P",
                "proj_simscience_prod",
                "-i",
                f"{artifact_path}/{location}.hdf",
                "-o",
                working_dir_str,
                str(script_dir / "code" / "lbwsg_paf_enn.yaml"),
                str(script_dir / "code" / "lbwsg_paf_branches.yaml"),
            ],
            "first psimulate run (early neonatal PAFs)",
            conda_env="vivarium_gates_mncnh_simulation",
            capture_full_output=True,
        )

        # Step 3: Check if psimulate finished properly
        if not check_psimulate_finished(psimulate_output):
            raise RuntimeError("First psimulate run: Not all jobs finished successfully")

        # Step 4: Copy first set of results
        results_dir = extract_results_dir(psimulate_output)
        move_results(
            f"{results_dir}/calculated_lbwsg_paf*",
            f"{script_dir}/outputs/paf_outputs/{location}",
            "early neonatal PAF output files",
        )

        # Step 5: Generate artifact with early neonatal PAFs
        run_command(
            [
                "make_artifacts",
                "-vvv",
                "-l",
                location.capitalize(),
                "-o",
                artifact_path,
                "-r",
                "risk_factor.low_birth_weight_and_short_gestation.population_attributable_fraction",
                "-r",
                "cause.neonatal_preterm_birth.population_attributable_fraction",
            ],
            f"early neonatal artifact generation for {location.capitalize()}",
            conda_env="vivarium_gates_mncnh_artifact",
            auto_confirm=True,
        )

        # Step 6: Run second psimulate with standard model spec (two time steps)
        psimulate_output = run_command(
            [
                "psimulate",
                "run",
                "-vvv",
                "-P",
                "proj_simscience_prod",
                "-i",
                f"{artifact_path}/{location}.hdf",
                "-o",
                working_dir_str,
                str(script_dir / "code" / "lbwsg_paf.yaml"),
                str(script_dir / "code" / "lbwsg_paf_branches.yaml"),
            ],
            "second psimulate run (late neonatal PAFs and preterm prevalence)",
            conda_env="vivarium_gates_mncnh_simulation",
            capture_full_output=True,
        )

        # Step 7: Check if psimulate finished properly
        if not check_psimulate_finished(psimulate_output):
            # Error if not finished
            raise RuntimeError("Second psimulate run: Not all jobs finished successfully")

        # Step 8: Copy second set of results
        results_dir = extract_results_dir(psimulate_output)
        move_results(
            f"{results_dir}/calculated_lbwsg_paf*",
            f"{script_dir}/outputs/paf_outputs/{location}",
            "late neonatal PAF output files",
        )

        move_results(
            f"{results_dir}/calculated_late_neonatal_preterm*",
            f"{script_dir}/outputs/preterm_prevalence_outputs/{location}",
            "preterm prevalence output files",
        )

        # Step 9: Final artifact generation
        run_command(
            [
                "make_artifacts",
                "-vvv",
                "-l",
                location.capitalize(),
                "-o",
                artifact_path,
                "-r",
                "risk_factor.low_birth_weight_and_short_gestation.population_attributable_fraction",
                "-r",
                "cause.neonatal_preterm_birth.population_attributable_fraction",
            ],
            f"final artifact generation for {location.capitalize()}",
            conda_env="vivarium_gates_mncnh_artifact",
            auto_confirm=True,
        )

        print("\n" + "=" * 80)
        print("PAF Simulation Workflow Completed Successfully!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"ERROR: {str(e)}", file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        sys.exit(1)
    finally:
        if working_dir.exists():
            shutil.rmtree(working_dir, ignore_errors=True)


if __name__ == "__main__":
    generate_artifact()
