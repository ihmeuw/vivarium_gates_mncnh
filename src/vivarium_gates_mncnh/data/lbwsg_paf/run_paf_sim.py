#!/usr/bin/env python
"""
Script to run PAF simulations to generate artifact data..
"""
import argparse
import glob
import re
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional

from vivarium.framework.configuration import build_model_specification

# Suppress pandas PerformanceWarning from vivarium_inputs
warnings.filterwarnings("ignore", message=".*DataFrame is highly fragmented.*")


def check_conda_environments() -> None:
    """
    Check that required conda environments are installed.

    Raises
    ------
    RuntimeError
        If required conda environments are not found
    """
    required_envs = ["vivarium_gates_mncnh_simulation", "vivarium_gates_mncnh_artifact"]

    print("\nChecking for required conda environments...")

    try:
        result = subprocess.run(
            ["conda", "env", "list"], capture_output=True, text=True, check=True
        )

        installed_envs = result.stdout
        missing_envs = []

        for env in required_envs:
            if env not in installed_envs:
                missing_envs.append(env)
            else:
                print(f"  ✓ Found: {env}")

        if missing_envs:
            raise RuntimeError(
                f"Missing required conda environments: {', '.join(missing_envs)}\n"
                f"Please install them by running bash environment.sh before running this script."
            )

        print("All required conda environments found.\n")

    except FileNotFoundError:
        raise RuntimeError(
            "conda command not found. Please ensure conda is installed and in your PATH."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to check conda environments: {e}")


def run_command(
    cmd: List[str],
    description: str,
    conda_env: str,
    auto_confirm: bool = False,
    capture_full_output: bool = False,
) -> str | None:
    """
    Run a shell command and handle errors.

    Parameters
    ----------
    cmd : List[str]
        Command and arguments to execute
    description : str
        Description of what the command does (for error messages)
    conda_env : str
        Name of the conda environment to run the command in
    auto_confirm : bool, optional
        If True, automatically answer 'y' to any prompts (useful for make_artifacts)
    capture_full_output : bool, optional
        If True, capture and return the full command output as a string

    Returns
    -------
    str | None
        The full output if capture_full_output is True, otherwise None
    """
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Environment: {conda_env}")

    # Build the command
    if auto_confirm:
        # Use shell with 'yes y' to continuously pipe 'y' to the command
        cmd_str = " ".join(cmd)
        full_cmd = f"yes y | conda run --no-capture-output -n {conda_env} {cmd_str}"
        print(f"Command: {full_cmd}")
        print("Auto-confirm: y (continuous)")
    else:
        cmd = ["conda", "run", "-n", conda_env] + cmd
        print(f"Command: {' '.join(cmd)}")

    print(f"{'='*80}\n")

    full_output = []

    if capture_full_output:
        # Use Popen to capture output in real-time
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if auto_confirm else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # If auto_confirm, send a single 'y' response
        if auto_confirm:
            # Send in a separate thread to avoid blocking
            import threading

            def send_confirm():
                try:
                    time.sleep(0.1)  # Brief delay to ensure prompt is ready
                    process.stdin.write("y\n")
                    process.stdin.flush()
                    process.stdin.close()
                except:
                    pass

            confirm_thread = threading.Thread(target=send_confirm, daemon=True)
            confirm_thread.start()

        # Read output line by line
        for line in process.stdout:
            # Print the line to maintain visibility
            print(line, end="")

            # Capture output if requested
            if capture_full_output:
                full_output.append(line)

        # Wait for process to complete
        return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(f"Failed {description}. Exit code: {return_code}")

        return "".join(full_output)
    else:
        # Print output to screen
        if auto_confirm:
            # Use shell command with yes to pipe 'y'
            result = subprocess.run(full_cmd, shell=True)

            if result.returncode != 0:
                raise RuntimeError(f"Failed {description}. Exit code: {result.returncode}")
        else:
            # Normal execution with conda run
            result = subprocess.run(cmd)

            if result.returncode != 0:
                raise RuntimeError(f"Failed {description}. Exit code: {result.returncode}")


def create_temp_model_spec(source_yaml_path: str, temp_dir: Path, end_day: int) -> str:
    """
    Create a temporary model specification with specified end day.

    Parameters
    ----------
    source_yaml_path : str
        Path to the source model specification YAML file
    temp_dir : Path
        Directory to create temporary model spec in
    end_day : int
        End day value for the simulation

    Returns
    -------
    str
        Path to the created temporary model spec file
    """
    import yaml

    # Create temp directory if it doesn't exist
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Load the source YAML file
    with open(source_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Update the end day
    config["configuration"]["time"]["end"]["day"] = end_day

    # Create temp file path
    temp_file = temp_dir / f"lbwsg_paf_end_day_{end_day}.yaml"

    # Write to temp file
    with open(temp_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Created temporary model spec: {temp_file}")
    print(f"  End day: {end_day}\n")

    return str(temp_file)


def check_psimulate_finished(psimulate_output: str) -> bool:
    """
    Check whether psimulate finished properly by parsing its output.

    Parses the second-to-last line of psimulate output which should be:
    "(M of N total jobs completed successfully overall)"

    Parameters
    ----------
    psimulate_output : str
        The full output from the psimulate command

    Returns
    -------
    bool
        True if M equals N (all jobs completed successfully), False otherwise
    """
    print("\nChecking psimulate completion status...")

    # Split output into lines and get the last non-empty lines
    lines = [line.strip() for line in psimulate_output.strip().split("\n") if line.strip()]

    # Parse the second-to-last line for job completion status
    # Expected format: "(M of N total jobs completed successfully overall)"
    completion_line = lines[-2]
    completion_pattern = re.compile(
        r"\((\d+) of (\d+) total jobs completed successfully overall\)"
    )
    completion_match = completion_pattern.search(completion_line)

    if completion_match:
        completed = int(completion_match.group(1))
        total = int(completion_match.group(2))
        print(f"Job completion: {completed} of {total}")

        if completed == total:
            print("✓ All jobs completed successfully")
            return True
        else:
            print(f"✗ Only {completed} of {total} jobs completed successfully")
            return False
    else:
        print(f"WARNING: Could not parse completion status from: {completion_line}")
        return False


def extract_results_dir(psimulate_output: str) -> Optional[str]:
    """
    Extract the results directory from psimulate output.

    Parses the last line of psimulate output which should be:
    "Results written to: {results_dir}"

    Parameters
    ----------
    psimulate_output : str
        The full output from the psimulate command

    Returns
    -------
    Optional[str]
        The results directory path, or None if not found
    """
    # Split output into lines and get the last non-empty lines
    lines = [line.strip() for line in psimulate_output.strip().split("\n") if line.strip()]

    # Parse the last line for results directory
    # Expected format: "Results written to: {results_dir}"
    results_line = lines[-1]
    results_pattern = re.compile(r"Results written to:\s*(.+)")
    results_match = results_pattern.search(results_line)

    if results_match:
        results_dir = results_match.group(1).strip()
        # Remove ANSI escape codes (color codes) from the path
        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        results_dir = ansi_escape.sub("", results_dir)
        print(f"Results directory: {results_dir}")
        return results_dir
    else:
        print(f"WARNING: Could not parse results directory from: {results_line}")
        return None


def copy_results(source_pattern: str, dest_dir: str, description: str) -> None:
    """
    Copy result files to the destination directory.

    Expects directories matching the pattern, each containing a single file 0000.parquet.
    Copies each parquet file and renames it to {directory_name}.parquet.

    Parameters
    ----------
    source_pattern : str
        Source directory pattern (can include wildcards)
    dest_dir : str
        Destination directory
    description : str
        Description of the files being copied
    """
    print(f"\nCopying {description}")
    print(f"From: {source_pattern}")
    print(f"To: {dest_dir}")

    # Create destination directory if it doesn't exist
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    # Expand wildcards using glob
    matching_paths = glob.glob(source_pattern)

    if not matching_paths:
        raise RuntimeError(
            f"Failed to copy {description}. No directories matched pattern: {source_pattern}"
        )

    # Copy files from matching directories
    copied_count = 0
    for source_path in matching_paths:
        source_path_obj = Path(source_path)

        try:
            # Look for 0000.parquet file
            parquet_file = source_path_obj / "0000.parquet"
            if not parquet_file.exists():
                raise RuntimeError(f"Expected file 0000.parquet not found in {source_path}")

            # Copy and rename to {directory_name}.parquet
            dest_filename = f"{source_path_obj.name}.parquet"
            dest_file = Path(dest_dir) / dest_filename
            shutil.copy2(parquet_file, dest_file)
            copied_count += 1
        except Exception as e:
            raise RuntimeError(f"Failed to copy from {source_path} to {dest_dir}. Error: {e}")

    print(f"Copied {copied_count} file(s) successfully\n")


def main():
    parser = argparse.ArgumentParser(description="Run PAF simulation workflow")
    parser.add_argument(
        "-l",
        "--location",
        type=str,
        default="Ethiopia",
        help="Location for the simulation (default: 'Ethiopia')",
    )
    parser.add_argument(
        "-o",
        "--intermediate_dir",
        type=str,
        default="~/",
        help="Directory for intermediate files and results (default: '~/')",
    )
    parser.add_argument(
        "-a",
        "--artifact_name",
        type=str,
        required=True,
        dest="artifact_name",
        help="Name of the artifact file. Will write to /mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/artifacts/{artifact_name}",
    )

    args = parser.parse_args()

    location = args.location.lower()
    artifact_name = args.artifact_name
    intermediate_dir = args.intermediate_dir

    # Define artifact path
    artifact_path = f"/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/artifacts/{artifact_name}"

    # Create working directory inside intermediate_dir
    working_dir = Path(intermediate_dir).expanduser() / "paf_sim_results"
    working_dir.mkdir(parents=True, exist_ok=True)
    working_dir_str = str(working_dir)

    # Get the script directory to find data/lbwsg_paf
    script_dir = Path(__file__).parent

    print("\n" + "=" * 80)
    print("PAF Simulation Workflow")
    print("=" * 80)
    print(f"Location: {location}")
    print(f"Artifact: {artifact_path}")
    print(f"Intermediate directory: {intermediate_dir}")
    print(f"Working directory: {working_dir_str}")
    print("=" * 80)

    try:
        # Check for required conda environments
        check_conda_environments()

        # Step 1: Initial artifact generation
        # run_command(
        #     ["make_artifacts", "-vvv", "-l", location.capitalize(), "-o", artifact_path],
        #     f"initial artifact generation for {location.capitalize()}",
        #     conda_env="vivarium_gates_mncnh_artifact",
        #     auto_confirm=True,
        # )

        # Step 2: Create temporary model spec with end day = 2
        source_model_spec_path = script_dir / "lbwsg_paf.yaml"
        temp_specs_dir = script_dir / "temp_model_specs"
        model_spec_day2 = create_temp_model_spec(
            str(source_model_spec_path), temp_specs_dir, 2
        )

        # Step 3: Run first psimulate
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
                model_spec_day2,
                str(script_dir / "lbwsg_paf_branches.yaml"),
            ],
            "first psimulate run (early neonatal PAFs)",
            conda_env="vivarium_gates_mncnh_simulation",
            capture_full_output=True,
        )

        # Step 4: Check if psimulate finished properly
        if not check_psimulate_finished(psimulate_output):
            raise RuntimeError("First psimulate run: Not all jobs finished successfully")

        # Step 5: Copy first set of results
        results_dir = extract_results_dir(psimulate_output)
        copy_results(
            f"{results_dir}/calculated_lbwsg_paf*",
            f"{script_dir}/outputs/paf_outputs/{location}",
            "early neonatal PAF output files",
        )

        # Step 6: Generate artifact with early neonatal PAFs
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

        # Step 7: Run second psimulate
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
                str(source_model_spec_path),
                str(script_dir / "lbwsg_paf_branches.yaml"),
            ],
            "second psimulate run (late neonatal PAFs and preterm prevalence)",
            conda_env="vivarium_gates_mncnh_simulation",
            capture_full_output=True,
        )

        # Step 8: Check if psimulate finished properly
        if not check_psimulate_finished(psimulate_output):
            # Error if not finished
            raise RuntimeError("Second psimulate run: Not all jobs finished successfully")

        # Step 9: Copy second set of results
        results_dir = extract_results_dir(psimulate_output)
        copy_results(
            f"{results_dir}/calculated_lbwsg_paf*",
            f"{script_dir}/outputs/paf_outputs/{location}",
            "late neonatal PAF output files",
        )

        copy_results(
            f"{results_dir}/calculated_late_neonatal_preterm*",
            f"{script_dir}/outputs/preterm_prevalence_outputs/{location}",
            "preterm prevalence output files",
        )

        # Step 10: Final artifact generation
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

        # Cleanup: Delete temporary model specs
        print(f"\nCleaning up temporary model specs: {temp_specs_dir}")
        if temp_specs_dir.exists():
            shutil.rmtree(temp_specs_dir)
            print(f"Temporary model specs deleted successfully\n")
        else:
            print(f"Temporary model specs directory does not exist, skipping cleanup\n")

    except Exception as e:
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"ERROR: {str(e)}", file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
