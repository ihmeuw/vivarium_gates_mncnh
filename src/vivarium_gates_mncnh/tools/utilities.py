"""
Utility functions for PAF simulation workflow.
"""
import glob
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import pyarrow.parquet as pq


def tag_commit(tag: str) -> Optional[str]:
    """Return the commit SHA a tag points to, or ``None`` if it doesn't exist."""
    result = subprocess.run(
        ["git", "rev-list", "-n1", tag],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def head_commit() -> str:
    """Return the SHA of the current HEAD."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def commit_pending_changes(message: str) -> None:
    """Commit any uncommitted tracked-file changes with ``message`` and push to origin.

    No-op if the working tree has nothing to commit. Untracked files are ignored.
    """
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    if not status.stdout.strip():
        print("No pending changes to commit.")
        return

    try:
        subprocess.run(["git", "add", "-u"], check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "commit", "-m", message], check=True, capture_output=True, text=True
        )
        print(f"Committed pending changes: {message!r}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to commit pending changes.\n  stderr: {e.stderr.strip()}")

    try:
        subprocess.run(
            ["git", "push", "origin", "HEAD"], check=True, capture_output=True, text=True
        )
        print("Pushed commit to origin.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to push commit to origin.\n  stderr: {e.stderr.strip()}")


def create_and_push_tag(model_number: str) -> None:
    """Create a git tag ``v{model_number}`` for the current HEAD and push it to origin.

    If the tag already exists on the current commit, it is left as-is. If it
    exists on a *different* commit the user is prompted to force-update;
    when stdin is not a TTY (e.g. running inside a jobmon task) the prompt
    aborts cleanly instead of crashing.
    """
    tag = f"v{model_number}"
    force = False

    existing_commit = tag_commit(tag)
    if existing_commit is not None:
        head = head_commit()
        if existing_commit == head:
            print(f"\nGit tag '{tag}' already exists on the current commit. Skipping.")
            return
        print(f"\nWARNING: Git tag '{tag}' already exists (on commit {existing_commit[:8]}).")
        try:
            response = input(
                f"Update tag '{tag}' to the current commit and force-push? [y/N] "
            )
        except EOFError:
            raise RuntimeError(
                f"Git tag '{tag}' already exists on a different commit "
                f"({existing_commit[:8]}). Refusing to force-update non-interactively. "
                "Resolve the tag conflict manually before re-running."
            )
        if response.strip().lower() != "y":
            raise RuntimeError(f"Aborted: tag '{tag}' already exists.")
        force = True

    print(f"\n{'Updating' if force else 'Creating'} git tag '{tag}' and pushing to origin...")

    try:
        cmd = ["git", "tag", "-f", tag] if force else ["git", "tag", tag]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  {'Updated' if force else 'Created'} tag '{tag}'")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to create git tag '{tag}'.\n  stderr: {e.stderr.strip()}")

    try:
        cmd = (
            ["git", "push", "--force", "origin", tag]
            if force
            else ["git", "push", "origin", tag]
        )
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  Pushed tag '{tag}' to origin")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to push git tag '{tag}' to origin.\n  stderr: {e.stderr.strip()}"
        )


def check_clean_tree() -> None:
    """Abort if there are uncommitted changes to tracked files in src/vivarium_gates_mncnh,
    excluding the validation/ and tools/ subdirectories."""
    result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "src/vivarium_gates_mncnh",
            ":!src/vivarium_gates_mncnh/validation",
            ":!src/vivarium_gates_mncnh/tools",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "There are uncommitted changes to tracked files in src/vivarium_gates_mncnh "
            "(excluding validation/ and tools/). "
            "Please commit or stash them before running this script.\n"
            f"{result.stdout.strip()}"
        )


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
                f"Please install them by running 'source environment.sh' before running this script."
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
        cmd = ["conda", "run", "--no-capture-output", "-n", conda_env] + cmd
        print(f"Command: {' '.join(cmd)}")

    print(f"{'='*80}\n")

    full_output = []

    if capture_full_output:
        # Use Popen to capture output in real-time.
        # Start in a new process group so we can kill the entire tree on interrupt.
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if auto_confirm else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            start_new_session=True,
        )

        try:
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
        except KeyboardInterrupt:
            print(f"\nInterrupted. Terminating {description}...")
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            raise

        if return_code != 0:
            raise RuntimeError(f"Failed {description}. Exit code: {return_code}")

        return "".join(full_output)
    else:
        # Print output to screen
        try:
            if auto_confirm:
                # Use shell command with yes to pipe 'y'
                result = subprocess.run(
                    full_cmd,
                    shell=True,
                    start_new_session=True,
                )

                if result.returncode != 0:
                    raise RuntimeError(
                        f"Failed {description}. Exit code: {result.returncode}"
                    )
            else:
                # Normal execution with conda run
                result = subprocess.run(cmd, start_new_session=True)

                if result.returncode != 0:
                    raise RuntimeError(
                        f"Failed {description}. Exit code: {result.returncode}"
                    )
        except KeyboardInterrupt:
            print(f"\nInterrupted. Terminating {description}...")
            raise


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


def move_results(source_pattern: str, dest_dir: str, description: str) -> None:
    """
    Move result files to the destination directory.

    Expects directories matching the pattern. Each directory contains one or more
    partition parquet files written by psimulate. They are concatenated into a
    single parquet file at ``{dest_dir}/{directory_name}.parquet`` and the source
    directory is removed.

    Parameters
    ----------
    source_pattern : str
        Source directory pattern (can include wildcards)
    dest_dir : str
        Destination directory
    description : str
        Description of the files being moved
    """
    print(f"\nMoving {description}")
    print(f"From: {source_pattern}")
    print(f"To: {dest_dir}")

    # Create destination directory if it doesn't exist
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    # Expand wildcards using glob
    matching_paths = glob.glob(source_pattern)

    if not matching_paths:
        raise RuntimeError(
            f"Failed to move {description}. No directories matched pattern: {source_pattern}"
        )

    moved_count = 0
    for source_path in matching_paths:
        source_path_obj = Path(source_path)

        try:
            partition_files = sorted(source_path_obj.glob("*.parquet"))
            if not partition_files:
                raise RuntimeError(f"No parquet partition files found in {source_path}")

            combined = pq.ParquetDataset([str(p) for p in partition_files]).read()

            dest_file = Path(dest_dir) / f"{source_path_obj.name}.parquet"
            pq.write_table(combined, dest_file)

            shutil.rmtree(source_path_obj)
            moved_count += 1
        except Exception as e:
            raise RuntimeError(f"Failed to move from {source_path} to {dest_dir}. Error: {e}")

    print(f"Moved {moved_count} file(s) successfully\n")
