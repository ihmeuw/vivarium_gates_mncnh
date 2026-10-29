"""
Module to test Jupyter notebooks in the model_notebooks directory. This module provides  
framework to run each notebook and record whether the notebook runs successfully or not. 
If the notebook has any cell that errors out, the test will fail.
"""
import fcntl
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import papermill as pm
import pytest
from loguru import logger
from pytest_vivarium.plugin import IS_ON_SLURM

from vivarium_gates_mncnh.constants.paths import MODEL_NOTEBOOKS_DIR


def discover_notebook_paths(notebook_directory) -> List[Path]:
    """
    Discover all Jupyter notebook files in the notebook directory.

    Searches only the top level of the directory (non-recursive) and filters
    out checkpoint files.

    Returns:
        List of Path objects for discovered notebooks
    """
    # Filter out checkpoint files and nbdime merge conflict artifacts (e.g. *_BACKUP_*, *_BASE_*, *_LOCAL_*, *_REMOTE_*)
    nbdime_suffixes = ("_BACKUP_", "_BASE_", "_LOCAL_", "_REMOTE_")
    notebooks = [
        nb
        for nb in notebook_directory.glob("*.ipynb")
        if ".ipynb_checkpoints" not in str(nb)
        and not any(suffix in nb.stem for suffix in nbdime_suffixes)
    ]

    logger.info(f"Found {len(notebooks)} notebook(s) in {notebook_directory}")

    return sorted(notebooks)


class NotebookTestRunner:
    """
    A class to execute and test Jupyter notebook using papermill.

    This runner executes the notebook with
    papermill while injecting specified parameters, and reports success/failure.

    All exceptions (including SystemExit and KeyboardInterrupt) are caught and
    converted to test failures to prevent pytest exit code 2, which would kill
    SLURM sessions. Tests fail with exit code 1 instead, allowing the test suite
    to continue running.

    Attributes:
        notebook_path: Path of notebook to run
        parameters: Dictionary of parameters to inject into notebook
    """

    def __init__(
        self,
        notebook_path: str,
        parameters: Dict[str, any] = {},
    ):
        """
        Initialize the notebook test runner.

        Args:
            notebook_path: Path of notebook to run
            parameters: Dictionary of parameters to inject into notebook

        Raises:
            FileNotFoundError: If notebook_path does not exist
        """
        self.notebook_path = Path(notebook_path)
        self.parameters = parameters
        self.timeout = -1  # No timeout by default
        # Run notebooks with the interpreter that runs pytest. Every worktree's
        # environment has the same name, so a hash of its path keeps their kernels apart.
        self.environment_python = Path(sys.executable)
        environment_dir = Path(sys.prefix).resolve()
        path_hash = hashlib.sha1(str(environment_dir).encode()).hexdigest()[:8]
        self.kernel_name = f"vgm-nb-{environment_dir.name}-{path_hash}"

        # Validate notebook directory exists
        if not self.notebook_path.exists():
            raise FileNotFoundError(f"Notebook does not exist: {self.notebook_path}")

        if self.notebook_path.is_dir():
            raise IsADirectoryError(f"Path is a directory: {self.notebook_path}")

        # Ensure the required Jupyter kernel is registered
        self._ensure_kernel_registered()

    def _check_kernel_exists(self) -> bool:
        """
        Check if the required Jupyter kernel is already registered.

        Returns:
            True if kernel exists, False otherwise
        """
        try:
            result = subprocess.run(
                ["jupyter", "kernelspec", "list", "--json"],
                capture_output=True,
                text=True,
                check=True,
            )
            kernelspecs = json.loads(result.stdout)
            kernel_exists = self.kernel_name in kernelspecs.get("kernelspecs", {})

            if kernel_exists:
                logger.debug(f"Kernel '{self.kernel_name}' is already registered")
            else:
                logger.debug(f"Kernel '{self.kernel_name}' not found")

            return kernel_exists
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Could not check kernel specs: {e}")
            return False

    def _register_kernel(self) -> bool:
        """
        Register the environment as a Jupyter kernel.

        Returns:
            True if registration succeeded, False otherwise
        """
        try:
            logger.info(
                f"Registering kernel '{self.kernel_name}' for {self.environment_python}"
            )

            cmd = [
                str(self.environment_python),
                "-m",
                "ipykernel",
                "install",
                "--user",
                "--name",
                self.kernel_name,
                "--display-name",
                "Python (vivarium_gates_mncnh)",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            logger.success(f"Successfully registered kernel '{self.kernel_name}'")
            logger.debug(f"Output: {result.stdout}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to register kernel: {e.stderr}")
            return False

    def _ensure_kernel_registered(self) -> None:
        """
        Ensure the required Jupyter kernel is registered, registering it if necessary.

        Raises:
            RuntimeError: If kernel cannot be registered
        """
        # pytest workers register one at a time: each install deletes and rewrites the kernel.
        lock_path = Path(tempfile.gettempdir()) / f"{self.kernel_name}.lock"
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if not self._check_kernel_exists():
                logger.info(f"Kernel '{self.kernel_name}' not found, attempting to register...")

                if not self._register_kernel():
                    raise RuntimeError(
                        f"Failed to register kernel '{self.kernel_name}'. "
                        f"Ensure environment {self.environment_python} has ipykernel installed."
                    )

    def _execute_notebook(self, notebook_path: Path) -> bool:
        """
        Execute a single notebook using papermill.

        Catches all exceptions including SystemExit and KeyboardInterrupt to prevent
        pytest from exiting with code 2 (which would kill the SLURM session).
        All failures are converted to regular test failures (AssertionError).

        Args:
            notebook_path: Path to the notebook to execute

        Returns:
            True if execution succeeded, False if it failed
        """
        try:
            logger.info(f"Running notebook {notebook_path.name}...")
            logger.info(f"Using kernel: {self.kernel_name}")

            # Execute notebook with papermill
            # The kernel is automatically registered in __init__ if not already present
            pm.execute_notebook(
                input_path=str(notebook_path),
                output_path=notebook_path.parent / "executed" / notebook_path.name,
                parameters=self.parameters,
                kernel_name=self.kernel_name,
                execution_timeout=self.timeout,
                autosave_cell_every=10,
            )

            logger.success(f"✓ {notebook_path.name} completed successfully")

            return True

        except BaseException as e:
            # Catch ALL exceptions including SystemExit and KeyboardInterrupt
            # to prevent pytest exit code 2 which kills the SLURM session.
            # This ensures test failures (exit code 1) instead of interruptions (exit code 2).
            error_msg = f"{type(e).__name__}: {str(e)}" if str(e) else type(e).__name__
            logger.error(f"✗ {notebook_path.name} failed with error: {error_msg}")

            return False

    def test_run_notebook(self) -> None:
        """
        Execute notebook and track results.

        This method orchestrates the testing process:
        1. Executes notebook with papermill
        2. Catches all exceptions (including SystemExit/KeyboardInterrupt) to prevent
           pytest exit code 2 which would kill SLURM sessions
        3. Tracks successes and failures
        4. Logs summary of results
        5. Raises AssertionError (exit code 1) if notebook failed

        This ensures notebooks can fail without terminating the SLURM session.

        Raises:
            AssertionError: If notebook failed to execute successfully
        """
        # Execute each notebook
        if self._execute_notebook(self.notebook_path):
            logger.success("Notebook passed!")
        else:
            raise AssertionError(f"Notebook failed to execute: {self.notebook_path}")


# Pytest test functions
@pytest.mark.slow
@pytest.mark.parametrize(
    "notebook_path",
    discover_notebook_paths(MODEL_NOTEBOOKS_DIR / "interactive"),
    ids=lambda x: x.stem,
)
def test_interactive_context_notebooks(notebook_path) -> None:
    """Test all notebooks in the interactive directory."""
    # Skip test if not on SLURM
    if not IS_ON_SLURM:
        pytest.skip("Test skipped: must be run on SLURM cluster")

    runner = NotebookTestRunner(
        notebook_path=notebook_path,
    )
    runner.test_run_notebook()


@pytest.mark.slow
@pytest.mark.parametrize(
    "notebook_path",
    discover_notebook_paths(MODEL_NOTEBOOKS_DIR / "results"),
    ids=lambda x: x.stem,
)
def test_results_notebook(notebook_path) -> None:
    """Test all notebooks in the results directory."""
    # Skip test if not on SLURM
    if not IS_ON_SLURM:
        pytest.skip("Test skipped: must be run on SLURM cluster")

    runner = NotebookTestRunner(
        notebook_path=notebook_path,
    )
    runner.test_run_notebook()


@pytest.mark.slow
@pytest.mark.parametrize(
    "notebook_path",
    discover_notebook_paths(MODEL_NOTEBOOKS_DIR / "artifact"),
    ids=lambda x: x.stem,
)
def test_artifact_notebooks(notebook_path) -> None:
    """
    Test notebooks in the artifact directory.
    """
    # Skip test if not on SLURM
    if not IS_ON_SLURM:
        pytest.skip("Test skipped: must be run on SLURM cluster")

    runner = NotebookTestRunner(
        notebook_path=notebook_path,
    )
    runner.test_run_notebook()
