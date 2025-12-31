"""
Module to test Jupyter notebooks in the model_notebooks directory. This module provides  
framework to run each notebook and record whether the notebook runs successfully or not. 
If the notebook has any cell that errors out, the test will fail.
"""
import cmd
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import papermill as pm
import pytest
from loguru import logger

from tests.conftest import IS_ON_SLURM
from vivarium_gates_mncnh.constants.paths import MODEL_NOTEBOOKS_DIR, MODEL_RESULTS_DIR


class NotebookTestRunner:
    """
    A class to execute and test Jupyter notebooks using papermill.

    This runner discovers notebooks in a specified directory, executes them with
    papermill while injecting a results_dir parameter, and tracks success/failure
    of each notebook execution.

    All exceptions (including SystemExit and KeyboardInterrupt) are caught and
    converted to test failures to prevent pytest exit code 2, which would kill
    SLURM sessions. Tests fail with exit code 1 instead, allowing the test suite
    to continue running.

    Attributes:
        notebook_directory: Directory containing notebooks to test
        results_dir: Directory path to inject into notebooks as a parameter
        timeout: Maximum execution time per notebook in seconds
        cleanup_notebooks: Whether to delete executed notebooks after testing
        notebooks_found: List of discovered notebook paths
        successful_notebooks: List of successfully executed notebook paths
        failed_notebooks: Dictionary mapping failed notebook paths to their exceptions
    """

    def __init__(
        self,
        notebook_directory: str,
        environment_type: str = "simulation",
    ):
        """
        Initialize the notebook test runner.

        Args:
            notebook_directory: Path to directory containing notebooks to test
            model_dir: Path to model results directory (injected as parameter into notebooks)
            timeout: Maximum execution time per notebook in seconds (default: 300 = 5 minutes)
            cleanup_notebooks: Whether to delete executed notebooks after testing (default: True)

        Raises:
            FileNotFoundError: If notebook_directory does not exist
        """
        self.notebook_directory = Path(notebook_directory)
        self.model_dir = Path(MODEL_RESULTS_DIR)
        self.timeout = -1  # No timeout by default
        self.environment_type = environment_type
        self.conda_env_name = f"vivarium_gates_mncnh_{environment_type}"
        self.kernel_name = f"conda-env-{self.conda_env_name}-py"

        self.notebooks_found: List[Path] = []
        self.successful_notebooks: List[Path] = []
        self.failed_notebooks: Dict[Path, Exception] = {}

        # Validate notebook directory exists
        if not self.notebook_directory.exists():
            raise FileNotFoundError(
                f"Notebook directory does not exist: {self.notebook_directory}"
            )

        if not self.notebook_directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.notebook_directory}")

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
        Register the conda environment as a Jupyter kernel.

        Returns:
            True if registration succeeded, False otherwise
        """
        try:
            logger.info(
                f"Registering kernel '{self.kernel_name}' for conda env '{self.conda_env_name}'"
            )

            # Use conda run to execute ipykernel install in the target environment
            cmd = (
                f" conda run -n {self.conda_env_name} python -m ipykernel install --user --name {self.kernel.name} "
                "--display-name Python (vivarium_gates_mncnh_{self.environment_type})"
            )
            result = subprocess.run(
                cmd.split(" "), capture_output=True, text=True, check=True
            )

            logger.success(f"Successfully registered kernel '{self.kernel_name}'")
            logger.debug(f"Output: {result.stdout}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to register kernel: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("conda command not found. Ensure conda is in PATH.")
            return False

    def _ensure_kernel_registered(self) -> None:
        """
        Ensure the required Jupyter kernel is registered, registering it if necessary.

        Raises:
            RuntimeError: If kernel cannot be registered
        """
        if not self._check_kernel_exists():
            logger.info(f"Kernel '{self.kernel_name}' not found, attempting to register...")

            if not self._register_kernel():
                raise RuntimeError(
                    f"Failed to register kernel '{self.kernel_name}'. "
                    f"Ensure conda environment '{self.conda_env_name}' exists and "
                    f"has ipykernel installed."
                )

    def _discover_notebook_paths(self) -> List[Path]:
        """
        Discover all Jupyter notebook files in the notebook directory.

        Searches only the top level of the directory (non-recursive) and filters
        out checkpoint files.

        Returns:
            List of Path objects for discovered notebooks
        """
        notebooks = [
            nb
            for nb in self.notebook_directory.glob("*.ipynb")
            if ".ipynb_checkpoints" not in str(nb)
        ]

        logger.info(f"Found {len(notebooks)} notebook(s) in {self.notebook_directory}")

        return sorted(notebooks)

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
                output_path=str(notebook_path),
                parameters={"model_dir": str(self.model_dir)},
                kernel_name=self.kernel_name,
                execution_timeout=self.timeout,
            )

            logger.success(f"✓ {notebook_path.name} completed successfully")

            return True

        except BaseException as e:
            # Catch ALL exceptions including SystemExit and KeyboardInterrupt
            # to prevent pytest exit code 2 which kills the SLURM session.
            # This ensures test failures (exit code 1) instead of interruptions (exit code 2).
            error_msg = f"{type(e).__name__}: {str(e)}" if str(e) else type(e).__name__
            logger.error(f"✗ {notebook_path.name} failed with error: {error_msg}")
            self.failed_notebooks[notebook_path] = e

            return False

    def test_run_notebooks(self) -> None:
        """
        Execute all discovered notebooks and track results.

        This method orchestrates the testing process:
        1. Discovers notebooks in the directory
        2. Executes each notebook with papermill
        3. Catches all exceptions (including SystemExit/KeyboardInterrupt) to prevent
           pytest exit code 2 which would kill SLURM sessions
        4. Tracks successes and failures
        5. Logs summary of results
        6. Raises AssertionError (exit code 1) if any notebooks failed

        This ensures notebooks can fail without terminating the SLURM session.

        Raises:
            AssertionError: If any notebooks failed to execute successfully
        """
        logger.info("Starting notebook test run")
        logger.info(f"Notebook directory: {self.notebook_directory}")
        logger.info(f"Model directory: {self.model_dir}")
        logger.info(f"Kernel name: {self.kernel_name}")
        # Discover notebooks
        self.notebooks_found = self._discover_notebook_paths()

        # Handle empty directory case
        if not self.notebooks_found:
            logger.info("No notebooks found")
            return

        # Execute each notebook
        for notebook_path in self.notebooks_found:
            if self._execute_notebook(notebook_path):
                self.successful_notebooks.append(notebook_path)

        # Log summary
        total_count = len(self.notebooks_found)
        success_count = len(self.successful_notebooks)
        fail_count = len(self.failed_notebooks)

        logger.info("=" * 60)
        logger.info("Test run complete")
        logger.info(f"{success_count}/{total_count} notebook(s) passed")

        if self.failed_notebooks:
            failed_names = [nb.name for nb in self.failed_notebooks]
            logger.error(f"{fail_count} notebook(s) failed: {', '.join(failed_names)}")

            # Raise assertion error to fail the test
            raise AssertionError(
                f"{fail_count} notebook(s) failed to execute: {', '.join(failed_names)}"
            )
        else:
            logger.success("All notebooks passed!")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the test run.

        Returns:
            Dictionary containing test run metadata:
                - total_notebooks: Total number of notebooks found
                - successful_notebooks: Number of successful executions
                - failed_notebooks: Number of failed executions
                - failed_notebook_names: List of failed notebook names
                - notebook_directory: Path to notebook directory
                - results_directory: Path to results directory
        """
        return {
            "total_notebooks": len(self.notebooks_found),
            "successful_notebooks": len(self.successful_notebooks),
            "failed_notebooks": len(self.failed_notebooks),
            "failed_notebook_names": [nb.name for nb in self.failed_notebooks],
            "notebook_directory": str(self.notebook_directory),
            "model_directory": str(self.model_dir),
        }


# Pytest test functions
@pytest.mark.slow
def test_interactive_context_notebooks() -> None:
    """Test all notebooks in the interactive directory."""
    # Skip test if not on SLURM
    if not IS_ON_SLURM:
        pytest.skip("Test skipped: must be run on SLURM cluster")

    runner = NotebookTestRunner(
        notebook_directory=MODEL_NOTEBOOKS_DIR / "interactive",
        environment_type="simulation",
    )
    runner.test_run_notebooks()


@pytest.mark.slow
def test_results_notebooks() -> None:
    """Test all notebooks in the results directory."""
    # Skip test if not on SLURM
    if not IS_ON_SLURM:
        pytest.skip("Test skipped: must be run on SLURM cluster")

    runner = NotebookTestRunner(
        notebook_directory=MODEL_NOTEBOOKS_DIR / "results",
        environment_type="artifact",
    )
    runner.test_run_notebooks()


@pytest.mark.slow
@pytest.mark.skip(reason="No notebooks currently in artifact directory")
def test_artifact_notebooks() -> None:
    """
    Test notebooks in the artifact directory.
    
    This test requires --results-dir to be specified.
    It will skip if the argument is not provided.
    
    Usage:
        pytest tests/model_notebooks/test_notebooks.py::test_artifact_notebooks \
            --results-dir=path/to/results
    """
    # Skip test if not on SLURM
    if not IS_ON_SLURM:
        pytest.skip("Test skipped: must be run on SLURM cluster")

    runner = NotebookTestRunner(
        notebook_directory=MODEL_NOTEBOOKS_DIR / "artifact",
        environment_type="artifact",
    )
    runner.test_run_notebooks()
