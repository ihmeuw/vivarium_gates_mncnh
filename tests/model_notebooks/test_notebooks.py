"""
Script to test Jupyter notebooks in the model_notebooks directory. This script 
will run each notebook and record whether the notebook runs successfully or not. 
If the notebook has any cell that errors out, the test will fail.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import papermill as pm
import pytest
from loguru import logger

from tests.conftest import IS_ON_SLURM


class NotebookTestRunner:
    """
    A class to execute and test Jupyter notebooks using papermill.

    This runner discovers notebooks in a specified directory, executes them with
    papermill while injecting a results_dir parameter, and tracks success/failure
    of each notebook execution.

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
        results_dir: str,
        timeout: int = 300,
        cleanup_notebooks: bool = True,
    ):
        """
        Initialize the notebook test runner.

        Args:
            notebook_directory: Path to directory containing notebooks to test
            results_dir: Path to results directory (injected as parameter into notebooks)
            timeout: Maximum execution time per notebook in seconds (default: 300 = 5 minutes)
            cleanup_notebooks: Whether to delete executed notebooks after testing (default: True)

        Raises:
            FileNotFoundError: If notebook_directory does not exist
        """
        self.notebook_directory = Path(notebook_directory)
        self.results_dir = Path(results_dir)
        self.timeout = timeout
        self.cleanup_notebooks = cleanup_notebooks

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

        Args:
            notebook_path: Path to the notebook to execute

        Returns:
            True if execution succeeded, False if it failed
        """
        # Create output path with _executed suffix
        output_path = (
            notebook_path.parent / f"{notebook_path.stem}_executed{notebook_path.suffix}"
        )

        try:
            logger.info(f"Running notebook {notebook_path.name}...")

            # Execute notebook with papermill
            pm.execute_notebook(
                input_path=str(notebook_path),
                output_path=str(output_path),
                parameters={"results_dir": str(self.results_dir)},
                execution_timeout=self.timeout,
            )

            logger.success(f"✓ {notebook_path.name} completed successfully")

            # Clean up executed notebook if requested
            if self.cleanup_notebooks and output_path.exists():
                output_path.unlink()
                logger.debug(f"Cleaned up executed notebook: {output_path.name}")

            return True

        except Exception as e:
            logger.error(
                f"✗ {notebook_path.name} failed with error: {type(e).__name__}: {str(e)}"
            )
            self.failed_notebooks[notebook_path] = e

            # Clean up failed executed notebook if requested
            if self.cleanup_notebooks and output_path.exists():
                output_path.unlink()
                logger.debug(f"Cleaned up failed executed notebook: {output_path.name}")

            return False

    def test_run_notebooks(self) -> None:
        """
        Execute all discovered notebooks and track results.

        This is the main method that orchestrates the testing process:
        1. Discovers notebooks in the directory
        2. Executes each notebook with papermill
        3. Tracks successes and failures
        4. Logs summary of results
        5. Raises AssertionError if any notebooks failed

        Raises:
            AssertionError: If any notebooks failed to execute successfully
        """
        logger.info("Starting notebook test run")
        logger.info(f"Notebook directory: {self.notebook_directory}")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Timeout: {self.timeout} seconds")
        logger.info(f"Cleanup notebooks: {self.cleanup_notebooks}")

        # Discover notebooks
        self.notebooks_found = self._discover_notebook_paths()

        # Handle empty directory case
        if not self.notebooks_found:
            logger.info("No notebooks found")
            return

        # Execute each notebook
        for notebook_path in self.notebooks_found:
            success = self._execute_notebook(notebook_path)
            if success:
                self.successful_notebooks.append(notebook_path)

        # Log summary
        total_count = len(self.notebooks_found)
        success_count = len(self.successful_notebooks)
        fail_count = len(self.failed_notebooks)

        logger.info("=" * 60)
        logger.info("Test run complete")
        logger.info(f"{success_count}/{total_count} notebook(s) passed")

        if self.failed_notebooks:
            failed_names = [nb.name for nb in self.failed_notebooks.keys()]
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
            "failed_notebook_names": [nb.name for nb in self.failed_notebooks.keys()],
            "notebook_directory": str(self.notebook_directory),
            "results_directory": str(self.results_dir),
        }


# Pytest test functions
def test_interactive_notebooks(notebook_config: dict[str, Any]) -> None:
    """Test all notebooks in the interactive directory."""
    # Skip test if not on SLURM
    if not IS_ON_SLURM:
        pytest.skip("Test skipped: must be run on SLURM cluster")

    results_dir = notebook_config["results_dir"]

    # Skip test if results_dir not provided
    if results_dir is None:
        pytest.skip("Test skipped: use --results-dir to specify results directory")

    runner = NotebookTestRunner(
        notebook_directory="tests/model_notebooks/interactive",
        results_dir=results_dir,
        timeout=notebook_config["timeout"],
        cleanup_notebooks=notebook_config["cleanup_notebooks"],
    )
    runner.test_run_notebooks()


def test_results_notebooks(notebook_config: dict[str, Any]) -> None:
    """Test all notebooks in the results directory."""
    # Skip test if not on SLURM
    if not IS_ON_SLURM:
        pytest.skip("Test skipped: must be run on SLURM cluster")

    results_dir = notebook_config["results_dir"]

    # Skip test if results_dir not provided
    if results_dir is None:
        pytest.skip("Test skipped: use --results-dir to specify results directory")

    runner = NotebookTestRunner(
        notebook_directory="tests/model_notebooks/results",
        results_dir=results_dir,
        timeout=notebook_config["timeout"],
        cleanup_notebooks=notebook_config["cleanup_notebooks"],
    )
    runner.test_run_notebooks()


def test_artifact_notebooks(notebook_config: dict[str, Any]) -> None:
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

    results_dir = notebook_config["results_dir"]

    # Skip test if results_dir not provided
    if results_dir is None:
        pytest.skip("Test skipped: use --results-dir to specify results directory")

    runner = NotebookTestRunner(
        notebook_directory="tests/model_notebooks/artifact",
        results_dir=results_dir,
        timeout=notebook_config["timeout"],
        cleanup_notebooks=notebook_config["cleanup_notebooks"],
    )
    runner.test_run_notebooks()
