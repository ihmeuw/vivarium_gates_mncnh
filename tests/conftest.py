import os
import shutil
from pathlib import Path
from typing import Any, Generator

import pytest
from layered_config_tree import LayeredConfigTree
from vivarium import Artifact
from vivarium_testing_utils import FuzzyChecker

from vivarium_gates_mncnh.constants import paths
from vivarium_gates_mncnh.constants.data_values import SIMULATION_EVENT_NAMES

SIMULATION_STEPS = [
    SIMULATION_EVENT_NAMES.FIRST_TRIMESTER_ANC,
    SIMULATION_EVENT_NAMES.LATER_PREGNANCY_SCREENING,
    SIMULATION_EVENT_NAMES.LATER_PREGNANCY_INTERVENTION,
    SIMULATION_EVENT_NAMES.ULTRASOUND,
    SIMULATION_EVENT_NAMES.DELIVERY_FACILITY,
    SIMULATION_EVENT_NAMES.AZITHROMYCIN_ACCESS,
    SIMULATION_EVENT_NAMES.MISOPROSTOL_ACCESS,
    SIMULATION_EVENT_NAMES.CPAP_ACCESS,
    SIMULATION_EVENT_NAMES.ACS_ACCESS,
    SIMULATION_EVENT_NAMES.ANTIBIOTICS_ACCESS,
    SIMULATION_EVENT_NAMES.PROBIOTICS_ACCESS,
    SIMULATION_EVENT_NAMES.MATERNAL_SEPSIS,
    SIMULATION_EVENT_NAMES.MATERNAL_HEMORRHAGE,
    SIMULATION_EVENT_NAMES.OBSTRUCTED_LABOR,
    SIMULATION_EVENT_NAMES.MORTALITY,
    SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY,
    SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY,
    SIMULATION_EVENT_NAMES.POSTPARTUM_DEPRESSION,
]


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    # Notebook testing options
    parser.addoption(
        "--notebook-dir",
        action="store",
        default=None,
        help="Directory containing notebooks to test",
    )
    parser.addoption(
        "--results-dir",
        action="store",
        default=None,
        help="Directory for results (passed as parameter to notebooks)",
    )
    parser.addoption(
        "--notebook-kernel",
        action="store",
        default=None,
        help="Kernel name to use for notebook execution",
    )
    parser.addoption(
        "--notebook-timeout",
        action="store",
        type=int,
        default=300,
        help="Timeout in seconds for notebook execution (default: 300)",
    )
    parser.addoption(
        "--keep-notebooks",
        action="store_true",
        default=False,
        help="Keep executed notebooks instead of cleaning them up",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_jenkins = pytest.mark.skip(reason="skipping tests in jenkins")
    is_on_jenkins = os.environ.get("JENKINS_URL")

    if is_on_jenkins:
        for item in items:
            item.add_marker(skip_jenkins)

    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return

    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def model_spec_path() -> Path:
    repo_path = paths.BASE_DIR
    config_path = repo_path / "model_specifications" / "model_spec.yaml"
    return config_path


@pytest.fixture(scope="session")
def sim_state_step_mapper() -> dict[str, int]:
    step_mapper = {}
    for i in range(len(SIMULATION_STEPS)):
        step_mapper[SIMULATION_STEPS[i]] = i + 1
    return step_mapper


@pytest.fixture(scope="session")
def artifact(model_spec_path) -> Artifact:
    config = LayeredConfigTree(model_spec_path)
    artifact = Artifact(config.configuration.input_data.artifact_path)
    return artifact


@pytest.fixture(scope="session")
def output_directory() -> str:
    v_v_path = Path(os.path.dirname(__file__)) / "v_and_v_output"
    return v_v_path


@pytest.fixture(scope="session")
def fuzzy_checker(output_directory) -> Generator[FuzzyChecker, Any, Any]:
    checker = FuzzyChecker()

    yield checker

    checker.save_diagnostic_output(output_directory)


def is_on_slurm() -> bool:
    """Returns True if the current environment is a SLURM cluster."""
    return shutil.which("sbatch") is not None


IS_ON_SLURM = is_on_slurm()


@pytest.fixture
def notebook_config(request):
    """Fixture to provide notebook testing configuration from CLI args."""
    return {
        "notebook_directory": request.config.getoption("--notebook-dir"),
        "results_dir": request.config.getoption("--results-dir"),
        "kernel_name": request.config.getoption("--notebook-kernel"),
        "timeout": request.config.getoption("--notebook-timeout"),
        "cleanup_notebooks": not request.config.getoption("--keep-notebooks"),
    }
