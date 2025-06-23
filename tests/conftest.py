import os
from pathlib import Path
from typing import Any, Generator

import pytest
from layered_config_tree import LayeredConfigTree
from vivarium import Artifact
from vivarium_testing_utils import FuzzyChecker

from vivarium_gates_mncnh.constants import paths
from vivarium_gates_mncnh.constants.data_values import SIMULATION_EVENT_NAMES

SIMULATION_STEPS = [
    SIMULATION_EVENT_NAMES.PREGNANCY,
    SIMULATION_EVENT_NAMES.DELIVERY_FACILITY,
    SIMULATION_EVENT_NAMES.AZITHROMYCIN_ACCESS,
    SIMULATION_EVENT_NAMES.MISOPROSTOL_ACCESS,
    SIMULATION_EVENT_NAMES.CPAP_ACCESS,
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


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
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
