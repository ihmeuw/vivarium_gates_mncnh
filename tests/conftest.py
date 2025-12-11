import os
import shutil
from pathlib import Path
from typing import Any, Generator

import pandas as pd
import pytest
from layered_config_tree import LayeredConfigTree
from pytest import TempPathFactory
from vivarium import Artifact
from vivarium_testing_utils import FuzzyChecker
from vivarium_testing_utils.automated_validation.constants import DRAW_INDEX, SEED_INDEX

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


def _create_births_observer_data() -> pd.DataFrame:
    """Create births observer data for testing."""
    return pd.DataFrame(
        {
            "value": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 1.0, 2.0] * 2,
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("births", pd.NA, pd.NA, pd.NA, "Male", "live_birth", "A", 0, 0),
                ("births", pd.NA, pd.NA, pd.NA, "Male", "live_birth", "A", 0, 1),
                ("births", pd.NA, pd.NA, pd.NA, "Male", "live_birth", "A", 1, 0),
                ("births", pd.NA, pd.NA, pd.NA, "Male", "live_birth", "A", 1, 1),
                ("births", pd.NA, pd.NA, pd.NA, "Male", "live_birth", "B", 0, 0),
                ("births", pd.NA, pd.NA, pd.NA, "Male", "live_birth", "B", 0, 1),
                ("births", pd.NA, pd.NA, pd.NA, "Male", "live_birth", "B", 1, 0),
                ("births", pd.NA, pd.NA, pd.NA, "Male", "live_birth", "B", 1, 1),
                ("births", pd.NA, pd.NA, pd.NA, "Female", "live_birth", "A", 0, 0),
                ("births", pd.NA, pd.NA, pd.NA, "Female", "live_birth", "A", 0, 1),
                ("births", pd.NA, pd.NA, pd.NA, "Female", "live_birth", "A", 1, 0),
                ("births", pd.NA, pd.NA, pd.NA, "Female", "live_birth", "A", 1, 1),
                ("births", pd.NA, pd.NA, pd.NA, "Female", "live_birth", "B", 0, 0),
                ("births", pd.NA, pd.NA, pd.NA, "Female", "live_birth", "B", 0, 1),
                ("births", pd.NA, pd.NA, pd.NA, "Female", "live_birth", "B", 1, 0),
                ("births", pd.NA, pd.NA, pd.NA, "Female", "live_birth", "B", 1, 1),
                ("births", pd.NA, pd.NA, pd.NA, "Male", "still_birth", "A", 0, 0),
                ("births", pd.NA, pd.NA, pd.NA, "Male", "still_birth", "A", 0, 1),
                ("births", pd.NA, pd.NA, pd.NA, "Female", "still_birth", "A", 1, 0),
                ("births", pd.NA, pd.NA, pd.NA, "Female", "still_birth", "A", 1, 1),
            ],
            names=[
                "measure",
                "entity_type",
                "entity",
                "sub_entity",
                "child_sex",
                "pregnancy_outcome",
                "common_stratify_column",
                DRAW_INDEX,
                SEED_INDEX,
            ],
        ),
    )


@pytest.fixture
def get_births_observer_data() -> pd.DataFrame:
    """Get births observer data for testing."""
    return _create_births_observer_data()


@pytest.fixture(scope="session")
def mncnh_results_dir(tmp_path_factory: TempPathFactory) -> Path:
    """Create a temporary directory for simulation outputs."""
    # Create the temporary directory at session scope
    tmp_path = tmp_path_factory.mktemp("mncnh_data")

    # Create the directory structure
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)

    # Create data directly within this session-scoped fixture
    # so we don't depend on function-scoped fixtures
    _births = _create_births_observer_data()

    # Save Sim DataFrames
    _births.reset_index().to_parquet(results_dir / "births.parquet")

    # TODO: MIC-6667. Create Artifact when updating measures to include artifact data
    # artifact_dir = tmp_path / "artifacts"
    # artifact_dir.mkdir(exist_ok=True)
    # artifact_path = artifact_dir / "artifact.hdf"
    # artifact = Artifact(artifact_path)
    # for key, data in _artifact_keys_mapper.items():
    #     artifact.write(key, data)

    # # Save model specification
    # with open(tmp_path / "model_specification.yaml", "w") as f:
    #     yaml.dump(get_model_spec(artifact_path), f)

    return tmp_path
