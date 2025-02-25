from pathlib import Path
import pytest

from vivarium import InteractiveContext

from vivarium_gates_mncnh.constants.data_values import SIMULATION_EVENT_NAMES
from vivarium_gates_mncnh.constants import paths


SIMULATION_STEPS = [
    SIMULATION_EVENT_NAMES.PREGNANCY,
    SIMULATION_EVENT_NAMES.INTRAPARTUM,
    SIMULATION_EVENT_NAMES.MATERNAL_SEPSIS,
    SIMULATION_EVENT_NAMES.MATERNAL_HEMORRHAGE,
    SIMULATION_EVENT_NAMES.OBSTRUCTED_LABOR,
    SIMULATION_EVENT_NAMES.MORTALITY,
    SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY,
    SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY,
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
def simulation_states(model_spec_path: Path) -> dict[str, InteractiveContext]:
    # Set up interactive sim
    sim = InteractiveContext(model_spec_path)
    sim_states = {}
    for step in SIMULATION_STEPS:
        sim_states[step] = sim
        # The simulation starts on the pregnancy time step
        if step == SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY:
            continue
        sim.step()
    
    return sim_states
