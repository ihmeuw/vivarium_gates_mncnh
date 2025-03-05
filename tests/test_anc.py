import pytest
from vivarium import Artifact, InteractiveContext
from vivarium_testing_utils import FuzzyChecker

from vivarium_gates_mncnh.constants.data_keys import ANC
from vivarium_gates_mncnh.constants.data_values import (
    ANC_RATES,
    COLUMNS,
    SIMULATION_EVENT_NAMES,
    ULTRASOUND_TYPES,
)


@pytest.fixture(scope="module")
def anc_state(simulation_states) -> InteractiveContext:
    return simulation_states[SIMULATION_EVENT_NAMES.PREGNANCY]


@pytest.fixture(scope="module")
def attended_anc_facility_proportion(
    anc_state: InteractiveContext, artifact: Artifact
) -> float:
    draw = f"draw_{anc_state.model_specification.configuration.input_data.input_draw_number}"
    # This is loading a one row dataframe we want to get the value depending on the draw
    attended_facility_proportion = artifact.load(ANC.ESTIMATE)[draw].iloc[0]
    return attended_facility_proportion


def test_attended_care_facility_proportions(
    attended_anc_facility_proportion: float,
    artifact: Artifact,
    fuzzy_checker: FuzzyChecker,
    anc_state,
) -> None:
    pop = anc_state.get_population()
    fuzzy_checker.fuzzy_assert_proportion(
        (pop[COLUMNS.ATTENDED_CARE_FACILITY] == True).sum(),
        len(pop),
        attended_anc_facility_proportion,
        name="attended_care_facility_proportion",
    )


def test_received_ultrasound_proportions(
    attended_anc_facility_proportion: float,
    artifact: Artifact,
    fuzzy_checker: FuzzyChecker,
    anc_state,
) -> None:
    pop = anc_state.get_population()
    location = pop[COLUMNS.LOCATION].iloc[0]
    ultrasound_proportion = ANC_RATES.RECEIVED_ULTRASOUND[location]
    fuzzy_checker.fuzzy_assert_proportion(
        (pop[COLUMNS.ULTRASOUND_TYPE] != ULTRASOUND_TYPES.NO_ULTRASOUND).sum(),
        len(pop),
        ultrasound_proportion * attended_anc_facility_proportion,
        name="received_ultrasound_proportion",
    )


@pytest.mark.parametrize(
    "ultrasound_type", [ULTRASOUND_TYPES.STANDARD, ULTRASOUND_TYPES.AI_ASSISTED]
)
def test_ultrasound_type_proportions(
    ultrasound_type: str, anc_state: InteractiveContext, fuzzy_checker: FuzzyChecker
) -> None:
    pop = anc_state.get_population()
    location = pop[COLUMNS.LOCATION].iloc[0]
    ultrasound_type_proportion = ANC_RATES.ULTRASOUND_TYPE[location][ultrasound_type]
    fuzzy_checker.fuzzy_assert_proportion(
        (pop[COLUMNS.ULTRASOUND_TYPE] == ultrasound_type).sum(),
        (pop[COLUMNS.ULTRASOUND_TYPE] != ULTRASOUND_TYPES.NO_ULTRASOUND).sum(),
        ultrasound_type_proportion,
        name=f"ultrasound_type_proportion_{ultrasound_type}",
    )
