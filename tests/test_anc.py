from pathlib import Path

import pandas as pd
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

from .utilities import get_interactive_context_state

pytest.skip(
    allow_module_level=True,
    reason="Model 14 ANC and model 15 facility choice updates have obsoleted these tests.",
)


@pytest.fixture(scope="module")
def anc_state(
    sim_state_step_mapper: dict[str, int], model_spec_path: Path
) -> InteractiveContext:
    sim = InteractiveContext(model_spec_path)
    return get_interactive_context_state(
        sim, sim_state_step_mapper, SIMULATION_EVENT_NAMES.PREGNANCY
    )


@pytest.fixture(scope="module")
def population(anc_state: InteractiveContext) -> pd.DataFrame:
    return anc_state.get_population()


@pytest.fixture(scope="module")
def attended_anc_facility_proportion(
    anc_state: InteractiveContext, artifact: Artifact
) -> float:
    draw = f"draw_{anc_state.model_specification.configuration.input_data.input_draw_number}"
    # This is loading a one row dataframe we want to get the value depending on the draw
    attended_facility_proportion = artifact.load(ANC.ANC1)[draw].iloc[0]
    return attended_facility_proportion


def test_attended_care_facility_proportions(
    attended_anc_facility_proportion: float,
    population: pd.DataFrame,
    fuzzy_checker: FuzzyChecker,
) -> None:
    fuzzy_checker.fuzzy_assert_proportion(
        (population[COLUMNS.ATTENDED_CARE_FACILITY] == True).sum(),
        len(population),
        attended_anc_facility_proportion,
        name="attended_care_facility_proportion",
    )


def test_received_ultrasound_proportions(
    attended_anc_facility_proportion: float,
    population: pd.DataFrame,
    fuzzy_checker: FuzzyChecker,
) -> None:
    location = population[COLUMNS.LOCATION].iloc[0]
    ultrasound_proportion = ANC_RATES.RECEIVED_ULTRASOUND[location]
    breakpoint()
    fuzzy_checker.fuzzy_assert_proportion(
        (population[COLUMNS.ULTRASOUND_TYPE] != ULTRASOUND_TYPES.NO_ULTRASOUND).sum(),
        len(population),
        ultrasound_proportion * attended_anc_facility_proportion,
        name="received_ultrasound_proportion",
    )


@pytest.mark.parametrize(
    "ultrasound_type", [ULTRASOUND_TYPES.STANDARD, ULTRASOUND_TYPES.AI_ASSISTED]
)
def test_ultrasound_type_proportions(
    ultrasound_type: str,
    population: pd.DataFrame,
    fuzzy_checker: FuzzyChecker,
) -> None:
    location = population[COLUMNS.LOCATION].iloc[0]
    ultrasound_type_proportion = ANC_RATES.ULTRASOUND_TYPE[location][ultrasound_type]
    fuzzy_checker.fuzzy_assert_proportion(
        (population[COLUMNS.ULTRASOUND_TYPE] == ultrasound_type).sum(),
        (population[COLUMNS.ULTRASOUND_TYPE] != ULTRASOUND_TYPES.NO_ULTRASOUND).sum(),
        ultrasound_type_proportion,
        name=f"ultrasound_type_proportion_{ultrasound_type}",
    )
