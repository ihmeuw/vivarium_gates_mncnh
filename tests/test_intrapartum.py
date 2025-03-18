from pathlib import Path

import pandas as pd
import pytest
from vivarium import InteractiveContext
from vivarium_testing_utils import FuzzyChecker

from vivarium_gates_mncnh.constants.data_keys import NO_CPAP_RISK
from vivarium_gates_mncnh.constants.data_values import (
    COLUMNS,
    CPAP_ACCESS_PROBABILITIES,
    DELIVERY_FACILITY_TYPE_PROBABILITIES,
    DELIVERY_FACILITY_TYPES,
    SIMULATION_EVENT_NAMES,
)

from .utilities import get_interactive_context_state


@pytest.fixture(scope="module")
def intrapartum_state(
    model_spec_path: Path, sim_state_step_mapper: dict[str, int]
) -> InteractiveContext:
    sim = InteractiveContext(model_spec_path)
    return get_interactive_context_state(
        sim, sim_state_step_mapper, SIMULATION_EVENT_NAMES.CPAP_ACCESS
    )


@pytest.fixture(scope="module")
def population(intrapartum_state: InteractiveContext) -> pd.DataFrame:
    return intrapartum_state.get_population()


@pytest.mark.parametrize(
    "facility_type",
    [
        DELIVERY_FACILITY_TYPES.HOME,
        DELIVERY_FACILITY_TYPES.CEmONC,
        DELIVERY_FACILITY_TYPES.BEmONC,
    ],
)
def test_delivery_facility_proportions(
    facility_type: str,
    fuzzy_checker: FuzzyChecker,
    population: pd.DataFrame,
) -> None:
    location = population[COLUMNS.LOCATION].unique()[0]
    facility_type_mapper = {
        DELIVERY_FACILITY_TYPES.HOME: NO_CPAP_RISK.P_HOME,
        DELIVERY_FACILITY_TYPES.CEmONC: NO_CPAP_RISK.P_CEmONC,
        DELIVERY_FACILITY_TYPES.BEmONC: NO_CPAP_RISK.P_BEmONC,
    }
    birth_idx = population.index[population[COLUMNS.PREGNANCY_OUTCOME] != "partial_term"]
    fuzzy_checker.fuzzy_assert_proportion(
        (population.loc[birth_idx, COLUMNS.DELIVERY_FACILITY_TYPE] == facility_type).sum(),
        len(birth_idx),
        DELIVERY_FACILITY_TYPE_PROBABILITIES[location][facility_type_mapper[facility_type]],
        name=f"facility_type_{facility_type}_proportion",
    )


@pytest.mark.parametrize(
    "facility_type",
    [
        DELIVERY_FACILITY_TYPES.HOME,
        DELIVERY_FACILITY_TYPES.CEmONC,
        DELIVERY_FACILITY_TYPES.BEmONC,
    ],
)
def test_cpap_availability(
    facility_type: str,
    fuzzy_checker: FuzzyChecker,
    population: pd.DataFrame,
) -> None:
    location = population[COLUMNS.LOCATION].unique()[0]
    has_cpap_idx = population.index[population[COLUMNS.CPAP_AVAILABLE] == True]
    facility_idx = population.index[
        population[COLUMNS.DELIVERY_FACILITY_TYPE] == facility_type
    ]
    facility_type_mapper = {
        DELIVERY_FACILITY_TYPES.HOME: NO_CPAP_RISK.P_CPAP_HOME,
        DELIVERY_FACILITY_TYPES.CEmONC: NO_CPAP_RISK.P_CPAP_CEmONC,
        DELIVERY_FACILITY_TYPES.BEmONC: NO_CPAP_RISK.P_CPAP_BEmONC,
    }
    fuzzy_checker.fuzzy_assert_proportion(
        len(has_cpap_idx.intersection(facility_idx)),
        len(facility_idx),
        CPAP_ACCESS_PROBABILITIES[location][facility_type_mapper[facility_type]],
        name=f"cpap_availability_{facility_type}_proportion",
    )
