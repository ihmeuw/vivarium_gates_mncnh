import pytest
from vivarium import InteractiveContext
from vivarium_testing_utils import FuzzyChecker

from vivarium_gates_mncnh.constants.data_keys import NO_CPAP_INTERVENTION
from vivarium_gates_mncnh.constants.data_values import (
    COLUMNS,
    CPAP_ACCESS_PROBABILITIES,
    DELIVERY_FACILITY_TYPE_PROBABILITIES,
    DELIVERY_FACILITY_TYPES,
    SIMULATION_EVENT_NAMES,
)


@pytest.fixture(scope="module")
def intrapartum_state(simulation_states) -> InteractiveContext:
    return simulation_states[SIMULATION_EVENT_NAMES.INTRAPARTUM]


@pytest.mark.parametrize(
    "facility_type",
    [
        DELIVERY_FACILITY_TYPES.HOME,
        DELIVERY_FACILITY_TYPES.CEmONC,
        DELIVERY_FACILITY_TYPES.BEmONC,
    ],
)
def test_delivery_facility_proportions(
    facility_type: str, fuzzy_checker: FuzzyChecker, intrapartum_state
) -> None:
    pop = intrapartum_state.get_population()
    location = pop[COLUMNS.LOCATION].unique()[0]
    facility_type_mapper = {
        DELIVERY_FACILITY_TYPES.HOME: NO_CPAP_INTERVENTION.P_HOME,
        DELIVERY_FACILITY_TYPES.CEmONC: NO_CPAP_INTERVENTION.P_CEmONC,
        DELIVERY_FACILITY_TYPES.BEmONC: NO_CPAP_INTERVENTION.P_BEmONC,
    }
    fuzzy_checker.fuzzy_assert_proportion(
        (pop[COLUMNS.DELIVERY_FACILITY_TYPE] == facility_type).sum(),
        len(pop),
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
    facility_type: str, fuzzy_checker: FuzzyChecker, intrapartum_state
) -> None:
    pop = intrapartum_state.get_population()
    location = pop[COLUMNS.LOCATION].unique()[0]
    has_cpap_idx = pop.index[pop[COLUMNS.CPAP_AVAILABLE] == True]
    facility_idx = pop.index[pop[COLUMNS.DELIVERY_FACILITY_TYPE] == facility_type]
    facility_type_mapper = {
        DELIVERY_FACILITY_TYPES.HOME: NO_CPAP_INTERVENTION.P_CPAP_HOME,
        DELIVERY_FACILITY_TYPES.CEmONC: NO_CPAP_INTERVENTION.P_CPAP_CEmONC,
        DELIVERY_FACILITY_TYPES.BEmONC: NO_CPAP_INTERVENTION.P_CPAP_BEmONC,
    }
    fuzzy_checker.fuzzy_assert_proportion(
        len(has_cpap_idx.intersection(facility_idx)),
        len(facility_idx),
        CPAP_ACCESS_PROBABILITIES[location][facility_type_mapper[facility_type]],
        name=f"cpap_availability_{facility_type}_proportion",
    )
