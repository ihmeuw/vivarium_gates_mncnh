import pytest

from vivarium import InteractiveContext

from vivarium_gates_mncnh.constants.data_keys import NO_CPAP_INTERVENTION
from vivarium_gates_mncnh.constants.data_values import COLUMNS, DELIVERY_FACILITY_TYPES, DELIVERY_FACILITY_TYPE_PROBABILITIES, SIMULATION_EVENT_NAMES


@pytest.fixture(scope="module")
def intrapartum_state(simulation_states) -> InteractiveContext:
    return simulation_states[SIMULATION_EVENT_NAMES.INTRAPARTUM]


def test_delivery_facility_proportions(intrapartum_state) -> None:
    pop = intrapartum_state.get_population()
    location = pop[COLUMNS.LOCATION].unique()[0]
    facility_type_mapper = {
        DELIVERY_FACILITY_TYPES.HOME: NO_CPAP_INTERVENTION.P_HOME,
        DELIVERY_FACILITY_TYPES.CEmONC: NO_CPAP_INTERVENTION.P_CEmONC,
        DELIVERY_FACILITY_TYPES.BEmONC: NO_CPAP_INTERVENTION.P_BEmONC,
    }
    for facility_type, probability_name in facility_type_mapper:
        facility_proportion = (pop[COLUMNS.DELIVERY_FACILITY_TYPE] == facility_type) / len(pop)
        breakpoint()
        assert facility_proportion == DELIVERY_FACILITY_TYPE_PROBABILITIES[location][probability_name]
    