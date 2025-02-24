import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_gates_mncnh.constants.data_keys import NO_CPAP_INTERVENTION
from vivarium_gates_mncnh.constants.data_values import (
    COLUMNS,
    CPAP_ACCESS_PROBABILITIES,
    DELIVERY_FACILITY_TYPE_PROBABILITIES,
    DELIVERY_FACILITY_TYPES,
    SIMULATION_EVENT_NAMES,
)
from vivarium_gates_mncnh.utilities import get_location


class Intrapartum(Component):
    """This component functions to make key decisions about events in the intrapartum step of the model.
    Specifically, it determines the delivery facility type and whether or not CPAP is available to the simulant."""

    @property
    def columns_created(self) -> list[str]:
        return [
            COLUMNS.DELIVERY_FACILITY_TYPE,
            COLUMNS.CPAP_AVAILABLE,
        ]

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.location = get_location(builder)

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        anc_data = pd.DataFrame(
            {
                COLUMNS.DELIVERY_FACILITY_TYPE: DELIVERY_FACILITY_TYPES.NONE,
                COLUMNS.CPAP_AVAILABLE: False,
            },
            index=pop_data.index,
        )
        self.population_view.update(anc_data)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.INTRAPARTUM:
            return

        pop = self.population_view.get(event.index)
        # Choose delivery facility type
        delivery_facility_type = self.randomness.choice(
            pop.index,
            [
                DELIVERY_FACILITY_TYPES.NONE,
                DELIVERY_FACILITY_TYPES.CEmONC,
                DELIVERY_FACILITY_TYPES.BEmONC,
            ],
            p=list(DELIVERY_FACILITY_TYPE_PROBABILITIES[self.location].values()),
            additional_key="delivery_facility_type",
        )
        pop[COLUMNS.DELIVERY_FACILITY_TYPE] = delivery_facility_type

        # Determine if simulant had access to CPAP
        facility_type_mapper = {
            DELIVERY_FACILITY_TYPES.BEmONC: NO_CPAP_INTERVENTION.P_CPAP_BEmONC,
            DELIVERY_FACILITY_TYPES.CEmONC: NO_CPAP_INTERVENTION.P_CPAP_CEmONC,
        }
        for facility_type in [
            DELIVERY_FACILITY_TYPES.BEmONC,
            DELIVERY_FACILITY_TYPES.CEmONC,
        ]:
            facility_idx = pop.index[pop[COLUMNS.DELIVERY_FACILITY_TYPE] == facility_type]
            cpap_access_probability = CPAP_ACCESS_PROBABILITIES[self.location][
                facility_type_mapper[facility_type]
            ]
            cpap_access_idx = self.randomness.filter_for_probability(
                facility_idx, cpap_access_probability, f"cpap_access_{facility_type}"
            )
            pop.loc[cpap_access_idx, COLUMNS.CPAP_AVAILABLE] = True

        self.population_view.update(pop)
