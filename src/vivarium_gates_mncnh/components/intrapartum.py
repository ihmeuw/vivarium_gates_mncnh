import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium_gates_mncnh.constants.data_values import (
    COLUMNS,
    DELIVERY_FACILITY_TYPES,
    DELIVERY_FACILITY_TYPE_PROBABILITIES,
    SIMULATION_EVENT_NAMES,
)
from vivarium_gates_mncnh.utilities import get_location


class Intrapartum(Component):
    @property
    def on_time_Step_priority(self) -> int:
        # We want this to happen first so delivery facility type is chosen
        return 2
    
    @property
    def columns_created(self) -> list[str]:
        return [
            COLUMNS.DELIVERY_FACILITY_TYPE,
        ]

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.location = get_location(builder)

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        anc_data = pd.DataFrame(
            {
                COLUMNS.DELIVERY_FACILITY_TYPE: DELIVERY_FACILITY_TYPES.NONE
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
            list(DELIVERY_FACILITY_TYPE_PROBABILITIES[self.location].keys()),
            p=list(DELIVERY_FACILITY_TYPE_PROBABILITIES[self.location].values()),
            additional_key="delivery_facility_type",    
        )
        pop[COLUMNS.DELIVERY_FACILITY_TYPE] = delivery_facility_type

        self.population_view.update(pop)
        