from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event

from vivarium_gates_mncnh.constants.data_values import (
    COLUMNS,
    DELIVERY_FACILITY_TYPE_PROBABILITIES,
    DELIVERY_FACILITY_TYPES,
    PREGNANCY_OUTCOMES,
    SIMULATION_EVENT_NAMES,
)
from vivarium_gates_mncnh.utilities import get_location


class DeliveryFacility(Component):
    @property
    def columns_required(self) -> list[str]:
        return [COLUMNS.DELIVERY_FACILITY_TYPE, COLUMNS.PREGNANCY_OUTCOME]

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.location = get_location(builder)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.DELIVERY_FACILITY:
            return

        pop = self.population_view.get(event.index)
        pop[COLUMNS.DELIVERY_FACILITY_TYPE] = DELIVERY_FACILITY_TYPES.NONE

        # Choose delivery facility type
        birth_idx = pop.index[
            pop[COLUMNS.PREGNANCY_OUTCOME] != PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME
        ]
        delivery_facility_type = self.randomness.choice(
            birth_idx,
            [
                DELIVERY_FACILITY_TYPES.HOME,
                DELIVERY_FACILITY_TYPES.CEmONC,
                DELIVERY_FACILITY_TYPES.BEmONC,
            ],
            p=list(DELIVERY_FACILITY_TYPE_PROBABILITIES[self.location].values()),
            additional_key="delivery_facility_type",
        )
        pop.loc[birth_idx, COLUMNS.DELIVERY_FACILITY_TYPE] = delivery_facility_type

        self.population_view.update(pop)
