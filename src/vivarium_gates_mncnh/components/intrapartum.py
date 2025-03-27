import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_gates_mncnh.constants.data_keys import NO_CPAP_RISK
from vivarium_gates_mncnh.constants.data_values import (
    COLUMNS,
    DELIVERY_FACILITY_TYPES,
    SIMULATION_EVENT_NAMES,
)
from vivarium_gates_mncnh.constants.scenarios import INTERVENTION_SCENARIOS
from vivarium_gates_mncnh.utilities import get_location


class CPAPAccess(Component):
    """Component for determining if a simulant has access to CPAP."""

    @property
    def columns_created(self) -> list[str]:
        return [COLUMNS.CPAP_AVAILABLE]

    @property
    def columns_required(self) -> list[str]:
        return [COLUMNS.DELIVERY_FACILITY_TYPE]

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.location = get_location(builder)
        self.scenario = INTERVENTION_SCENARIOS[builder.configuration.intervention.scenario]
        self.coverage_values = self.get_coverage_values(builder)

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        anc_data = pd.DataFrame(
            {
                COLUMNS.CPAP_AVAILABLE: False,
            },
            index=pop_data.index,
        )
        self.population_view.update(anc_data)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.CPAP_ACCESS:
            return

        pop = self.population_view.get(event.index)

        for (
            facility_type,
            coverage_value,
        ) in self.coverage_values.items():
            facility_idx = pop.index[pop[COLUMNS.DELIVERY_FACILITY_TYPE] == facility_type]
            cpap_access_idx = self.randomness.filter_for_probability(
                facility_idx,
                coverage_value,
                f"cpap_access_{facility_type}",
            )
            pop.loc[cpap_access_idx, COLUMNS.CPAP_AVAILABLE] = True

        self.population_view.update(pop)

    def get_coverage_values(self, builder: Builder) -> dict[str, float]:
        delivery_facility_access_probabilities = {
            DELIVERY_FACILITY_TYPES.BEmONC: builder.data.load(NO_CPAP_RISK.P_CPAP_BEmONC),
            DELIVERY_FACILITY_TYPES.CEmONC: builder.data.load(NO_CPAP_RISK.P_CPAP_CEmONC),
        }
        bemonc_cpap_access = (
            1.0
            if self.scenario.bemonc_cpap_access == "full"
            else delivery_facility_access_probabilities[DELIVERY_FACILITY_TYPES.BEmONC]
        )
        cemonc_cpap_access = (
            1.0
            if self.scenario.cemonc_cpap_access == "full"
            else delivery_facility_access_probabilities[DELIVERY_FACILITY_TYPES.CEmONC]
        )
        return {
            DELIVERY_FACILITY_TYPES.BEmONC: bemonc_cpap_access,
            DELIVERY_FACILITY_TYPES.CEmONC: cemonc_cpap_access,
        }
