import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_gates_mncnh.constants.data_keys import NO_CPAP_RISK
from vivarium_gates_mncnh.constants.data_values import COLUMNS, DELIVERY_FACILITY_TYPES
from vivarium_gates_mncnh.constants.scenarios import INTERVENTION_SCENARIOS


class NeonatalInterventionAccess(Component):
    """Component for determining if a simulant has access to neonatal interventions."""

    @property
    def configuration_defaults(self) -> dict:
        return {
            self.name: {
                "data_sources": {
                    "bemonc_access_probability": f"intervention.no_{self.intervention}_risk.probability_{self.intervention}_bemonc",
                    "cemonc_access_probability": f"intervention.no_{self.intervention}_risk.probability_{self.intervention}_cemonc",
                }
            }
        }

    @property
    def columns_created(self) -> list[str]:
        return [self.intervention_column]

    @property
    def columns_required(self) -> list[str]:
        return [COLUMNS.DELIVERY_FACILITY_TYPE]

    def __init__(self, intervention: str) -> None:
        super().__init__()
        self.intervention = intervention
        self.intervention_column = f"{self.intervention}_available"
        self.time_step = f"{self.intervention}_access"

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.scenario = INTERVENTION_SCENARIOS[builder.configuration.intervention.scenario]
        self.delivery_facility_access_probabilities = (
            self.get_delivery_facility_access_probabilities()
        )
        self.coverage_values = self.get_coverage_values()

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        simulants = pd.DataFrame(
            {
                self.intervention_column: False,
            },
            index=pop_data.index,
        )
        self.population_view.update(simulants)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != self.time_step:
            return

        pop = self.population_view.get(event.index)

        for facility_type in self.delivery_facility_access_probabilities:
            facility_idx = pop.index[pop[COLUMNS.DELIVERY_FACILITY_TYPE] == facility_type]
            get_intervention_idx = self.randomness.filter_for_probability(
                facility_idx,
                self.coverage_values[facility_type](facility_idx),
                f"intervention_access_for_{facility_type}",
            )
            pop.loc[get_intervention_idx, self.intervention_column] = True

        self.population_view.update(pop)

    def get_delivery_facility_access_probabilities(self) -> dict[str, float]:
        return {
            DELIVERY_FACILITY_TYPES.BEmONC: self.lookup_tables["bemonc_access_probability"],
            DELIVERY_FACILITY_TYPES.CEmONC: self.lookup_tables["cemonc_access_probability"],
        }

    def get_coverage_values(self) -> dict[str, float]:
        bemonc_scenario = getattr(self.scenario, f"bemonc_{self.intervention}_access")
        cemonc_scenario = getattr(self.scenario, f"cemonc_{self.intervention}_access")
        bemonc_cpap_access = (
            1.0
            if bemonc_scenario == "full"
            else self.delivery_facility_access_probabilities[DELIVERY_FACILITY_TYPES.BEmONC]
        )
        cemonc_cpap_access = (
            1.0
            if cemonc_scenario == "full"
            else self.delivery_facility_access_probabilities[DELIVERY_FACILITY_TYPES.CEmONC]
        )
        return {
            DELIVERY_FACILITY_TYPES.BEmONC: bemonc_cpap_access,
            DELIVERY_FACILITY_TYPES.CEmONC: cemonc_cpap_access,
        }
