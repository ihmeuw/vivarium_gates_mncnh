from functools import partial

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import SimulantData

from vivarium_gates_mncnh.constants.data_values import (
    CHILD_LOOKUP_COLUMN_MAPPER,
    COLUMNS,
    DELIVERY_FACILITY_TYPES,
)
from vivarium_gates_mncnh.constants.scenarios import INTERVENTION_SCENARIOS


class NeonatalInterventionAccess(Component):
    """Component for determining if a simulant has access to neonatal interventions."""

    @property
    def configuration_defaults(self) -> dict:
        return {
            self.name: {
                "data_sources": {
                    "bemonc_access_probability": partial(
                        self.load_coverage_data, key="bemonc"
                    ),
                    "cemonc_access_probability": partial(
                        self.load_coverage_data, key="cemonc"
                    ),
                    "home_access_probability": partial(self.load_coverage_data, key="home"),
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

        for (
            facility_type,
            coverage_value,
        ) in self.coverage_values.items():
            facility_idx = pop.index[pop[COLUMNS.DELIVERY_FACILITY_TYPE] == facility_type]
            get_intervention_idx = self.randomness.filter_for_probability(
                facility_idx,
                coverage_value(facility_idx),
                f"cpap_access_{facility_type}",
            )
            pop.loc[get_intervention_idx, self.intervention_column] = True

        self.population_view.update(pop)

    def get_coverage_values(self) -> dict[str, float]:
        delivery_facility_access_probabilities = {
            DELIVERY_FACILITY_TYPES.BEmONC: self.lookup_tables["bemonc_access_probability"],
            DELIVERY_FACILITY_TYPES.CEmONC: self.lookup_tables["cemonc_access_probability"],
            DELIVERY_FACILITY_TYPES.HOME: self.lookup_tables["home_access_probability"],
        }
        bemonc_scenario = getattr(self.scenario, f"bemonc_{self.intervention}_access")
        cemonc_scenario = getattr(self.scenario, f"cemonc_{self.intervention}_access")
        bemonc_intervention_access = (
            1.0
            if bemonc_scenario == "full"
            else delivery_facility_access_probabilities[DELIVERY_FACILITY_TYPES.BEmONC]
        )
        cemonc_intervention_access = (
            1.0
            if cemonc_scenario == "full"
            else delivery_facility_access_probabilities[DELIVERY_FACILITY_TYPES.CEmONC]
        )
        home_intervention_access = delivery_facility_access_probabilities[
            DELIVERY_FACILITY_TYPES.HOME
        ]

        return {
            DELIVERY_FACILITY_TYPES.BEmONC: bemonc_intervention_access,
            DELIVERY_FACILITY_TYPES.CEmONC: cemonc_intervention_access,
            DELIVERY_FACILITY_TYPES.HOME: home_intervention_access,
        }

    def load_coverage_data(self, builder: Builder, key: str) -> LookupTable:
        data = builder.data.load(
            f"intervention.no_{self.intervention}_risk.probability_{self.intervention}_{key}"
        )
        if isinstance(data, pd.DataFrame):
            data = data.rename(columns=CHILD_LOOKUP_COLUMN_MAPPER)

        return data
