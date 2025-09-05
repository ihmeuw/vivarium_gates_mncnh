from functools import partial

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import SimulantData

from vivarium_gates_mncnh.constants.data_values import (
    ANC_ATTENDANCE_TYPES,
    COLUMNS,
    DELIVERY_FACILITY_TYPES,
    INTERVENTION_TYPE_MAPPER,
    INTERVENTIONS,
    PREGNANCY_OUTCOMES,
)
from vivarium_gates_mncnh.constants.metadata import PRETERM_AGE_CUTOFF
from vivarium_gates_mncnh.constants.scenarios import INTERVENTION_SCENARIOS

INTERVENTION_TYPE_COLUMN_MAP = {
    "neonatal": [
        COLUMNS.DELIVERY_FACILITY_TYPE,
        COLUMNS.GESTATIONAL_AGE_EXPOSURE,
        COLUMNS.PREGNANCY_OUTCOME,
    ],
    "maternal": [
        COLUMNS.DELIVERY_FACILITY_TYPE,
        COLUMNS.MOTHER_AGE,
        COLUMNS.ANC_ATTENDANCE,
    ],
}
INTERVENTION_SCENARIO_ACCESS_MAP = {
    "full": 1.0,
    "scale_up": 0.5,
}


class InterventionAccess(Component):
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
        return INTERVENTION_TYPE_COLUMN_MAP[self.intervention_type]

    def __init__(self, intervention: str) -> None:
        super().__init__()
        self.intervention = intervention
        self.intervention_column = f"{self.intervention}_available"
        self.time_step = f"{self.intervention}_access"
        self.intervention_type = INTERVENTION_TYPE_MAPPER[self.intervention]

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
        pop = self.filter_pop_for_intervention(pop)

        for (
            facility_type,
            coverage_value,
        ) in self.coverage_values.items():
            facility_idx = pop.index[pop[COLUMNS.DELIVERY_FACILITY_TYPE] == facility_type]
            coverage_value = (
                coverage_value
                if isinstance(coverage_value, float)
                else coverage_value(facility_idx)
            )
            get_intervention_idx = self.randomness.filter_for_probability(
                facility_idx,
                coverage_value,
                f"{self.intervention}_access_{facility_type}",
            )
            pop.loc[get_intervention_idx, self.intervention_column] = True

        self.population_view.update(pop)

    def get_coverage_values(self) -> dict[str, float]:
        delivery_facility_access_probabilities = {
            DELIVERY_FACILITY_TYPES.BEmONC: self.lookup_tables["bemonc_access_probability"],
            DELIVERY_FACILITY_TYPES.CEmONC: self.lookup_tables["cemonc_access_probability"],
            DELIVERY_FACILITY_TYPES.HOME: self.lookup_tables["home_access_probability"],
        }
        bemonc_scenario = getattr(
            self.scenario, f"bemonc_{self.intervention}_access", "baseline"
        )
        cemonc_scenario = getattr(
            self.scenario, f"cemonc_{self.intervention}_access", "baseline"
        )
        # As of model 9.0, misoprostol is the only intervention where we have a home scale up
        home_scenario = getattr(self.scenario, f"home_{self.intervention}_access", "baseline")
        bemonc_intervention_access = (
            INTERVENTION_SCENARIO_ACCESS_MAP[bemonc_scenario]
            if bemonc_scenario != "baseline"
            else delivery_facility_access_probabilities[DELIVERY_FACILITY_TYPES.BEmONC]
        )
        cemonc_intervention_access = (
            INTERVENTION_SCENARIO_ACCESS_MAP[cemonc_scenario]
            if cemonc_scenario != "baseline"
            else delivery_facility_access_probabilities[DELIVERY_FACILITY_TYPES.CEmONC]
        )
        home_intervention_access = (
            INTERVENTION_SCENARIO_ACCESS_MAP[home_scenario]
            if home_scenario != "baseline"
            else delivery_facility_access_probabilities[DELIVERY_FACILITY_TYPES.HOME]
        )
        return {
            DELIVERY_FACILITY_TYPES.BEmONC: bemonc_intervention_access,
            DELIVERY_FACILITY_TYPES.CEmONC: cemonc_intervention_access,
            DELIVERY_FACILITY_TYPES.HOME: home_intervention_access,
        }

    def load_coverage_data(self, builder: Builder, key: str) -> LookupTable:
        data = builder.data.load(
            f"intervention.no_{self.intervention}_risk.probability_{self.intervention}_{key}"
        )
        return data

    def filter_pop_for_intervention(self, pop: pd.DataFrame) -> pd.DataFrame:
        # Only live births are considered for neonatal interventions
        if self.intervention_type == "neonatal":
            pop = pop.loc[
                pop[COLUMNS.PREGNANCY_OUTCOME] == PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME
            ]
        # If intervention is probiotics, filter for preterm births
        if self.intervention == INTERVENTIONS.PROBIOTICS:
            pop = pop.loc[pop[COLUMNS.GESTATIONAL_AGE_EXPOSURE] < PRETERM_AGE_CUTOFF]
        # Misoprostol is only available to mothers who attended ANC and gave birth at home
        if self.intervention == INTERVENTIONS.MISOPROSTOL:
            pop = pop.loc[
                (pop[COLUMNS.ANC_ATTENDANCE] != ANC_ATTENDANCE_TYPES.NONE)  # attended ANC
                & (pop[COLUMNS.DELIVERY_FACILITY_TYPE] == DELIVERY_FACILITY_TYPES.HOME)
            ]
        return pop


class ACSAccess(Component):
    """Component for determining if a simulant has access to antenatal corticosteroids (ACS).
    We do this by making ACS available to everyone who has access to CPAP and a predicted
    gestational age between 26 and 33 weeks."""

    @property
    def columns_created(self) -> list[str]:
        return [COLUMNS.ACS_AVAILABLE]

    @property
    def columns_required(self) -> list[str]:
        return [
            COLUMNS.CPAP_AVAILABLE,
            COLUMNS.STATED_GESTATIONAL_AGE,
            COLUMNS.PREGNANCY_OUTCOME,
        ]

    def __init__(self) -> None:
        super().__init__()
        self.time_step = "acs_access"

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        simulants = pd.DataFrame(
            {
                COLUMNS.ACS_AVAILABLE: False,
            },
            index=pop_data.index,
        )
        self.population_view.update(simulants)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != self.time_step:
            return

        pop = self.population_view.get(event.index)

        pop = pop.loc[pop[COLUMNS.PREGNANCY_OUTCOME] == PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME]
        is_early_or_moderate_preterm = pop.loc[
            pop[COLUMNS.STATED_GESTATIONAL_AGE].between(26, 33)
        ]
        has_cpap = pop.loc[pop[COLUMNS.CPAP_AVAILABLE] == True]
        has_acs = is_early_or_moderate_preterm.index.intersection(has_cpap.index)
        pop.loc[has_acs, COLUMNS.ACS_AVAILABLE] = True

        self.population_view.update(pop)
