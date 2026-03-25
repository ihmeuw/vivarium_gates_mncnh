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
        self.bemonc_access_probability = self.build_lookup_table(
            builder, "bemonc_access_probability"
        )
        self.cemonc_access_probability = self.build_lookup_table(
            builder, "cemonc_access_probability"
        )
        self.home_access_probability = self.build_lookup_table(
            builder, "home_access_probability"
        )
        self.coverage_values = self.get_coverage_values()
        builder.population.register_initializer(
            self.initialize_intervention_access,
            columns=[self.intervention_column],
            required_resources=INTERVENTION_TYPE_COLUMN_MAP[self.intervention_type],
        )

    def initialize_intervention_access(self, pop_data: SimulantData) -> None:
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

        required_cols = INTERVENTION_TYPE_COLUMN_MAP[self.intervention_type]
        attrs = self.population_view.get_attributes(event.index, required_cols)
        eligible_idx = self._get_eligible_index(attrs)

        for (
            facility_type,
            coverage_value,
        ) in self.coverage_values.items():
            facility_idx = attrs.loc[eligible_idx].index[
                attrs.loc[eligible_idx, COLUMNS.DELIVERY_FACILITY_TYPE] == facility_type
            ]
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
            intervention_col = pd.Series(
                True, index=get_intervention_idx, name=self.intervention_column
            )
            self.population_view.update(intervention_col)

    def get_coverage_values(self) -> dict[str, float]:
        delivery_facility_access_probabilities = {
            DELIVERY_FACILITY_TYPES.BEmONC: self.bemonc_access_probability,
            DELIVERY_FACILITY_TYPES.CEmONC: self.cemonc_access_probability,
            DELIVERY_FACILITY_TYPES.HOME: self.home_access_probability,
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

    def _get_eligible_index(self, attrs: pd.DataFrame) -> pd.Index:
        """Return the index of simulants eligible for this intervention."""
        eligible = attrs
        # Only live births are considered for neonatal interventions
        if self.intervention_type == "neonatal":
            eligible = eligible.loc[
                eligible[COLUMNS.PREGNANCY_OUTCOME] == PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME
            ]
        # If intervention is probiotics, filter for preterm births
        if self.intervention == INTERVENTIONS.PROBIOTICS:
            eligible = eligible.loc[
                eligible[COLUMNS.GESTATIONAL_AGE_EXPOSURE] < PRETERM_AGE_CUTOFF
            ]
        # Misoprostol is only available to mothers who attended ANC and gave birth at home
        if self.intervention == INTERVENTIONS.MISOPROSTOL:
            eligible = eligible.loc[
                (eligible[COLUMNS.ANC_ATTENDANCE] != ANC_ATTENDANCE_TYPES.NONE)
                & (eligible[COLUMNS.DELIVERY_FACILITY_TYPE] == DELIVERY_FACILITY_TYPES.HOME)
            ]
        return eligible.index


class ACSAccess(Component):
    """Component for determining if a simulant has access to antenatal corticosteroids (ACS).
    We do this by making ACS available to everyone who has access to CPAP and a predicted
    gestational age between 26 and 33 weeks."""

    def __init__(self) -> None:
        super().__init__()
        self.time_step = "acs_access"

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.scenario = INTERVENTION_SCENARIOS[builder.configuration.intervention.scenario]
        builder.population.register_initializer(
            self.initialize_acs_access,
            columns=[COLUMNS.ACS_AVAILABLE],
            required_resources=[
                COLUMNS.CPAP_AVAILABLE,
                COLUMNS.STATED_GESTATIONAL_AGE,
                COLUMNS.PREGNANCY_OUTCOME,
            ],
        )

    def initialize_acs_access(self, pop_data: SimulantData) -> None:
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
        if self.scenario.acs_access == "none":
            return

        attrs = self.population_view.get_attributes(
            event.index,
            [
                COLUMNS.PREGNANCY_OUTCOME,
                COLUMNS.STATED_GESTATIONAL_AGE,
                COLUMNS.CPAP_AVAILABLE,
            ],
        )
        live_births = attrs.loc[
            attrs[COLUMNS.PREGNANCY_OUTCOME] == PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME
        ]
        is_early_or_moderate_preterm = live_births.loc[
            live_births[COLUMNS.STATED_GESTATIONAL_AGE].between(26, 33)
        ]
        has_cpap = live_births.loc[live_births[COLUMNS.CPAP_AVAILABLE] == True]
        has_acs = is_early_or_moderate_preterm.index.intersection(has_cpap.index)
        acs_update = pd.Series(True, index=has_acs, name=COLUMNS.ACS_AVAILABLE)
        self.population_view.update(acs_update)
