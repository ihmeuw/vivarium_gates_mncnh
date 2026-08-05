from __future__ import annotations

from functools import partial

import pandas as pd
from vivarium.engine import Component
from vivarium.engine.framework.engine import Builder
from vivarium.engine.framework.event import Event
from vivarium.engine.framework.lookup import LookupTable
from vivarium.engine.framework.population import SimulantData

from vivarium_gates_mncnh.constants.data_values import (
    ACS_ELIGIBLE_GESTATIONAL_AGE_RANGE,
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
    "none": 0.0,
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
    def access_columns(self) -> list[str]:
        """State table columns needed to assign this intervention's coverage."""
        return INTERVENTION_TYPE_COLUMN_MAP[self.intervention_type]

    @property
    def coverage_intervention(self) -> str:
        """Intervention whose baseline coverage data this component loads."""
        return self.intervention

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
        self.bemonc_access_probability_table = self.build_lookup_table(
            builder, "bemonc_access_probability"
        )
        self.cemonc_access_probability_table = self.build_lookup_table(
            builder, "cemonc_access_probability"
        )
        self.home_access_probability_table = self.build_lookup_table(
            builder, "home_access_probability"
        )
        self.coverage_values = self.get_coverage_values()
        builder.population.register_initializer(
            self.initialize_intervention_access,
            columns=[self.intervention_column],
        )

    def initialize_intervention_access(self, pop_data: SimulantData) -> None:
        simulants = pd.DataFrame(
            {self.intervention_column: False},
            index=pop_data.index,
        )
        self.population_view.initialize(simulants)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != self.time_step:
            return

        pop = self.population_view.get(event.index, self.access_columns)
        pop = self.filter_pop_for_intervention(pop)

        has_intervention = pd.Series(False, index=pop.index, name=self.intervention_column)
        for facility_type, coverage_value in self.coverage_values.items():
            facility_idx = pop.index[pop[COLUMNS.DELIVERY_FACILITY_TYPE] == facility_type]
            effective_coverage = (
                coverage_value
                if isinstance(coverage_value, float)
                else coverage_value(facility_idx)
            )
            get_intervention_idx = self.select_covered_simulants(
                pop, facility_idx, effective_coverage, facility_type
            )
            has_intervention.loc[get_intervention_idx] = True

        self.population_view.update(
            self.intervention_column,
            lambda _: has_intervention,
        )

    def get_coverage_values(self) -> dict[str, float | LookupTable]:
        delivery_facility_access_probabilities = {
            DELIVERY_FACILITY_TYPES.BEmONC: self.bemonc_access_probability_table,
            DELIVERY_FACILITY_TYPES.CEmONC: self.cemonc_access_probability_table,
            DELIVERY_FACILITY_TYPES.HOME: self.home_access_probability_table,
        }
        bemonc_scenario = getattr(
            self.scenario, f"bemonc_{self.intervention}_access", "baseline"
        )
        cemonc_scenario = getattr(
            self.scenario, f"cemonc_{self.intervention}_access", "baseline"
        )
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

    def select_covered_simulants(
        self,
        pop: pd.DataFrame,
        facility_index: pd.Index,
        coverage: float | pd.Series[float],
        facility_type: str,
    ) -> pd.Index:
        """Simulants delivering at this facility type who receive the intervention."""
        return self.randomness.filter_for_probability(
            facility_index,
            coverage,
            f"{self.intervention}_access_{facility_type}",
        )

    def load_coverage_data(self, builder: Builder, key: str) -> LookupTable:
        intervention = self.coverage_intervention
        data = builder.data.load(
            f"intervention.no_{intervention}_risk.probability_{intervention}_{key}"
        )
        return data

    def filter_pop_for_intervention(self, pop: pd.DataFrame) -> pd.DataFrame:
        if self.intervention_type == "neonatal":
            pop = pop.loc[
                pop[COLUMNS.PREGNANCY_OUTCOME] == PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME
            ]
        if self.intervention == INTERVENTIONS.PROBIOTICS:
            pop = pop.loc[pop[COLUMNS.GESTATIONAL_AGE_EXPOSURE] < PRETERM_AGE_CUTOFF]
        if self.intervention == INTERVENTIONS.MISOPROSTOL:
            pop = pop.loc[
                (pop[COLUMNS.ANC_ATTENDANCE] != ANC_ATTENDANCE_TYPES.NONE)
                & (pop[COLUMNS.DELIVERY_FACILITY_TYPE] == DELIVERY_FACILITY_TYPES.HOME)
            ]
        return pop


class RDSInterventionAccess(InterventionAccess):
    @property
    def access_columns(self) -> list[str]:
        return super().access_columns + [COLUMNS.RDS_INTERVENTION_PROPENSITY]

    def select_covered_simulants(
        self,
        pop: pd.DataFrame,
        facility_index: pd.Index,
        coverage: float | pd.Series[float],
        facility_type: str,
    ) -> pd.Index:
        propensity = pop.loc[facility_index, COLUMNS.RDS_INTERVENTION_PROPENSITY]
        return propensity.index[propensity < coverage]


class ACSAccess(RDSInterventionAccess):
    """Component for determining if a simulant has access to antenatal corticosteroids (ACS)."""

    @property
    def access_columns(self) -> list[str]:
        return super().access_columns + [COLUMNS.STATED_GESTATIONAL_AGE]

    @property
    def coverage_intervention(self) -> str:
        # ACS has no coverage data of its own; baseline coverage in each delivery facility
        # is assumed equal to baseline CPAP coverage there.
        # https://vivarium-research.readthedocs.io/en/latest/models/concept_models/vivarium_mncnh_portfolio/intrapartum_interventions/module_document.html#baseline-coverage
        return INTERVENTIONS.CPAP

    def __init__(self) -> None:
        super().__init__(INTERVENTIONS.ACS)

    def filter_pop_for_intervention(self, pop: pd.DataFrame) -> pd.DataFrame:
        pop = super().filter_pop_for_intervention(pop)
        return pop.loc[
            pop[COLUMNS.STATED_GESTATIONAL_AGE].between(*ACS_ELIGIBLE_GESTATIONAL_AGE_RANGE)
        ]
