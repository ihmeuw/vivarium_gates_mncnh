import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.resource import Resource

from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.constants.data_values import (
    CHILD_LOOKUP_COLUMN_MAPPER,
    CPAP_ACCESS_PROBABILITIES,
    COLUMNS,
    DELIVERY_FACILITY_TYPES,
    NEONATAL_CAUSES,
    PIPELINES,
    PRETERM_DEATHS_DUE_TO_RDS_PROBABILITY,
    SIMULATION_EVENT_NAMES,
)
from vivarium_gates_mncnh.utilities import get_location


class NeonatalCause(Component):
    @property
    def columns_required(self) -> list[str]:
        return [COLUMNS.GESTATIONAL_AGE, COLUMNS.DELIVERY_FACILITY_TYPE]

    @property
    def configuration_defaults(self) -> dict:
        return {
            self.name: {
                "data_sources": {
                    "csmr": self.load_csmr,
                }
            }
        }

    def __init__(self, neonatal_cause: str) -> None:
        super().__init__()
        self.neonatal_cause = neonatal_cause

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder):
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.location = get_location(builder)
        self.acmr_paf = builder.value.get_value(PIPELINES.ACMR_PAF)
        # Register csmr pipeline
        self.csmr = builder.value.register_value_producer(
            f"cause.{self.neonatal_cause}.cause_specific_mortality_rate",
            source=self.get_normalized_csmr,
            component=self,
            required_resources=[PIPELINES.ACMR_PAF],
        )
        builder.value.register_value_modifier(
            "death_in_age_group_probability",
            modifier=self.modify_death_in_age_group_probability,
            component=self,
            required_resources=[PIPELINES.ACMR_PAF, PIPELINES.DEATH_IN_AGE_GROUP_PROBABILITY],
        )

    ##################
    # Helper methods #
    ##################

    def load_csmr(self, builder: Builder) -> pd.DataFrame:
        csmr = builder.data.load(f"cause.{self.neonatal_cause}.cause_specific_mortality_rate")
        csmr = csmr.rename(columns=CHILD_LOOKUP_COLUMN_MAPPER)
        return csmr

    def get_normalized_csmr(self, index: pd.Index) -> pd.Series:
        # CSMR = CSMR * (1-PAF) * RR
        # NOTE: There is LBWSG RR on this pipeline
        raw_csmr = self.lookup_tables["csmr"](index)
        normalizing_constant = 1 - self.acmr_paf(index)
        normalized_csmr = raw_csmr * normalizing_constant

        return normalized_csmr

    def modify_death_in_age_group_probability(
        self, index: pd.Index, probability_death_in_age_group: pd.Series
    ) -> pd.Series:
        csmr_pipeline = self.csmr(index)
        csmr_source = self.get_normalized_csmr(index)
        # ACMR = ACMR - CSMR + CSMR
        modified_acmr = probability_death_in_age_group - csmr_source + csmr_pipeline
        return modified_acmr


class PretermBirth(NeonatalCause):
    @property
    def columns_created(self) -> list[str]:
        return [
            COLUMNS.CPAP_AVAILABLE,
        ]
    
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        builder.value.register_value_modifier(
            self.csmr.name,
            self.calculate_cpap_path_probability,
            required_resources=[self.csmr.name, COLUMNS.DELIVERY_FACILITY_TYPE, COLUMNS.CPAP_AVAILABLE]
        )
    
    #####################
    # Lifecycle methods #
    #####################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop = pd.DataFrame(
            {COLUMNS.CPAP_AVAILABLE: False},
            index=pop_data.index,
        )
        self.population_view.update(pop)

    def on_time_step(self, event: Event) -> None:

        if self._sim_step_name() != SIMULATION_EVENT_NAMES.INTRAPARTUM:
            return

        pop = self.population_view.get(event.index)
        # Determine if simulant had access to CPAP
        for facility_type in [DELIVERY_FACILITY_TYPES.CLINIC, DELIVERY_FACILITY_TYPES.HOSPITAL]:
            cpap_access_probability = CPAP_ACCESS_PROBABILITIES[self.location][facility_type]
            cpap_access_idx = self.randomness.filter_for_probability(
                pop.index, cpap_access_probability, f"cpap_access_{facility_type}"
            )
            pop.loc[cpap_access_idx, COLUMNS.CPAP_AVAILABLE] = True

        self.population_view.update(pop)

    def get_normalized_csmr(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get(index)
        ga_greater_than_37 = pop[COLUMNS.GESTATIONAL_AGE] >= 37

        normalized_csmr = super().get_normalized_csmr(index)
        normalized_csmr.loc[ga_greater_than_37] = 0
        # Weight csmr for preterm birth with rds
        if self.neonatal_cause == NEONATAL_CAUSES.PRETERM_BIRTH_WITH_RDS:
            normalized_csmr = normalized_csmr * PRETERM_DEATHS_DUE_TO_RDS_PROBABILITY
        else:
            normalized_csmr = normalized_csmr * (1 - PRETERM_DEATHS_DUE_TO_RDS_PROBABILITY)
        return normalized_csmr

    def load_csmr(self, builder: Builder) -> pd.DataFrame:
        # Hard codes preterm csmr key since it is the same for both preterm subcauses
        csmr = builder.data.load(data_keys.PRETERM_BIRTH.CSMR)
        csmr = csmr.rename(columns=CHILD_LOOKUP_COLUMN_MAPPER)
        return csmr
    
    def calculate_cpap_path_probability(self, index: pd.Index, csmr: pd.Series) -> pd.Series:
        # TODO: implement
        pass
