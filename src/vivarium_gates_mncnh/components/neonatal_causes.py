import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.resource import Resource

from vivarium_gates_mncnh.constants.data_values import (
    CHILD_LOOKUP_COLUMN_MAPPER,
    COLUMNS,
    NEONATAL_CAUSES,
    PIPELINES,
)


class NeonatalCause(Component):
    @property
    def columns_required(self) -> list[str]:
        return [COLUMNS.GESTATIONAL_AGE]

    # @property
    # def initialization_requirements(
    #     self,
    # ) -> list[str | Resource]:
    #     return [PIPELINES.ACMR_PAF, PIPELINES.DEATH_IN_AGE_GROUP_PROBABILITY]

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
        self.paf = builder.value.get_value(PIPELINES.ACMR_PAF)
        # Register csmr pipeline
        self.csmr = builder.value.register_value_producer(
            f"cause.{self.neonatal_cause}.cause_specific_mnoortality_rate",
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
        pop = self.population_view.get(index)
        # CSMR = CSMR * (1-PAF) * RR
        # NOTE: There is LBWSG RR on this pipeline
        raw_csmr = self.lookup_tables["csmr"]
        normalizing_constant = 1 - self.paf(index)
        normalized_csmr = raw_csmr * normalizing_constant
        # Account for structural zeros in preterm birth
        if self.neonatal_cause == NEONATAL_CAUSES.PRETERM_BIRTH:
            ga_less_than_37 = pop[COLUMNS.GESTATIONAL_AGE] < 37
            normalized_csmr.loc[ga_less_than_37] = 0
        return normalized_csmr

    def modify_death_in_age_group_probability(
        self, index: pd.Index, probability_death_in_age_group: pd.Series
    ) -> pd.Series:
        csmr_pipeline = self.csmr(index)
        # ACMR = ACMR - CSMR + CSMR
        modified_acmr = probability_death_in_age_group - csmr_pipeline + csmr_pipeline
        return modified_acmr
