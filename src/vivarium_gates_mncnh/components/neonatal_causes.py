import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.values import Pipeline

from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.constants.data_values import (
    CHILD_LOOKUP_COLUMN_MAPPER,
    COLUMNS,
    NEONATAL_CAUSES,
    PIPELINES,
    PRETERM_DEATHS_DUE_TO_RDS_PROBABILITY,
)


class NeonatalCause(Component):
    @property
    def columns_required(self) -> list[str]:
        return [COLUMNS.GESTATIONAL_AGE]

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

    def setup(self, builder: Builder) -> None:
        # This is the ACMR PAF pipeline. For preterm we will get the custom preterm PAF
        self.paf = self.get_paf(self, builder)
        # Register csmr pipeline
        self.csmr = builder.value.register_value_producer(
            f"cause.{self.neonatal_cause}.cause_specific_mortality_rate",
            source=self.get_normalized_csmr,
            component=self,
            required_resources=[self.paf],
        )
        builder.value.register_value_modifier(
            "death_in_age_group_probability",
            modifier=self.modify_death_in_age_group_probability,
            component=self,
            required_resources=[self.paf],
        )

    ##################
    # Helper methods #
    ##################

    def get_paf(self, builder: Builder) -> Pipeline:
        return builder.value.get_value(PIPELINES.ACMR_PAF)

    def load_csmr(self, builder: Builder) -> pd.DataFrame:
        csmr = builder.data.load(f"cause.{self.neonatal_cause}.cause_specific_mortality_rate")
        csmr = csmr.rename(columns=CHILD_LOOKUP_COLUMN_MAPPER)
        return csmr

    def get_normalized_csmr(self, index: pd.Index) -> pd.Series:
        # CSMR = CSMR * (1-PAF) * RR
        # NOTE: There is LBWSG RR on this pipeline
        raw_csmr = self.lookup_tables["csmr"](index)
        normalizing_constant = 1 - self.paf(index)
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
    def configuration_defaults(self) -> dict:
        return {
            self.name: {
                "data_sources": {
                    "csmr": self.load_csmr,
                    "paf": self.load_paf,
                }
            }
        }

    def load_paf(self, builder: Builder) -> pd.DataFrame:
        paf = builder.data.load(data_keys.PRETERM_BIRTH.PAF)
        paf = paf.rename(columns=CHILD_LOOKUP_COLUMN_MAPPER)
        return paf

    def get_paf(self, _: Builder) -> LookupTable:
        return self.lookup_tables["paf"]

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
