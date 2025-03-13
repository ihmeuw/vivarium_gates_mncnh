from functools import partial

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
        self.lbwsg_acmr_paf = self.get_paf(builder)
        required_pipeline_resources = (
            [self.lbwsg_acmr_paf] if isinstance(self.lbwsg_acmr_paf, Pipeline) else []
        )
        # Register csmr pipeline
        self.intermediate_csmr = builder.value.register_value_producer(
            f"{self.neonatal_cause}.cause_specific_mortality_rate",
            source=self.get_normalized_csmr,
            component=self,
            required_resources=required_pipeline_resources,
        )
        self.final_csmr = builder.value.register_value_producer(
            f"{self.neonatal_cause}.csmr",
            source=self.intermediate_csmr,
            component=self,
        )

        builder.value.register_value_modifier(
            PIPELINES.DEATH_IN_AGE_GROUP_PROBABILITY,
            modifier=self.modify_death_in_age_group_probability,
            component=self,
            required_resources=required_pipeline_resources,
        )
        # Create CSMR PAF pipeline which will do nothing but is needed for the LBWSGRiskEffect
        builder.value.register_value_producer(
            f"{self.neonatal_cause}.cause_specific_mortality_rate.paf",
            source=builder.lookup.build_table(0),
            component=self,
        )

    ##################
    # Helper methods #
    ##################

    def get_paf(self, builder: Builder) -> Pipeline:
        return builder.value.get_value(PIPELINES.LBWSG_ACMR_PAF_MODIFIER)

    def load_csmr(self, builder: Builder) -> pd.DataFrame:
        csmr = builder.data.load(f"cause.{self.neonatal_cause}.cause_specific_mortality_rate")
        csmr = csmr.rename(columns=CHILD_LOOKUP_COLUMN_MAPPER)
        return csmr

    def get_normalized_csmr(self, index: pd.Index) -> pd.Series:
        # CSMR = CSMR * (1-PAF) * RR
        # NOTE: There is LBWSG RR on this pipeline
        raw_csmr = self.lookup_tables["csmr"](index)
        normalizing_constant = 1 - self.lbwsg_acmr_paf(index)
        normalized_csmr = raw_csmr * normalizing_constant

        return normalized_csmr

    def modify_death_in_age_group_probability(
        self, index: pd.Index, probability_death_in_age_group: pd.Series
    ) -> pd.Series:
        intermediate_csmr = self.intermediate_csmr(index)
        final_csmr = self.final_csmr(index)
        # ACMR = ACMR - CSMR + CSMR
        modified_acmr = probability_death_in_age_group - intermediate_csmr + final_csmr

        return modified_acmr


class PretermBirth(NeonatalCause):
    @property
    def configuration_defaults(self) -> dict:
        return {
            self.name: {
                "data_sources": {
                    key: partial(self.load_lookup_data, key=key)
                    for key in ["csmr", "paf", "prevalence"]
                }
            }
        }

    def get_paf(self, _: Builder) -> LookupTable:
        return self.lookup_tables["paf"]

    def get_normalized_csmr(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get(index)
        ga_greater_than_37 = pop[COLUMNS.GESTATIONAL_AGE] >= 37

        # CSMR = (1 - PAF) * RR * (CSMR / PRETERM_PREVALENCE)
        # NOTE: This isn't technically a traditional PAF but it is the
        # PAF for the preterm population. We are accounting for this by
        # dividing the CSMR by the prevalence of the preterm categories
        raw_csmr = self.lookup_tables["csmr"](index)
        normalizing_constant = 1 - self.lbwsg_acmr_paf(index)
        prevalence = self.lookup_tables["prevalence"](index)
        normalized_csmr = normalizing_constant * (raw_csmr / prevalence)

        # Set CSMR to 0 for those who are not preterm
        normalized_csmr.loc[ga_greater_than_37] = 0
        # Weight csmr for preterm birth with rds
        if self.neonatal_cause == NEONATAL_CAUSES.PRETERM_BIRTH_WITH_RDS:
            normalized_csmr = normalized_csmr * PRETERM_DEATHS_DUE_TO_RDS_PROBABILITY
        else:
            normalized_csmr = normalized_csmr * (1 - PRETERM_DEATHS_DUE_TO_RDS_PROBABILITY)
        return normalized_csmr

    def load_lookup_data(self, builder: Builder, key: str) -> pd.DataFrame:
        # Hard codes preterm csmr key since it is the same for both preterm subcauses
        key_mapper = {
            "csmr": data_keys.PRETERM_BIRTH.CSMR,
            "paf": data_keys.PRETERM_BIRTH.PAF,
            "prevalence": data_keys.PRETERM_BIRTH.PREVALENCE,
        }
        art_key = key_mapper[key]
        data = builder.data.load(art_key)
        data = data.rename(columns=CHILD_LOOKUP_COLUMN_MAPPER)
        return data
