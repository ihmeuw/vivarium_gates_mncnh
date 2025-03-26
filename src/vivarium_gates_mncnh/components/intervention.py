from __future__ import annotations

from functools import partial

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder

from vivarium_gates_mncnh.constants import data_values
from vivarium_gates_mncnh.constants.data_values import COLUMNS


# class NoCPAPRisk(Component):
#     """Component for CPAP intervention. This is essentially a risk effect."""

#     @property
#     def configuration_defaults(self) -> dict:
#         return {
#             self.name: {
#                 "data_sources": {
#                     "relative_risk": NO_CPAP_RISK.RELATIVE_RISK,
#                     "paf": self.load_paf_data,
#                 }
#             }
#         }

#     @property
#     def columns_required(self) -> list[str]:
#         return [COLUMNS.CPAP_AVAILABLE]

#     def __init__(self) -> None:
#         super().__init__()
#         self.preterm_csmr_target = PIPELINES.PRETERM_WITH_RDS_FINAL_CSMR

#     def setup(self, builder: Builder) -> None:
#         self.randomness = builder.randomness.get_stream(self.name)
#         self.location = get_location(builder)
#         self.preterm_with_rds_csmr = builder.value.get_value(self.preterm_csmr_target)
#         builder.value.register_value_modifier(
#             self.preterm_with_rds_csmr.name,
#             self.modify_preterm_with_rds_csmr,
#             required_resources=[
#                 COLUMNS.DELIVERY_FACILITY_TYPE,
#                 COLUMNS.CPAP_AVAILABLE,
#             ],
#         )

#     ##################
#     # Helper nethods #
#     ##################

#     def load_paf_data(self, builder: Builder) -> pd.Series:
#         data = builder.data.load(NO_CPAP_RISK.PAF)
#         data = data.rename(columns=data_values.CHILD_LOOKUP_COLUMN_MAPPER)
#         return data

#     def modify_preterm_with_rds_csmr(
#         self, index: pd.Index, preterm_with_rds_csmr: pd.Series[float]
#     ) -> pd.Series[float]:
#         # No CPAP access is like a dichotomous risk factor, meaning those that have access to CPAP will
#         # not have their CSMR modify by no CPAP RR
#         pop = self.population_view.get(index)
#         no_cpap_idx = pop.index[pop[COLUMNS.CPAP_AVAILABLE] == False]
#         # NOTE: RR is relative risk for no CPAP
#         no_cpap_rr = self.lookup_tables["relative_risk"](no_cpap_idx)
#         # NOTE: PAF is for no CPAP
#         paf = self.lookup_tables["paf"](index)

#         # Modify the CSMR pipeline
#         modified_csmr = preterm_with_rds_csmr * (1 - paf)
#         modified_csmr.loc[no_cpap_idx] = modified_csmr * no_cpap_rr
#         return modified_csmr


class NeonatalNoInterventionRisk(Component):
    """Component that modifies a neonatal CSMR pipeline based on the lack of an intervention. 
    This is the implementation of the RiskEffect for these dichoctomous risks."""

    @property
    def configuration_defaults(self) -> dict:
        return {
            self.name: {
                "data_sources": {
                    "relative_risk": self.load_relative_risk_data,
                    "paf": self.load_paf_data,
                }
            }
        }

    @property
    def columns_required(self) -> list[str]:
        return [self.col_required]

    def __init__(
            self, 
            lack_of_intervention_risk: str,
            column_required: str,
            target_pipeline: str
        ) -> None:
        super().__init__()
        self.lack_of_intervention_risk = lack_of_intervention_risk
        self.col_required = column_required
        self.csmr_target_pipeline_name = target_pipeline

    def setup(self, builder: Builder) -> None:
        self.randomness = builder.randomness.get_stream(self.name)
        self.csmr_target_pipeline = builder.value.get_value(self.csmr_target_pipeline_name)
        builder.value.register_value_modifier(
            self.csmr_target_pipeline,
            self.modify_preterm_with_rds_csmr,
            required_resources=[
                COLUMNS.DELIVERY_FACILITY_TYPE,
                self.col_required,
            ],
        )

    ##################
    # Helper nethods #
    ##################

    def load_relative_risk_data(self, builder: Builder) -> pd.DataFrame:
        data = builder.data.load(f"intervention.{self.lack_of_intervention_risk}.relative_risk")
        if isinstance(data, pd.DataFrame):
            data = data.rename(columns=data_values.CHILD_LOOKUP_COLUMN_MAPPER)
        return data

    def load_paf_data(self, builder: Builder) -> pd.DataFrame:
        data = builder.data.load(f"intervention.{self.lack_of_intervention_risk}.population_attributable_fraction")
        data = data.rename(columns=data_values.CHILD_LOOKUP_COLUMN_MAPPER)
        return data

    def modify_preterm_with_rds_csmr(
        self, index: pd.Index, preterm_with_rds_csmr: pd.Series[float]
    ) -> pd.Series[float]:
        # No CPAP access is like a dichotomous risk factor, meaning those that have access to CPAP will
        # not have their CSMR modify by no CPAP RR
        pop = self.population_view.get(index)
        no_intervention_idx = pop.index[pop[self.col_required] == False]
        # NOTE: RR is relative risk for no intervention
        no_intervention_rr = self.lookup_tables["relative_risk"](no_intervention_idx)
        # NOTE: PAF is for no intervention
        paf = self.lookup_tables["paf"](index)

        # Modify the CSMR pipeline
        modified_csmr = preterm_with_rds_csmr * (1 - paf)
        modified_csmr.loc[no_intervention_idx] = modified_csmr * no_intervention_rr
        return modified_csmr