from __future__ import annotations

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium_public_health.utilities import get_lookup_columns

from vivarium_gates_mncnh.constants.data_values import COLUMNS, INTERVENTIONS, PIPELINES


class InterventionRiskEffect(Component):
    """Component that modifies a neonatal CSMR pipeline based on the lack of an intervention.
    This is the implementation of the RiskEffect for these dichoctomous risks."""

    INTERVENTION_PIPELINE_MODIFIERS_MAP = {
        # TODO: add neonatal interventions below here
        INTERVENTIONS.ANTIBIOTICS: PIPELINES.NEONATAL_SEPSIS_FINAL_CSMR,
        INTERVENTIONS.PROBIOTICS: PIPELINES.NEONATAL_SEPSIS_FINAL_CSMR,
        # TODO: add maternal interventions below here
        INTERVENTIONS.AZITHROMYCIN: PIPELINES.MATERNAL_SEPSIS_INCIDENCE_RISK,
        INTERVENTIONS.MISOPROSTOL: PIPELINES.MATERNAL_HEMORRHAGE_INCIDENCE_RISK,
    }

    @property
    def configuration_defaults(self) -> dict:
        return {
            self.name: {
                "data_sources": {
                    "relative_risk": f"intervention.no_{self.lack_of_intervention_risk}_risk.relative_risk",
                    "paf": f"intervention.no_{self.lack_of_intervention_risk}_risk.population_attributable_fraction",
                }
            }
        }

    @property
    def columns_required(self) -> list[str]:
        return [self.col_required]

    @property
    def target_pipeline_name(self) -> str:
        return self.INTERVENTION_PIPELINE_MODIFIERS_MAP[self.lack_of_intervention_risk]

    def __init__(
        self,
        lack_of_intervention_risk: str,
    ) -> None:
        super().__init__()
        self.lack_of_intervention_risk = lack_of_intervention_risk
        self.col_required = f"{lack_of_intervention_risk}_available"

    def setup(self, builder: Builder) -> None:
        self.randomness = builder.randomness.get_stream(self.name)
        builder.value.register_value_modifier(
            self.target_pipeline_name,
            self.modify_target_pipeline,
            component=self,
            required_resources=[self.col_required]
            + get_lookup_columns(
                [self.lookup_tables["paf"], self.lookup_tables["relative_risk"]]
            ),
        )

    ##################
    # Helper nethods #
    ##################

    def modify_target_pipeline(
        self, index: pd.Index, target_pipeline: pd.Series[float]
    ) -> pd.Series[float]:
        # TODO: rename and update names
        # No intervention access is like a dichotomous risk factor, meaning those that have access to CPAP will
        # not have their CSMR modify by no intervention RR
        pop = self.population_view.get(index)
        no_intervention_idx = pop.index[pop[self.col_required] == False]
        # NOTE: RR is relative risk for no intervention
        no_intervention_rr = self.lookup_tables["relative_risk"](no_intervention_idx)
        # NOTE: PAF is for no intervention
        paf = self.lookup_tables["paf"](index)

        # Modify the pipeline
        modified_pipeline = target_pipeline * (1 - paf)
        modified_pipeline.loc[no_intervention_idx] = modified_pipeline * no_intervention_rr
        return modified_pipeline


class CPAPACSRiskEffect(Component):
    """Component that modifies a neonatal CSMR pipeline based on the lack of an ACS intervention.
    This component is unique because we need to distinguish between simulants who have CPAP/ACS and CPAP/no_ACS.
    There are no simulants who receive ACS but not CPAP."""

    @property
    def configuration_defaults(self) -> dict:
        return {
            self.name: {
                "data_sources": {
                    "no_cpap_relative_risk": f"intervention.no_cpap_risk.relative_risk",
                    "no_cpap_paf": f"intervention.no_cpap_risk.population_attributable_fraction",
                    "no_acs_relative_risk": f"intervention.no_acs_risk.relative_risk",
                    "no_acs_paf": f"intervention.no_acs_risk.population_attributable_fraction",
                }
            }
        }

    @property
    def columns_required(self) -> list[str]:
        return self.cols_required

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.cols_required = ["cpap_available", "acs_available"]

    def setup(self, builder: Builder) -> None:
        self.randomness = builder.randomness.get_stream(self.name)
        builder.value.register_value_modifier(
            PIPELINES.PRETERM_WITH_RDS_FINAL_CSMR,
            self.modify_target_pipeline,
            component=self,
            required_resources=self.cols_required
            + get_lookup_columns(
                [
                    self.lookup_tables["no_cpap_paf"],
                    self.lookup_tables["no_cpap_relative_risk"],
                    self.lookup_tables["no_acs_paf"],
                    self.lookup_tables["no_acs_relative_risk"],
                ]
            ),
        )

    ##################
    # Helper nethods #
    ##################

    def modify_target_pipeline(
        self, index: pd.Index, target_pipeline: pd.Series[float]
    ) -> pd.Series[float]:
        pop = self.population_view.get(index)
        modified_pipeline = target_pipeline.copy()

        # no CPAP
        no_cpap_idx = pop.index[(pop[COLUMNS.CPAP_AVAILABLE] == False)]
        # NOTE: RR is relative risk for no intervention
        no_cpap_rr = self.lookup_tables["no_cpap_relative_risk"](no_cpap_idx)
        # NOTE: PAF is for no intervention
        no_cpap_paf = self.lookup_tables["no_cpap_paf"](no_cpap_idx)

        modified_pipeline.loc[no_cpap_idx] = modified_pipeline.loc[no_cpap_idx] * (
            1 - no_cpap_paf
        )
        modified_pipeline.loc[no_cpap_idx] = modified_pipeline.loc[no_cpap_idx] * no_cpap_rr

        # no ACS
        no_acs_idx = pop.index[
            (pop[COLUMNS.CPAP_AVAILABLE] == True) & (pop[COLUMNS.ACS_AVAILABLE] == False)
        ]
        # NOTE: RR is relative risk for no intervention
        no_acs_rr = self.lookup_tables["no_acs_relative_risk"](no_acs_idx)
        # NOTE: PAF is for no intervention
        no_acs_paf = self.lookup_tables["no_acs_paf"](no_acs_idx)

        modified_pipeline.loc[no_acs_idx] = modified_pipeline.loc[no_acs_idx] * (
            1 - no_acs_paf
        )
        modified_pipeline.loc[no_acs_idx] = modified_pipeline.loc[no_acs_idx] * no_acs_rr
        breakpoint()
        # Modify the pipeline
        return modified_pipeline
