from __future__ import annotations

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium_public_health.utilities import get_lookup_columns

from vivarium_gates_mncnh.constants import (
    data_keys,
    data_values,
    models,
)

from vivarium_gates_mncnh.constants.data_values import COLUMNS, INTERVENTIONS, PIPELINES
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RESIDUAL_CHOICE


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


class CPAPAndACSRiskEffect(Component):
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
        return [COLUMNS.CPAP_AVAILABLE, COLUMNS.ACS_AVAILABLE, COLUMNS.STATED_GESTATIONAL_AGE]

    @property
    def target_pipeline_name(self) -> str:
        return PIPELINES.PRETERM_WITH_RDS_FINAL_CSMR

    def setup(self, builder: Builder) -> None:
        self.randomness = builder.randomness.get_stream(self.name)
        builder.value.register_value_modifier(
            self.target_pipeline_name,
            self.modify_target_pipeline,
            component=self,
            required_resources=self.columns_required
            + get_lookup_columns(
                [
                    self.lookup_tables["no_cpap_relative_risk"],
                    self.lookup_tables["no_cpap_paf"],
                    self.lookup_tables["no_acs_relative_risk"],
                    self.lookup_tables["no_acs_paf"],
                ]
            ),
        )

    def modify_target_pipeline(
        self, index: pd.Index, target_pipeline: pd.Series[float]
    ) -> pd.Series[float]:
        """
        Modifies the preterm RDS CSMR pipeline based on CPAP and ACS access and ACS eligibility.

        Logic:
        - For simulants without CPAP and who are ACS-eligible (gestational age 26-33 weeks):
            Apply both no_CPAP_RR and no_ACS_RR.
        - For all ACS-eligible simulants:
            Apply no_ACS_PAF.
        - For simulants without CPAP and not ACS-eligible:
            Apply no_CPAP_RR.
        - For all simulants not ACS-eligible:
            Apply no_CPAP_PAF.
        """
        pop = self.population_view.get(index)
        in_acs_gestational_age_range = pop[COLUMNS.STATED_GESTATIONAL_AGE].between(26, 33)
        has_no_cpap = pop[COLUMNS.CPAP_AVAILABLE] == False

        no_intervention_index = has_no_cpap.index
        no_acs_index = pop.index[has_no_cpap & in_acs_gestational_age_range]
        no_cpap_index = pop.index[has_no_cpap & ~in_acs_gestational_age_range]

        no_intervention_rr = pd.Series(1.0, index=index)
        no_intervention_rr.loc[no_cpap_index] = self.lookup_tables["no_cpap_relative_risk"](
            no_cpap_index
        )
        no_intervention_rr.loc[no_acs_index] = self.lookup_tables["no_acs_relative_risk"](
            no_acs_index
        ) * self.lookup_tables["no_cpap_relative_risk"](no_acs_index)

        no_intervention_paf = self.lookup_tables["no_cpap_paf"](index)
        no_intervention_paf.loc[in_acs_gestational_age_range] = self.lookup_tables[
            "no_acs_paf"
        ](no_acs_index)

        target_pipeline.loc[no_intervention_index] = (
            target_pipeline.loc[no_intervention_index]
            * (1 - no_intervention_paf)
            * no_intervention_rr
        )
        return target_pipeline


class OralIronInterventions(Component):
    CONFIGURATION_DEFAULTS = {
        "intervention": {
            "scenario": "baseline",
        }
    }

    @property
    def columns_created(self) -> list[str]:
        return ["oral_iron_intervention"]

    @property
    def columns_required(self) -> list[str]:
        return ["tracked"]

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.clock = builder.time.clock()
        self.randomness = builder.randomness.get_stream(self.name)

        self.scenario = builder.configuration.intervention.scenario
        self.ifa_coverage = builder.data.load(
            data_keys.MATERNAL_INTERVENTIONS.IFA_COVERAGE
        ).value[0]
        self.mms_stillbirth_rr = builder.data.load(
            data_keys.MATERNAL_INTERVENTIONS.MMS_STILLBIRTH_RR
        ).value[0]
        self.ifa_effect_size = builder.data.load(
            data_keys.MATERNAL_INTERVENTIONS.IFA_EFFECT_SIZE
        ).value[0]

        builder.value.register_value_modifier(
            "hemoglobin.exposure",
            self.update_exposure,
            requires_columns=self.columns_created,
        )

        builder.value.register_value_modifier(
            "birth_outcome_probabilities",
            self.adjust_stillbirth_probability,
            requires_columns=self.columns_created,
        )

    def on_initialize_simulants(self, pop: SimulantData) -> None:
        if self.scenario == "ifa":
            pop_update = pd.DataFrame(
                {"intervention": "ifa"},
                index=pop.index,
            )
        else:
            pop_update = pd.DataFrame(
                {"intervention": None},
                index=pop.index,
            )
            baseline_ifa = self.randomness.choice(
                pop.index,
                choices=[models.IFA_SUPPLEMENTATION, models.NO_TREATMENT],
                p=[self.ifa_coverage, RESIDUAL_CHOICE],
                additional_key="baseline_ifa",
            )
            low_bmi = pop["maternal_bmi_anemia_category"].isin(
                [models.LOW_BMI_NON_ANEMIC, models.LOW_BMI_ANEMIC]
            )
            coverage = data_values.INTERVENTION_SCENARIO_COVERAGE.loc[self.scenario]
            pop_update["intervention"] = np.where(
                low_bmi, coverage["low_bmi"], coverage["normal_bmi"]
            )

            unsampled_ifa = pop_update["intervention"] == "maybe_ifa"
            pop_update.loc[unsampled_ifa, "intervention"] = baseline_ifa.loc[unsampled_ifa]

        self.population_view.update(pop_update)

    def update_exposure(self, index, exposure):
        pop = self.population_view.get(index)
        exposure.loc[pop["intervention"] == models.NO_TREATMENT] -= (
            self.ifa_coverage * self.ifa_effect_size
        )
        exposure.loc[pop["intervention"] != models.NO_TREATMENT] += (
            1 - self.ifa_coverage
        ) * self.ifa_effect_size

        return exposure

    def adjust_stillbirth_probability(self, index, birth_outcome_probabilities):
        pop = self.population_view.subview(["intervention"]).get(index)
        rrs = {
            models.MMS_SUPPLEMENTATION: self.mms_stillbirth_rr,
        }
        for intervention, rr in rrs.items():
            on_treatment = pop["intervention"] == intervention
            # Add spare probability onto live births first
            birth_outcome_probabilities.loc[
                on_treatment, models.LIVE_BIRTH_OUTCOME
            ] += birth_outcome_probabilities.loc[on_treatment, models.STILLBIRTH_OUTCOME] * (
                1 - rr
            )
            # Then re-scale stillbirth probability
            birth_outcome_probabilities.loc[on_treatment, models.STILLBIRTH_OUTCOME] *= rr
            # This preserves normalization by construction

        return birth_outcome_probabilities
