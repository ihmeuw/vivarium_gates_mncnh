from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from layered_config_tree import ConfigurationError
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RESIDUAL_CHOICE
from vivarium.framework.values import Pipeline
from vivarium_public_health.risks import RiskEffect
from vivarium_public_health.utilities import get_lookup_columns

from vivarium_gates_mncnh.constants import data_keys, data_values, models
from vivarium_gates_mncnh.constants.data_values import (
    ANC_ATTENDANCE_TYPES,
    COLUMNS,
    INTERVENTIONS,
    PIPELINES,
    PREGNANCY_OUTCOMES,
    SIMULATION_EVENT_NAMES,
)
from vivarium_gates_mncnh.constants.scenarios import INTERVENTION_SCENARIOS


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

        # define RR
        no_intervention_rr = pd.Series(1.0, index=index)
        no_intervention_rr.loc[no_cpap_index] = self.lookup_tables["no_cpap_relative_risk"](
            no_cpap_index
        )
        no_intervention_rr.loc[no_acs_index] = self.lookup_tables["no_acs_relative_risk"](
            no_acs_index
        ) * self.lookup_tables["no_cpap_relative_risk"](no_acs_index)

        # define PAF
        no_intervention_paf = self.lookup_tables["no_cpap_paf"](index)
        in_acs_ga_range_index = pop.index[in_acs_gestational_age_range]
        no_intervention_paf.loc[in_acs_gestational_age_range] = self.lookup_tables[
            "no_acs_paf"
        ](in_acs_ga_range_index)

        # update pipeline
        target_pipeline.loc[no_intervention_index] = (
            target_pipeline.loc[no_intervention_index]
            * (1 - no_intervention_paf)
            * no_intervention_rr
        )

        return target_pipeline


class OralIronInterventionExposure(Component):
    CONFIGURATION_DEFAULTS = {
        "intervention": {
            "scenario": "baseline",
        },
    }

    @property
    def time_step_cleanup_priority(self) -> int:
        # ANC attendance will have been updated
        return 6

    @property
    def columns_created(self) -> list[str]:
        return [COLUMNS.ORAL_IRON_INTERVENTION]

    @property
    def columns_required(self) -> list[str]:
        return [COLUMNS.ANC_ATTENDANCE]

    def __init__(self):
        super().__init__()
        self.ifa_exposure_pipeline_name = f"{data_keys.IFA_SUPPLEMENTATION.name}.exposure"
        self.mmn_exposure_pipeline_name = f"{data_keys.MMN_SUPPLEMENTATION.name}.exposure"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.clock = builder.time.clock()
        self.randomness = builder.randomness.get_stream(self.name)

        self.scenario = builder.configuration.intervention.scenario
        self.ifa_coverage = builder.data.load(data_keys.IFA_SUPPLEMENTATION.COVERAGE).value[0]

        self.ifa_exposure_pipeline = builder.value.register_value_producer(
            self.ifa_exposure_pipeline_name,
            source=self._get_ifa_exposure,
            requires_columns=[COLUMNS.ORAL_IRON_INTERVENTION],
        )
        self.mmn_exposure_pipeline = builder.value.register_value_producer(
            self.mmn_exposure_pipeline_name,
            source=self._get_mmn_exposure,
            requires_columns=[COLUMNS.ORAL_IRON_INTERVENTION],
        )

    def on_initialize_simulants(self, pop: SimulantData) -> None:
        pop_update = pd.DataFrame(
            {COLUMNS.ORAL_IRON_INTERVENTION: "N/A"},
            index=pop.index,
        )

        self.population_view.update(pop_update)

    def on_time_step_cleanup(self, event: Event) -> None:
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.PREGNANCY:
            return
        pop = self.population_view.get(event.index)
        attends_anc = pop[COLUMNS.ANC_ATTENDANCE] != ANC_ATTENDANCE_TYPES.NONE
        anc_pop = pop.loc[attends_anc]

        if (
            INTERVENTION_SCENARIOS[self.scenario].ifa_mms_coverage
            == models.ORAL_IRON_INTERVENTION.MMS
        ):
            pop_update = pd.DataFrame(
                {COLUMNS.ORAL_IRON_INTERVENTION: models.ORAL_IRON_INTERVENTION.MMS},
                index=anc_pop.index,
            )
        else:
            pop_update = self.randomness.choice(
                anc_pop.index,
                choices=[
                    models.ORAL_IRON_INTERVENTION.IFA,
                    models.ORAL_IRON_INTERVENTION.NO_TREATMENT,
                ],
                p=[self.ifa_coverage, RESIDUAL_CHOICE],
                additional_key="baseline_ifa",
            )
            pop_update.name = COLUMNS.ORAL_IRON_INTERVENTION

        self.population_view.update(pop_update)

    def _get_ifa_exposure(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get(index)
        has_ifa = pop[COLUMNS.ORAL_IRON_INTERVENTION].isin(
            [models.ORAL_IRON_INTERVENTION.IFA, models.ORAL_IRON_INTERVENTION.MMS]
        )

        exposure = pd.Series(data_keys.IFA_SUPPLEMENTATION.CAT1, index=index)
        exposure[has_ifa] = data_keys.IFA_SUPPLEMENTATION.CAT2
        return exposure

    def _get_mmn_exposure(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get(index)
        has_mmn = pop[COLUMNS.ORAL_IRON_INTERVENTION] == models.ORAL_IRON_INTERVENTION.MMS

        exposure = pd.Series(data_keys.MMN_SUPPLEMENTATION.CAT1, index=index)
        exposure[has_mmn] = data_keys.MMN_SUPPLEMENTATION.CAT2
        return exposure


class OralIronInterventionEffect(Component):
    """IFA and MMS effects on hemoglobin and stillbirth."""

    @property
    def columns_required(self) -> list[str]:
        return [COLUMNS.ORAL_IRON_INTERVENTION, COLUMNS.ANC_ATTENDANCE]

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        self.ifa_coverage = builder.data.load(data_keys.IFA_SUPPLEMENTATION.COVERAGE).value[0]
        self.mms_stillbirth_rr = builder.data.load(
            data_keys.MMN_SUPPLEMENTATION.STILLBIRTH_RR
        ).value[0]
        self.ifa_effect_size = builder.data.load(
            data_keys.IFA_SUPPLEMENTATION.EFFECT_SIZE
        ).value[0]

        builder.value.register_value_modifier(
            PIPELINES.HEMOGLOBIN_EXPOSURE,
            self.update_hemoglobin_exposure,
            requires_columns=self.columns_created,
        )

        builder.value.register_value_modifier(
            PIPELINES.BIRTH_OUTCOME_PROBABILITIES,
            self.adjust_stillbirth_probability,
            requires_columns=self.columns_created,
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def update_hemoglobin_exposure(
        self, index: pd.Index, exposure: pd.Series[float]
    ) -> pd.Series[float]:
        pop = self.population_view.get(index)

        has_first_trimester_anc = pop[COLUMNS.ANC_ATTENDANCE].isin(
            [
                ANC_ATTENDANCE_TYPES.FIRST_TRIMESTER_ONLY,
                ANC_ATTENDANCE_TYPES.FIRST_TRIMESTER_AND_LATER_PREGNANCY,
            ]
        )
        has_later_pregnancy_anc = (
            pop[COLUMNS.ANC_ATTENDANCE] == ANC_ATTENDANCE_TYPES.LATER_PREGNANCY_ONLY
        )
        oral_iron_covered = (
            pop[COLUMNS.ORAL_IRON_INTERVENTION] != models.ORAL_IRON_INTERVENTION.NO_TREATMENT
        )

        needs_first_trimester_update = has_first_trimester_anc & oral_iron_covered
        # TODO: add boolean check for IV iron coverage when implemented
        needs_later_pregnancy_update = has_later_pregnancy_anc & oral_iron_covered

        exposure.loc[needs_first_trimester_update] += self.ifa_effect_size
        exposure.loc[needs_later_pregnancy_update] += self.ifa_effect_size

        return exposure

    def adjust_stillbirth_probability(
        self, index: pd.Index, birth_outcome_probabilities: pd.DataFrame
    ) -> pd.DataFrame:
        pop = self.population_view.subview([COLUMNS.ORAL_IRON_INTERVENTION]).get(index)

        on_treatment = (
            pop[COLUMNS.ORAL_IRON_INTERVENTION] == models.ORAL_IRON_INTERVENTION.MMS
        )
        # Add spare probability onto live births first
        birth_outcome_probabilities.loc[
            on_treatment, PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME
        ] += birth_outcome_probabilities.loc[
            on_treatment, PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME
        ] * (
            1 - self.mms_stillbirth_rr
        )
        # Then re-scale stillbirth probability
        birth_outcome_probabilities.loc[
            on_treatment, PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME
        ] *= self.mms_stillbirth_rr
        # This preserves normalization by construction

        return birth_outcome_probabilities


class AdditiveRiskEffect(RiskEffect):
    def __init__(self, risk: str, target: str):
        super().__init__(risk, target)
        self.effect_pipeline_name = f"{self.risk.name}_on_{self.target.name}.effect"

    @property
    def is_exposure_categorical(self) -> bool:
        return True

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.effect = self.get_effect_pipeline(builder)
        self.excess_shift = self.get_excess_shift(builder)

    def build_all_lookup_tables(self, builder: Builder) -> None:
        # NOTE: I have overwritten this method since PAF and RR lookup tables do not
        # get used in this class. This is to prevent us from having to configure a scalar for all
        # AdditiveRiskEffect instances in this model
        self.lookup_tables["relative_risk"] = self.build_lookup_table(builder, 1)
        self.lookup_tables["population_attributable_fraction"] = self.build_lookup_table(
            builder, 0
        )
        self.lookup_tables["excess_shift"] = self.get_excess_shift_lookup_table(builder)
        self.lookup_tables["risk_specific_shift"] = self.get_risk_specific_shift_lookup_table(
            builder
        )

    def get_effect_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.effect_pipeline_name,
            source=self.get_effect,
            requires_values=[self.exposure_pipeline_name],
        )

    def get_excess_shift_lookup_table(self, builder: Builder) -> LookupTable:
        excess_shift_data = builder.data.load(
            f"{self.risk}.excess_shift",
            affected_entity=self.target.name,
            affected_measure=self.target.measure,
        )
        excess_shift_data, value_cols = self.process_categorical_data(
            builder, excess_shift_data
        )
        return self.build_lookup_table(builder, excess_shift_data, value_cols)

    def get_relative_risk_source(self, builder: Builder) -> Callable[[pd.Index], pd.Series]:
        return lambda index: pd.Series(1.0, index=index)

    def get_target_modifier(
        self, builder: Builder
    ) -> Callable[[pd.Index, pd.Series], pd.Series]:
        def adjust_target(index: pd.Index, target: pd.Series) -> pd.Series:
            affected_rates = target + self.effect(index)
            return affected_rates

        return adjust_target

    def get_risk_specific_shift_lookup_table(self, builder: Builder) -> LookupTable:
        risk_specific_shift_data = builder.data.load(
            f"{self.risk}.risk_specific_shift",
            affected_entity=self.target.name,
            affected_measure=self.target.measure,
        )
        return self.build_lookup_table(builder, risk_specific_shift_data, ["value"])

    def register_paf_modifier(self, builder: Builder) -> None:
        pass

    def get_excess_shift(self, builder: Builder) -> LookupTable | Pipeline:
        return self.lookup_tables["excess_shift"]

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def get_effect(self, index: pd.Index) -> pd.Series:
        index_columns = ["index", self.risk.name]
        excess_shift = self.excess_shift(index)
        exposure = self.exposure(index).reset_index()
        exposure.columns = index_columns
        exposure = exposure.set_index(index_columns)

        relative_risk = excess_shift.stack().reset_index()
        relative_risk.columns = index_columns + ["value"]
        relative_risk = relative_risk.set_index(index_columns)

        raw_effect = relative_risk.loc[exposure.index, "value"].droplevel(self.risk.name)

        risk_specific_shift = self.lookup_tables["risk_specific_shift"](index)
        effect = raw_effect - risk_specific_shift
        return effect


class MMSEffectOnGestationalAge(AdditiveRiskEffect):
    """Model effect of multiple micronutrient supplementation on gestational age.
    Unique component because the excess shift value depends on IFA-shifted gestational age."""

    def __init__(self):
        super().__init__(
            "risk_factor.multiple_micronutrient_supplementation",
            "risk_factor.gestational_age.birth_exposure",
        )
        self.excess_shift_pipeline_name = (
            f"{self.risk.name}_on_{self.target.name}.excess_shift"
        )
        self.risk_specific_shift_pipeline_name = (
            f"{self.risk.name}_on_{self.target.name}.risk_specific_shift"
        )
        self.raw_gestational_age_exposure_column_name = "gestational_age_exposure"

    @property
    def columns_required(self) -> list[str]:
        return [self.raw_gestational_age_exposure_column_name]

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.ifa_on_gestational_age = builder.components.get_component(
            f"risk_effect.iron_folic_acid_supplementation_on_{self.target}"
        )

    def build_all_lookup_tables(self, builder: Builder) -> None:
        self.lookup_tables["relative_risk"] = self.build_lookup_table(builder, 1)
        self.lookup_tables["population_attributable_fraction"] = self.build_lookup_table(
            builder, 0
        )
        self.lookup_tables["risk_specific_shift"] = self.get_risk_specific_shift_lookup_table(
            builder
        )
        self.lookup_tables["mms_subpop1_excess_shift"] = self._get_mms_excess_shift_data(
            builder, data_keys.MMN_SUPPLEMENTATION.EXCESS_GA_SHIFT_SUBPOP_1
        )
        self.lookup_tables["mms_subpop2_excess_shift"] = self._get_mms_excess_shift_data(
            builder, data_keys.MMN_SUPPLEMENTATION.EXCESS_GA_SHIFT_SUBPOP_2
        )

    def _get_mms_excess_shift_data(self, builder: Builder, key: str) -> LookupTable:
        excess_shift_data = builder.data.load(
            key, affected_entity=self.target.name, affected_measure=self.target.measure
        )
        excess_shift_data, value_cols = self.process_categorical_data(
            builder, excess_shift_data
        )
        return self.build_lookup_table(builder, excess_shift_data, value_cols)

    def get_excess_shift(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.excess_shift_pipeline_name,
            source=self.get_excess_shift_source,
            requires_columns=[self.raw_gestational_age_exposure_column_name],
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def get_risk_specific_shift_lookup_table(self, builder: Builder) -> LookupTable:
        return self.build_lookup_table(builder, 0)

    def get_excess_shift_source(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get(index)
        raw_gestational_age = pop[self.raw_gestational_age_exposure_column_name]
        ifa_shifted_gestational_age = (
            raw_gestational_age + self.ifa_on_gestational_age.effect(index)
        )
        # excess shift is (mms_shift_1 + mms_shift_2) for subpop_2 and mms_shift_1 for subpop_1
        mms_shift_2 = (
            self.lookup_tables["mms_subpop2_excess_shift"](index)["cat2"]
            - self.lookup_tables["mms_subpop1_excess_shift"](index)["cat2"]
        )
        is_subpop_1 = ifa_shifted_gestational_age < (32 - mms_shift_2)
        is_subpop_2 = ifa_shifted_gestational_age >= (32 - mms_shift_2)

        subpop_1_index = pop[is_subpop_1].index
        subpop_2_index = pop[is_subpop_2].index

        excess_shift = pd.concat(
            [
                self.lookup_tables["mms_subpop1_excess_shift"](subpop_1_index),
                self.lookup_tables["mms_subpop2_excess_shift"](subpop_2_index),
            ]
        )

        return excess_shift
