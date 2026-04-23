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
from vivarium_public_health.risks import RiskEffect

from vivarium_gates_mncnh.constants import data_keys, data_values, models
from vivarium_gates_mncnh.constants.data_values import (
    ANC_ATTENDANCE_TYPES,
    COLUMNS,
    HEMOGLOBIN_TEST_RESULTS,
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

        self.paf_table = self.build_lookup_table(builder, "paf")
        self.relative_risk_table = self.build_lookup_table(builder, "relative_risk")

        builder.value.register_attribute_modifier(
            self.target_pipeline_name,
            self.modify_target_pipeline,
            required_resources=[self.col_required],
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
        pop = self.population_view.get(index, self.col_required)
        no_intervention_idx = pop.index[pop == False]
        # NOTE: RR is relative risk for no intervention
        no_intervention_rr = self.relative_risk_table(no_intervention_idx)
        # NOTE: PAF is for no intervention
        paf = self.paf_table(index)
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
    def target_pipeline_name(self) -> str:
        return PIPELINES.PRETERM_WITH_RDS_FINAL_CSMR

    def setup(self, builder: Builder) -> None:
        self.randomness = builder.randomness.get_stream(self.name)

        self.no_cpap_relative_risk_table = self.build_lookup_table(
            builder, "no_cpap_relative_risk"
        )
        self.no_cpap_paf_table = self.build_lookup_table(builder, "no_cpap_paf")
        self.no_acs_relative_risk_table = self.build_lookup_table(
            builder, "no_acs_relative_risk"
        )
        self.no_acs_paf_table = self.build_lookup_table(builder, "no_acs_paf")

        builder.value.register_attribute_modifier(
            self.target_pipeline_name,
            self.modify_target_pipeline,
            required_resources=[
                COLUMNS.CPAP_AVAILABLE,
                COLUMNS.ACS_AVAILABLE,
                COLUMNS.STATED_GESTATIONAL_AGE,
            ],
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
        pop = self.population_view.get(
            index,
            [COLUMNS.CPAP_AVAILABLE, COLUMNS.ACS_AVAILABLE, COLUMNS.STATED_GESTATIONAL_AGE],
        )
        in_acs_gestational_age_range = pop[COLUMNS.STATED_GESTATIONAL_AGE].between(26, 33)
        has_no_cpap = pop[COLUMNS.CPAP_AVAILABLE] == False

        no_intervention_index = has_no_cpap.index
        no_acs_index = pop.index[has_no_cpap & in_acs_gestational_age_range]
        no_cpap_index = pop.index[has_no_cpap & ~in_acs_gestational_age_range]

        # define RR
        no_intervention_rr = pd.Series(1.0, index=index)
        no_intervention_rr.loc[no_cpap_index] = self.no_cpap_relative_risk_table(
            no_cpap_index
        )
        no_intervention_rr.loc[no_acs_index] = self.no_acs_relative_risk_table(
            no_acs_index
        ) * self.no_cpap_relative_risk_table(no_acs_index)

        # define PAF
        no_intervention_paf = self.no_cpap_paf_table(index)
        in_acs_ga_range_index = pop.index[in_acs_gestational_age_range]
        no_intervention_paf.loc[in_acs_gestational_age_range] = self.no_acs_paf_table(
            in_acs_ga_range_index
        )

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
        self.ifa_coverage = (
            builder.data.load(data_keys.IFA_SUPPLEMENTATION.COVERAGE)
            .query("parameter=='cat2'")
            .reset_index()
            .value[0]
        )

        builder.value.register_attribute_producer(
            PIPELINES.ORAL_IRON_INTERVENTION,
            source=self._get_oral_iron_exposure,
            required_resources=[COLUMNS.ORAL_IRON_INTERVENTION],
        )
        builder.value.register_attribute_producer(
            self.ifa_exposure_pipeline_name,
            source=self._get_ifa_exposure,
            required_resources=[COLUMNS.ORAL_IRON_INTERVENTION],
        )
        builder.value.register_attribute_producer(
            self.mmn_exposure_pipeline_name,
            source=self._get_mmn_exposure,
            required_resources=[COLUMNS.ORAL_IRON_INTERVENTION],
        )

        builder.population.register_initializer(
            self._initialize_oral_iron,
            COLUMNS.ORAL_IRON_INTERVENTION,
            required_resources=[COLUMNS.ANC_ATTENDANCE],
        )

    def _initialize_oral_iron(self, pop: SimulantData) -> None:
        pop_update = pd.Series(
            "no_treatment",
            index=pop.index,
            name=COLUMNS.ORAL_IRON_INTERVENTION,
        )
        self.population_view.initialize(pop_update)

    def on_time_step(self, event: Event) -> None:
        def get_pop_with_oral_iron(pop: pd.DataFrame) -> pd.DataFrame:
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
            return pop_update

        if self._sim_step_name() == SIMULATION_EVENT_NAMES.FIRST_TRIMESTER_ANC:
            pop = self.population_view.get(
                event.index,
                [COLUMNS.ANC_ATTENDANCE, COLUMNS.ORAL_IRON_INTERVENTION],
            )
            attends_first_trimester_anc = pop[COLUMNS.ANC_ATTENDANCE].isin(
                [
                    ANC_ATTENDANCE_TYPES.FIRST_TRIMESTER_ONLY,
                    ANC_ATTENDANCE_TYPES.FIRST_TRIMESTER_AND_LATER_PREGNANCY,
                ]
            )
            anc_pop = pop.loc[attends_first_trimester_anc]

            updated_pop = get_pop_with_oral_iron(anc_pop)
            self.population_view.update(
                COLUMNS.ORAL_IRON_INTERVENTION,
                lambda _: updated_pop,
            )

        elif self._sim_step_name() == SIMULATION_EVENT_NAMES.LATER_PREGNANCY_INTERVENTION:
            pop = self.population_view.get(
                event.index,
                [COLUMNS.ANC_ATTENDANCE, COLUMNS.ORAL_IRON_INTERVENTION],
            )
            attends_later_pregnancy_anc = (
                pop[COLUMNS.ANC_ATTENDANCE] == ANC_ATTENDANCE_TYPES.LATER_PREGNANCY_ONLY
            )
            anc_pop = pop.loc[attends_later_pregnancy_anc]

            updated_pop = get_pop_with_oral_iron(anc_pop)
            self.population_view.update(
                COLUMNS.ORAL_IRON_INTERVENTION,
                lambda _: updated_pop,
            )

    def _get_oral_iron_exposure(self, index: pd.Index) -> pd.Series:
        return self.population_view.get(index, COLUMNS.ORAL_IRON_INTERVENTION)

    def _get_ifa_exposure(self, index: pd.Index) -> pd.Series:
        oral_iron = self.population_view.get(index, COLUMNS.ORAL_IRON_INTERVENTION)
        has_ifa = oral_iron.isin(
            [models.ORAL_IRON_INTERVENTION.IFA, models.ORAL_IRON_INTERVENTION.MMS]
        )

        exposure = pd.Series(data_keys.IFA_SUPPLEMENTATION.CAT1, index=index)
        exposure[has_ifa] = data_keys.IFA_SUPPLEMENTATION.CAT2
        return exposure

    def _get_mmn_exposure(self, index: pd.Index) -> pd.Series:
        oral_iron = self.population_view.get(index, COLUMNS.ORAL_IRON_INTERVENTION)
        has_mmn = oral_iron == models.ORAL_IRON_INTERVENTION.MMS

        exposure = pd.Series(data_keys.MMN_SUPPLEMENTATION.CAT1, index=index)
        exposure[has_mmn] = data_keys.MMN_SUPPLEMENTATION.CAT2
        return exposure


class OralIronEffectOnHemoglobin(Component):
    """IFA and MMS effects on hemoglobin."""

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        self.ifa_effect_size = (
            builder.data.load(data_keys.IFA_SUPPLEMENTATION.EFFECT_SIZE)
            .query("affected_target=='hemoglobin.exposure'")
            .reset_index()
            .value[0]
        )

        builder.value.register_attribute_modifier(
            PIPELINES.HEMOGLOBIN_EXPOSURE,
            self.apply_oral_iron_to_hemoglobin,
            required_resources=[
                COLUMNS.ORAL_IRON_INTERVENTION,
                COLUMNS.ANC_ATTENDANCE,
                COLUMNS.IV_IRON_INTERVENTION,
            ],
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def apply_oral_iron_to_hemoglobin(
        self, index: pd.Index, exposure: pd.Series[float]
    ) -> pd.Series[float]:
        pop = self.population_view.get(
            index,
            [
                COLUMNS.ORAL_IRON_INTERVENTION,
                COLUMNS.ANC_ATTENDANCE,
                COLUMNS.IV_IRON_INTERVENTION,
            ],
        )

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
        iv_iron_covered = (
            pop[COLUMNS.IV_IRON_INTERVENTION] == models.IV_IRON_INTERVENTION.COVERED
        )

        needs_first_trimester_update = has_first_trimester_anc & oral_iron_covered
        needs_later_pregnancy_update = (
            has_later_pregnancy_anc & oral_iron_covered & ~iv_iron_covered
        )

        exposure.loc[needs_first_trimester_update] += self.ifa_effect_size
        exposure.loc[needs_later_pregnancy_update] += self.ifa_effect_size

        return exposure


class OralIronEffectOnStillbirth(Component):
    """IFA and MMS effects on stillbirth."""

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        self.mms_stillbirth_rr = builder.data.load(
            data_keys.MMN_SUPPLEMENTATION.STILLBIRTH_RR
        ).value[0]

        builder.value.register_attribute_modifier(
            PIPELINES.BIRTH_OUTCOME_PROBABILITIES,
            self.adjust_stillbirth_probability,
            required_resources=[COLUMNS.ORAL_IRON_INTERVENTION],
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def adjust_stillbirth_probability(
        self, index: pd.Index, birth_outcome_probabilities: pd.DataFrame
    ) -> pd.DataFrame:
        on_treatment = (
            self.population_view.get(index, COLUMNS.ORAL_IRON_INTERVENTION)
            == models.ORAL_IRON_INTERVENTION.MMS
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


class IVIronEffectOnLBWSG(Component):
    """IV iron effects on birth weight and gestational age."""

    BIRTH_EXPOSURE_PIPELINE = "low_birth_weight_and_short_gestation.birth_exposure"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        data = builder.data.load(data_keys.IV_IRON.LBWSG_EFFECT_SIZE)

        birth_weight_data = data.loc[data["outcome"] == "birth_weight"].drop(
            "outcome", axis=1
        )
        gestational_age_data = data.loc[data["outcome"] == "gestational_age"].drop(
            "outcome", axis=1
        )

        self.birth_weight_risk_effect_table = self.build_lookup_table(
            builder,
            "birth_weight_risk_effect",
            data_source=birth_weight_data,
            value_columns="value",
        )
        self.gestational_age_risk_effect_table = self.build_lookup_table(
            builder,
            "gestational_age_risk_effect",
            data_source=gestational_age_data,
            value_columns="value",
        )

        builder.value.register_attribute_modifier(
            self.BIRTH_EXPOSURE_PIPELINE,
            self.apply_iv_iron_to_lbwsg,
            required_resources=[COLUMNS.IV_IRON_INTERVENTION],
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def apply_iv_iron_to_lbwsg(self, index: pd.Index, exposure: pd.DataFrame) -> pd.DataFrame:
        iv_iron = self.population_view.get(index, COLUMNS.IV_IRON_INTERVENTION)
        has_iv_iron = iv_iron == models.IV_IRON_INTERVENTION.COVERED
        covered_index = has_iv_iron.index[has_iv_iron]

        bw_effect = self.birth_weight_risk_effect_table(covered_index)
        ga_effect = self.gestational_age_risk_effect_table(covered_index)

        exposure.loc[covered_index, "birth_weight"] += bw_effect
        exposure.loc[covered_index, "gestational_age"] += ga_effect

        return exposure


class IVIronEffectOnStillbirth(Component):
    """IV iron effects on stillbirth."""

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        stillbirth_rrs = builder.data.load(data_keys.IV_IRON.STILLBIRTH_RR)
        self.stillbirth_relative_risk_table = self.build_lookup_table(
            builder,
            "stillbirth_relative_risk",
            data_source=stillbirth_rrs,
            value_columns="value",
        )

        builder.value.register_attribute_modifier(
            PIPELINES.BIRTH_OUTCOME_PROBABILITIES,
            self.adjust_stillbirth_probability,
            required_resources=[COLUMNS.IV_IRON_INTERVENTION],
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def adjust_stillbirth_probability(
        self, index: pd.Index, birth_outcome_probabilities: pd.DataFrame
    ) -> pd.DataFrame:
        iv_iron = self.population_view.get(index, COLUMNS.IV_IRON_INTERVENTION)
        has_iv_iron = iv_iron == models.IV_IRON_INTERVENTION.COVERED
        rrs = self.stillbirth_relative_risk_table(has_iv_iron.index[has_iv_iron])

        # Add spare probability onto live births first
        birth_outcome_probabilities.loc[
            has_iv_iron, PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME
        ] += birth_outcome_probabilities.loc[
            has_iv_iron, PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME
        ] * (
            1 - rrs
        )
        # Then re-scale stillbirth probability
        birth_outcome_probabilities.loc[
            has_iv_iron, PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME
        ] *= rrs
        # This preserves normalization by construction

        return birth_outcome_probabilities


class AdditiveRiskEffect(RiskEffect):
    BIRTH_EXPOSURE_PIPELINE = "low_birth_weight_and_short_gestation.birth_exposure"

    def __init__(self, risk: str, target: str):
        super().__init__(risk, target)
        self.effect_pipeline_name = f"{self.risk.name}_on_{self.target.name}.effect"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.get_effect_pipeline(builder)
        self.excess_shift_table = self.get_excess_shift(builder)

    def build_rr_lookup_table(self, builder: Builder) -> LookupTable:
        # NOTE: PAF and RR lookup tables do not get used in this class.
        # This is to prevent us from having to configure a scalar for all
        # AdditiveRiskEffect instances in this model
        return self.build_lookup_table(builder, "relative_risk", data_source=1)

    def build_paf_lookup_table(self, builder: Builder) -> LookupTable:
        return self.build_lookup_table(
            builder, "population_attributable_fraction", data_source=0
        )

    def get_effect_pipeline(self, builder: Builder) -> None:
        builder.value.register_attribute_producer(
            self.effect_pipeline_name,
            source=self.get_effect,
            required_resources=[self.exposure_name],
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
        return self.build_lookup_table(
            builder, "excess_shift", data_source=excess_shift_data, value_columns=value_cols
        )

    def get_relative_risk_source(self, builder: Builder) -> Callable[[pd.Index], pd.Series]:
        return lambda index: pd.Series(1.0, index=index)

    def adjust_target(self, index: pd.Index, target: pd.Series) -> pd.Series:
        effect = self.population_view.get(index, self.effect_pipeline_name)
        affected_rates = target + effect
        return affected_rates

    def get_risk_specific_shift_lookup_table(self, builder: Builder) -> LookupTable:
        risk_specific_shift_data = builder.data.load(
            f"{self.risk}.risk_specific_shift",
            affected_entity=self.target.name,
            affected_measure=self.target.measure,
        )
        return self.build_lookup_table(
            builder,
            "risk_specific_shift",
            data_source=risk_specific_shift_data,
            value_columns="value",
        )

    def register_paf_modifier(self, builder: Builder) -> None:
        pass

    def register_target_modifier(self, builder: Builder) -> None:
        builder.value.register_attribute_modifier(
            self.BIRTH_EXPOSURE_PIPELINE,
            modifier=self._adjust_birth_exposure,
            required_resources=[self.relative_risk_name],
        )

    def _adjust_birth_exposure(self, index: pd.Index, target: pd.DataFrame) -> pd.DataFrame:
        target[self.target.name] = self.adjust_target(index, target[self.target.name])
        return target

    def get_excess_shift(self, builder: Builder) -> LookupTable:
        self.excess_shift_lookup_table = self.get_excess_shift_lookup_table(builder)
        self.risk_specific_shift_table = self.get_risk_specific_shift_lookup_table(builder)
        return self.excess_shift_lookup_table

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def get_effect(self, index: pd.Index) -> pd.Series:
        index_columns = ["index", self.risk.name]
        excess_shift = self.excess_shift_table(index)
        exposure = self.population_view.get(index, self.exposure_name).reset_index()
        exposure.columns = index_columns
        exposure = exposure.set_index(index_columns)

        relative_risk = excess_shift.stack().reset_index()
        relative_risk.columns = index_columns + ["value"]
        relative_risk = relative_risk.set_index(index_columns)

        raw_effect = relative_risk.loc[exposure.index, "value"].droplevel(self.risk.name)

        risk_specific_shift = self.risk_specific_shift_table(index)
        effect = raw_effect - risk_specific_shift
        return effect


class OralIronEffectsOnGestationalAge(AdditiveRiskEffect):
    def __init__(self):
        super().__init__(
            f"risk_factor.{COLUMNS.ORAL_IRON_INTERVENTION}",
            "risk_factor.gestational_age.birth_exposure",
        )
        self.ifa_effect_pipeline_name = f"ifa_on_{self.target.name}.effect"
        self.exposure_name = PIPELINES.ORAL_IRON_INTERVENTION

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super(AdditiveRiskEffect, self).setup(builder)
        self.get_ifa_effect_pipeline(builder)
        self.ifa_excess_shift_table = self.get_ifa_excess_shift(builder)

    #######################
    # LookupTable methods #
    #######################

    def build_rr_lookup_table(self, builder: Builder) -> LookupTable:
        return self.build_lookup_table(builder, "relative_risk", data_source=1)

    def build_paf_lookup_table(self, builder: Builder) -> LookupTable:
        return self.build_lookup_table(
            builder, "population_attributable_fraction", data_source=0
        )

    def get_ifa_excess_shift_lookup_table(self, builder: Builder) -> LookupTable:
        excess_shift_data = builder.data.load(
            data_keys.IFA_SUPPLEMENTATION.EXCESS_SHIFT,
            affected_entity=self.target.name,
            affected_measure=self.target.measure,
        )
        excess_shift_data, value_cols = self.process_categorical_data(
            builder, excess_shift_data
        )
        # update data to match oral iron intervention exposure categories
        excess_shift_data = excess_shift_data.rename(
            {"cat1": "no_treatment", "cat2": "ifa"}, axis=1
        )
        excess_shift_data["mms"] = excess_shift_data["ifa"].values
        value_cols = ["no_treatment", "ifa", "mms"]

        return self.build_lookup_table(
            builder,
            "ifa_excess_shift",
            data_source=excess_shift_data,
            value_columns=value_cols,
        )

    def get_ifa_risk_specific_shift_lookup_table(self, builder: Builder) -> LookupTable:
        risk_specific_shift_data = builder.data.load(
            data_keys.IFA_SUPPLEMENTATION.RISK_SPECIFIC_SHIFT,
            affected_entity=self.target.name,
            affected_measure=self.target.measure,
        )
        return self.build_lookup_table(
            builder,
            "ifa_risk_specific_shift",
            data_source=risk_specific_shift_data,
            value_columns="value",
        )

    def _get_mms_excess_shift_data(
        self, builder: Builder, key: str, name: str
    ) -> LookupTable:
        excess_shift_data = builder.data.load(
            key, affected_entity=self.target.name, affected_measure=self.target.measure
        )
        excess_shift_data, value_cols = self.process_categorical_data(
            builder, excess_shift_data
        )
        excess_shift_data = excess_shift_data.rename(
            {"cat1": "no_treatment", "cat2": "mms"}, axis=1
        )
        excess_shift_data["ifa"] = excess_shift_data["no_treatment"].values
        value_cols = ["no_treatment", "ifa", "mms"]
        return self.build_lookup_table(
            builder, name, data_source=excess_shift_data, value_columns=value_cols
        )

    ###############
    # IFA methods #
    ###############

    def get_ifa_effect_pipeline(self, builder: Builder) -> None:
        builder.value.register_attribute_producer(
            self.ifa_effect_pipeline_name,
            source=self.get_ifa_effect,
            required_resources=[self.exposure_name],
        )

    def get_ifa_effect(self, index: pd.Index) -> pd.Series:
        excess_shift = self.ifa_excess_shift_table(index)
        raw_effect = self.calculate_raw_effect(excess_shift, index)

        risk_specific_shift = self.ifa_risk_specific_shift_table(index)

        ifa_effect = raw_effect - risk_specific_shift
        return ifa_effect

    def get_ifa_excess_shift(self, builder: Builder) -> LookupTable:
        self.ifa_excess_shift_lookup_table = self.get_ifa_excess_shift_lookup_table(builder)
        self.ifa_risk_specific_shift_table = self.get_ifa_risk_specific_shift_lookup_table(
            builder
        )
        self.mms_subpop1_excess_shift_table = self._get_mms_excess_shift_data(
            builder,
            data_keys.MMN_SUPPLEMENTATION.EXCESS_GA_SHIFT_SUBPOP_1,
            "mms_subpop1_excess_shift",
        )
        self.mms_subpop2_excess_shift_table = self._get_mms_excess_shift_data(
            builder,
            data_keys.MMN_SUPPLEMENTATION.EXCESS_GA_SHIFT_SUBPOP_2,
            "mms_subpop2_excess_shift",
        )
        return self.ifa_excess_shift_lookup_table

    def calculate_raw_effect(self, excess_shift: pd.Series, index: pd.Index) -> pd.Series:
        index_columns = ["index", self.risk.name]
        exposure = self.population_view.get(index, self.exposure_name).reset_index()
        exposure.columns = index_columns
        exposure = exposure.set_index(index_columns)

        relative_risk = excess_shift.stack().reset_index()
        relative_risk.columns = index_columns + ["value"]
        relative_risk = relative_risk.set_index(index_columns)

        raw_effect = relative_risk.loc[exposure.index, "value"].droplevel(self.risk.name)
        return raw_effect

    ###############
    # MMS methods #
    ###############

    def adjust_target(self, index: pd.Index, target: pd.Series) -> pd.Series:
        pregnancy_outcome = self.population_view.get(index, COLUMNS.PREGNANCY_OUTCOME)
        is_full_term = pregnancy_outcome != PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME

        full_term_index = index[is_full_term]
        result = target.copy()

        ifa_effect = self.population_view.get(full_term_index, self.ifa_effect_pipeline_name)
        ifa_shifted_gestational_age = target[full_term_index] + ifa_effect
        # mms shift is (mms_shift_1 + mms_shift_2) for subpop_2 and mms_shift_1 for subpop_1
        mms_shift_2 = (
            self.mms_subpop2_excess_shift_table(full_term_index)["mms"]
            - self.mms_subpop1_excess_shift_table(full_term_index)["mms"]
        )
        is_subpop_1 = ifa_shifted_gestational_age < (32 - mms_shift_2)
        is_subpop_2 = ifa_shifted_gestational_age >= (32 - mms_shift_2)

        subpop_1_index = full_term_index[is_subpop_1]
        subpop_2_index = full_term_index[is_subpop_2]

        excess_shift = pd.concat(
            [
                self.mms_subpop1_excess_shift_table(subpop_1_index),
                self.mms_subpop2_excess_shift_table(subpop_2_index),
            ]
        )
        mms_effect = self.calculate_raw_effect(excess_shift, full_term_index)

        result[full_term_index] = ifa_shifted_gestational_age + mms_effect
        return result


class IVIronExposure(Component):

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.scenario = builder.configuration.intervention.scenario

        builder.population.register_initializer(
            self._initialize_iv_iron,
            COLUMNS.IV_IRON_INTERVENTION,
            required_resources=[
                COLUMNS.ANC_ATTENDANCE,
                COLUMNS.TESTED_HEMOGLOBIN,
                COLUMNS.TESTED_FERRITIN,
            ],
        )

    def _initialize_iv_iron(self, pop_data: SimulantData) -> None:
        iv_iron_intervention = pd.Series(
            pd.NA,
            index=pop_data.index,
            name=COLUMNS.IV_IRON_INTERVENTION,
        )
        self.population_view.initialize(iv_iron_intervention)

    def on_time_step_prepare(self, event: Event) -> None:
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.LATER_PREGNANCY_INTERVENTION:
            return

        pop = self.population_view.get(
            event.index,
            [
                COLUMNS.ANC_ATTENDANCE,
                COLUMNS.TESTED_HEMOGLOBIN,
                COLUMNS.TESTED_FERRITIN,
                COLUMNS.IV_IRON_INTERVENTION,
            ],
        )

        def _compute_iv_iron(current: pd.Series) -> pd.Series:
            result = pd.Series(
                models.IV_IRON_INTERVENTION.UNCOVERED,
                index=current.index,
                name=COLUMNS.IV_IRON_INTERVENTION,
            )

            # IV iron coverage is either 0 or 100%
            if INTERVENTION_SCENARIOS[self.scenario].iv_iron_coverage == "full":
                attends_later_pregnancy_anc = pop[COLUMNS.ANC_ATTENDANCE].isin(
                    [
                        ANC_ATTENDANCE_TYPES.LATER_PREGNANCY_ONLY,
                        ANC_ATTENDANCE_TYPES.FIRST_TRIMESTER_AND_LATER_PREGNANCY,
                    ]
                )
                tested_low_hemoglobin = (
                    pop[COLUMNS.TESTED_HEMOGLOBIN] == HEMOGLOBIN_TEST_RESULTS.LOW
                )
                tested_low_ferritin = (
                    pop[COLUMNS.TESTED_FERRITIN] == HEMOGLOBIN_TEST_RESULTS.LOW
                )

                gets_iv_iron = (
                    attends_later_pregnancy_anc & tested_low_hemoglobin & tested_low_ferritin
                )
                result.loc[gets_iv_iron] = models.IV_IRON_INTERVENTION.COVERED

            return result

        self.population_view.update(COLUMNS.IV_IRON_INTERVENTION, _compute_iv_iron)


class IVIronEffectOnHemoglobin(Component):
    """IV iron effect on hemoglobin."""

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        self.iv_iron_on_hemoglobin_effect_size = (
            builder.data.load(data_keys.IV_IRON.HEMOGLOBIN_EFFECT_SIZE).reset_index().value[0]
        )

        builder.value.register_attribute_modifier(
            PIPELINES.HEMOGLOBIN_EXPOSURE,
            self.apply_iv_iron_to_hemoglobin,
            required_resources=[COLUMNS.IV_IRON_INTERVENTION],
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def apply_iv_iron_to_hemoglobin(
        self, index: pd.Index, exposure: pd.Series[float]
    ) -> pd.Series[float]:
        iv_iron = self.population_view.get(index, COLUMNS.IV_IRON_INTERVENTION)
        has_iv_iron = iv_iron == models.IV_IRON_INTERVENTION.COVERED
        exposure.loc[has_iv_iron] += self.iv_iron_on_hemoglobin_effect_size

        return exposure
