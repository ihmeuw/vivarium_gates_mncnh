from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable

import numpy as np
import pandas as pd
from layered_config_tree import LayeredConfigTree
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium_public_health.results import COLUMNS, PublicHealthObserver
from vivarium_public_health.results import ResultsStratifier as ResultsStratifier_

from vivarium_gates_mncnh.constants.data_keys import (
    IFA_SUPPLEMENTATION,
    MMN_SUPPLEMENTATION,
    POSTPARTUM_DEPRESSION,
)
from vivarium_gates_mncnh.constants.data_values import (
    ANC_ATTENDANCE_TYPES,
    ANEMIA_THRESHOLDS,
    CAUSES_OF_NEONATAL_MORTALITY,
    COLUMNS,
    DELIVERY_FACILITY_TYPES,
    INTERVENTIONS,
    LOW_HEMOGLOBIN_THRESHOLD,
    MATERNAL_DISORDERS,
    PIPELINES,
    PREGNANCY_OUTCOMES,
    SIMULATION_EVENT_NAMES,
    ULTRASOUND_TYPES,
)
from vivarium_gates_mncnh.constants.metadata import (
    ARTIFACT_INDEX_COLUMNS,
    PRETERM_AGE_CUTOFF,
)
from vivarium_gates_mncnh.utilities import get_child_age_bins


class ResultsStratifier(ResultsStratifier_):
    def setup(self, builder: Builder) -> None:
        self.age_bins = self.get_age_bins(builder)
        self.child_age_bins = get_child_age_bins(builder)
        self.delivery_facility_types = [
            DELIVERY_FACILITY_TYPES.HOME,
            DELIVERY_FACILITY_TYPES.BEmONC,
            DELIVERY_FACILITY_TYPES.CEmONC,
            DELIVERY_FACILITY_TYPES.NONE,
        ]
        self.register_stratifications(builder)

    def register_stratifications(self, builder: Builder) -> None:
        builder.results.register_stratification(
            "age_group",
            self.age_bins["age_group_name"].to_list(),
            mapper=self.map_age_groups,
            is_vectorized=True,
            requires_columns=["age"],
        )

        builder.results.register_stratification(
            "pregnancy_outcome",
            list(
                [
                    PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME,
                    PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME,
                    PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME,
                ]
            ),
            requires_columns=[COLUMNS.PREGNANCY_OUTCOME],
        )

        builder.results.register_stratification(
            "sex", ["Female", "Male"], requires_columns=["sex"]
        )
        builder.results.register_stratification(
            "child_sex",
            ["Female", "Male", "invalid"],
            ["invalid"],
            requires_columns=[COLUMNS.SEX_OF_CHILD],
        )
        builder.results.register_stratification(
            "child_age_group",
            self.child_age_bins["age_group_name"].to_list(),
            excluded_categories=["stillbirth"],
            mapper=self.map_child_age_groups,
            is_vectorized=True,
            requires_columns=[COLUMNS.CHILD_AGE],
        )
        builder.results.register_stratification(
            "delivery_facility_type",
            self.delivery_facility_types,
            is_vectorized=True,
            requires_columns=[COLUMNS.DELIVERY_FACILITY_TYPE],
        )
        builder.results.register_stratification(
            "anc_coverage",
            [
                ANC_ATTENDANCE_TYPES.NONE,
                ANC_ATTENDANCE_TYPES.LATER_PREGNANCY_ONLY,
                ANC_ATTENDANCE_TYPES.FIRST_TRIMESTER_ONLY,
                ANC_ATTENDANCE_TYPES.FIRST_TRIMESTER_AND_LATER_PREGNANCY,
            ],
            is_vectorized=True,
            requires_columns=[COLUMNS.ANC_ATTENDANCE],
        )
        builder.results.register_stratification(
            "ultrasound_type",
            [
                ULTRASOUND_TYPES.STANDARD,
                ULTRASOUND_TYPES.AI_ASSISTED,
                ULTRASOUND_TYPES.NO_ULTRASOUND,
            ],
            is_vectorized=True,
            requires_columns=[COLUMNS.ULTRASOUND_TYPE],
        )
        builder.results.register_stratification(
            "cpap_availability",
            [True, False],
            requires_columns=[COLUMNS.CPAP_AVAILABLE],
        )
        builder.results.register_stratification(
            "antibiotics_availability",
            [True, False],
            requires_columns=[COLUMNS.ANTIBIOTICS_AVAILABLE],
        )
        builder.results.register_stratification(
            "probiotics_availability",
            [True, False],
            requires_columns=[COLUMNS.PROBIOTICS_AVAILABLE],
        )
        builder.results.register_stratification(
            "believed_preterm",
            [True, False],
            mapper=self.map_believed_preterm,
            is_vectorized=True,
            requires_columns=[COLUMNS.STATED_GESTATIONAL_AGE],
        )
        builder.results.register_stratification(
            "preterm_birth",
            [True, False],
            mapper=self.map_preterm_birth,
            is_vectorized=True,
            requires_columns=[COLUMNS.GESTATIONAL_AGE_EXPOSURE],
        )
        builder.results.register_stratification(
            "acs_eligibility",
            [True, False],
            mapper=self.map_acs_eligibility,
            is_vectorized=True,
            requires_columns=[COLUMNS.STATED_GESTATIONAL_AGE],
        )
        builder.results.register_stratification(
            "acs_availability",
            [True, False],
            requires_columns=[COLUMNS.ACS_AVAILABLE],
        )
        builder.results.register_stratification(
            "azithromycin_availability",
            [True, False],
            requires_columns=[COLUMNS.AZITHROMYCIN_AVAILABLE],
        )
        builder.results.register_stratification(
            "misoprostol_availability",
            [True, False],
            requires_columns=[COLUMNS.MISOPROSTOL_AVAILABLE],
        )
        builder.results.register_stratification(
            "oral_iron_coverage",
            ["ifa", "mms", "none"],
            mapper=self.map_oral_iron_coverage,
            is_vectorized=True,
            requires_values=[PIPELINES.IFA_SUPPLEMENTATION, PIPELINES.MMN_SUPPLEMENTATION],
        )
        builder.results.register_stratification(
            "iv_iron_coverage",
            ["covered", "uncovered"],
            is_vectorized=True,
            requires_columns=[COLUMNS.IV_IRON_INTERVENTION],
        )
        builder.results.register_stratification(
            "hemoglobin_screening_coverage",
            [True, False],
            is_vectorized=True,
            requires_columns=[COLUMNS.HEMOGLOBIN_SCREENING_COVERAGE],
        )
        builder.results.register_stratification(
            "ferritin_screening_coverage",
            [True, False],
            is_vectorized=True,
            requires_columns=[COLUMNS.FERRITIN_SCREENING_COVERAGE],
        )
        builder.results.register_stratification(
            "true_hemoglobin_exposure",
            ["low", "adequate"],
            mapper=self.map_true_hemoglobin,
            is_vectorized=True,
            requires_columns=[COLUMNS.FIRST_TRIMESTER_HEMOGLOBIN_EXPOSURE],
        )
        builder.results.register_stratification(
            "tested_hemoglobin_exposure",
            ["low", "adequate", "not_tested"],
            is_vectorized=True,
            requires_columns=[COLUMNS.TESTED_HEMOGLOBIN],
        )
        builder.results.register_stratification(
            "ferritin_status",
            ["low", "adequate", "not_tested"],
            is_vectorized=True,
            requires_columns=[COLUMNS.TESTED_FERRITIN],
        )

    def map_child_age_groups(self, pop: pd.DataFrame) -> pd.Series:
        # Overwriting to use child_age_bins
        bins = self.child_age_bins["child_age_start"].to_list() + [
            self.child_age_bins["child_age_end"].iloc[-1]
        ]
        labels = self.child_age_bins["age_group_name"].to_list()
        age_group = pd.cut(pop.squeeze(axis=1), bins, labels=labels).rename("child_age_group")

        return age_group

    def map_preterm_birth(self, pop: pd.DataFrame) -> pd.Series:
        # Overwriting to use child_age_bins
        gestational_age = pop.squeeze(axis=1)
        preterm_births = gestational_age < PRETERM_AGE_CUTOFF
        return preterm_births.rename("preterm_birth")

    def map_believed_preterm(self, pop: pd.DataFrame) -> pd.Series:
        preterm_births = pop[COLUMNS.STATED_GESTATIONAL_AGE] < PRETERM_AGE_CUTOFF
        return preterm_births.rename("believed_preterm")

    def map_acs_eligibility(self, pop: pd.DataFrame) -> pd.Series:
        is_eligible = pop[COLUMNS.STATED_GESTATIONAL_AGE].between(26, 33)
        return is_eligible.rename("acs_eligibility")

    def map_true_hemoglobin(self, pop: pd.DataFrame) -> pd.Series:
        exposure = pop[COLUMNS.FIRST_TRIMESTER_HEMOGLOBIN_EXPOSURE]
        return pd.Series(
            np.where(exposure < LOW_HEMOGLOBIN_THRESHOLD, "low", "adequate"),
            index=exposure.index,
        )

    def map_oral_iron_coverage(self, pop: pd.DataFrame) -> pd.Series:
        mapped = pop.replace({"cat1": "uncovered", "cat2": "covered"})

        # Create coverage masks
        ifa_covered_mms_uncovered = (mapped[PIPELINES.IFA_SUPPLEMENTATION] == "covered") & (
            mapped[PIPELINES.MMN_SUPPLEMENTATION] == "uncovered"
        )
        mms_covered_ifa_covered = (mapped[PIPELINES.IFA_SUPPLEMENTATION] == "covered") & (
            mapped[PIPELINES.MMN_SUPPLEMENTATION] == "covered"
        )

        # Create the series with the mapping
        oral_iron_coverage = pd.Series("none", index=mapped.index)
        oral_iron_coverage[ifa_covered_mms_uncovered] = "ifa"
        oral_iron_coverage[mms_covered_ifa_covered] = "mms"

        return oral_iron_coverage


class PAFResultsStratifier(ResultsStratifier_):
    def setup(self, builder: Builder) -> None:
        self.child_age_bins = get_child_age_bins(builder)
        self.register_stratifications(builder)

    def register_stratifications(self, builder: Builder) -> None:
        builder.results.register_stratification(
            "child_sex",
            ["Female", "Male", "invalid"],
            ["invalid"],
            requires_columns=[COLUMNS.SEX_OF_CHILD],
        )
        builder.results.register_stratification(
            "child_age_group",
            self.child_age_bins["age_group_name"].to_list(),
            excluded_categories=["stillbirth"],
            mapper=self.map_child_age_groups,
            is_vectorized=True,
            requires_columns=[COLUMNS.CHILD_AGE],
        )

    def map_child_age_groups(self, pop: pd.DataFrame) -> pd.Series:
        # Overwriting to use child_age_bins
        bins = self.child_age_bins["child_age_start"].to_list() + [
            self.child_age_bins["child_age_end"].iloc[-1]
        ]
        labels = self.child_age_bins["age_group_name"].to_list()
        age_group = pd.cut(pop.squeeze(axis=1), bins, labels=labels).rename("child_age_group")

        return age_group


class BirthObserver(PublicHealthObserver):
    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()

    def get_configuration(self, builder: Builder) -> dict[str, Any]:
        return builder.configuration["stratification"][self.get_configuration_name()]

    def register_observations(self, builder: Builder) -> None:
        self.register_adding_observation(
            builder=builder,
            name="births",
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            to_observe=self.to_observe,
        )

    def to_observe(self, event: Event) -> bool:
        return self._sim_step_name() == SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY


class ANCHemoglobinObserver(PublicHealthObserver):
    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()

    def get_configuration(self, builder: Builder) -> dict[str, Any]:
        return builder.configuration["stratification"][self.get_configuration_name()]

    def register_observations(self, builder: Builder) -> None:
        self.register_adding_observation(
            builder=builder,
            name="anc_hemoglobin",
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            to_observe=self.to_observe,
        )

    def to_observe(self, event: Event) -> bool:
        return self._sim_step_name() == SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY


class ANCOtherObserver(PublicHealthObserver):
    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()

    def get_configuration(self, builder: Builder) -> dict[str, Any]:
        return builder.configuration["stratification"][self.get_configuration_name()]

    def register_observations(self, builder: Builder) -> None:
        self.register_adding_observation(
            builder=builder,
            name="anc_other",
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            to_observe=self.to_observe,
        )

    def to_observe(self, event: Event) -> bool:
        return self._sim_step_name() == SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY


class BurdenObserver(PublicHealthObserver):
    def __init__(
        self,
        burden_disorders: list[str],
        alive_column: str,
        ylls_column: str,
        cause_of_death_column: str,
        excluded_causes: list[str] = [],
    ):
        super().__init__()
        self.burden_disorders = burden_disorders
        self.alive_column = alive_column
        self.ylls_column = ylls_column
        self.cause_of_death_column = cause_of_death_column
        self.excluded_causes = excluded_causes

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()

    def get_configuration(self, builder: Builder) -> dict[str, Any]:
        return builder.configuration["stratification"][self.get_configuration_name()]

    def register_observations(self, builder: Builder) -> None:
        dead_pop_filter = f"{self.alive_column} == 'dead'"
        builder.results.register_stratification(
            name=f"{self.name}_cause_of_death",
            categories=self.burden_disorders + ["not_dead"],
            excluded_categories=["not_dead"] + self.excluded_causes,
            requires_columns=[self.cause_of_death_column],
        )

        self.register_adding_observation(
            builder=builder,
            name=f"{self.name}_disorder_deaths",
            pop_filter=dead_pop_filter,
            requires_columns=[self.alive_column],
            additional_stratifications=self.configuration.include
            + [f"{self.name}_cause_of_death"],
            excluded_stratifications=self.configuration.exclude + self.excluded_causes,
            to_observe=self.to_observe,
        )
        self.register_adding_observation(
            builder=builder,
            name=f"{self.name}_disorder_ylls",
            pop_filter=dead_pop_filter,
            requires_columns=[self.alive_column, self.ylls_column],
            additional_stratifications=self.configuration.include
            + [f"{self.name}_cause_of_death"],
            excluded_stratifications=self.configuration.exclude + self.excluded_causes,
            to_observe=self.to_observe,
            aggregator=self.calculate_ylls,
        )

    @abstractmethod
    def to_observe(self, event: Event) -> bool:
        pass

    def calculate_ylls(self, data: pd.DataFrame) -> float:
        return data[self.ylls_column].sum()


class MaternalDisordersBurdenObserver(BurdenObserver):
    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            "stratification": {
                self.get_configuration_name(): {
                    "exclude": [],
                    "include": [],
                    "data_sources": {
                        f"{cause}_ylds": partial(self.load_ylds_per_case, cause=cause)
                        for cause in self.burden_disorders
                    },
                },
            },
        }

    def __init__(self):
        super().__init__(
            burden_disorders=MATERNAL_DISORDERS,
            alive_column=COLUMNS.MOTHER_ALIVE,
            ylls_column=COLUMNS.MOTHER_YEARS_OF_LIFE_LOST,
            cause_of_death_column=COLUMNS.MOTHER_CAUSE_OF_DEATH,
        )

    def register_observations(self, builder: Builder) -> None:
        super().register_observations(builder)
        for cause in self.burden_disorders:
            self.register_adding_observation(
                builder=builder,
                name=f"{cause}_counts",
                pop_filter=f"{cause} == True",
                requires_columns=[cause],
                additional_stratifications=self.configuration.include,
                excluded_stratifications=self.configuration.exclude,
                to_observe=self.to_observe,
            )
            self.register_adding_observation(
                builder=builder,
                name=f"{cause}_ylds",
                pop_filter=f"{cause} == True",
                requires_columns=[cause],
                additional_stratifications=self.configuration.include,
                excluded_stratifications=self.configuration.exclude,
                to_observe=self.to_observe,
                aggregator=partial(self.calculate_ylds, cause=cause),
            )

    def to_observe(self, event: Event) -> bool:
        return self._sim_step_name() == SIMULATION_EVENT_NAMES.MORTALITY

    def calculate_ylds(self, data: pd.DataFrame, cause: str) -> float:
        yld_per_case = self.lookup_tables[f"{cause}_ylds"](data.index)
        return yld_per_case.sum()

    ##################
    # Helper methods #
    ##################

    def load_ylds_per_case(self, builder: Builder, cause: str) -> pd.DataFrame:
        yld_rate = builder.data.load(f"cause.{cause}.yld_rate").set_index(
            ARTIFACT_INDEX_COLUMNS
        )
        special_incidence_rates = {"residual_maternal_disorders": "population.birth_rate"}
        incidence_rate_key = special_incidence_rates.get(
            cause, f"cause.{cause}.incidence_rate"
        )
        incidence_rate = builder.data.load(incidence_rate_key).set_index(
            ARTIFACT_INDEX_COLUMNS
        )
        ylds = (yld_rate / incidence_rate).fillna(0).reset_index()

        return ylds


class NeonatalBurdenObserver(BurdenObserver):
    """Observer to capture death counts and ylls for neonatal sub causes."""

    def __init__(self):
        super().__init__(
            burden_disorders=CAUSES_OF_NEONATAL_MORTALITY
            + ["other_causes"]
            + [
                PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME,
                PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME,
            ],
            alive_column=COLUMNS.CHILD_ALIVE,
            ylls_column=COLUMNS.CHILD_YEARS_OF_LIFE_LOST,
            cause_of_death_column=COLUMNS.CHILD_CAUSE_OF_DEATH,
            excluded_causes=[
                PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME,
                PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME,
            ],
        )

    def register_observations(self, builder: Builder) -> None:
        super().register_observations(builder)
        for cause in set(self.burden_disorders) - set(self.excluded_causes):
            self.register_adding_observation(
                builder=builder,
                name=f"{cause}_death_counts",
                pop_filter=f"{self.cause_of_death_column} == '{cause}'",
                requires_columns=[self.cause_of_death_column],
                additional_stratifications=self.configuration.include,
                excluded_stratifications=self.configuration.exclude,
                to_observe=self.to_observe,
            )

    def to_observe(self, event: Event) -> bool:
        # Need to make single observeration of deaths after all time steps where neonates die.
        return self._sim_step_name() == SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY


class AnemiaYLDsObserver(PublicHealthObserver):
    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            "stratification": {
                self.get_configuration_name(): {
                    "exclude": [],
                    "include": ["age_group"],
                },
            },
        }

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.hemoglobin = builder.value.get_value(PIPELINES.HEMOGLOBIN_EXPOSURE)
        self.gestational_age = builder.value.get_value(PIPELINES.GESTATIONAL_AGE_EXPOSURE)
        self.STAND_IN_ANC1_YEARS = 6 / 52
        self.STAND_IN_ANC_LATER_YEARS = 12 / 52

    def register_observations(self, builder: Builder) -> None:
        self.register_adding_observation(
            builder=builder,
            name="anemia_ylds",
            when="time_step__prepare",
            pop_filter='tracked == True and alive == "alive"',
            requires_columns=[
                COLUMNS.TIME_OF_FIRST_ANC_VISIT,
                COLUMNS.TIME_OF_LATER_ANC_VISIT,
                COLUMNS.ANC_ATTENDANCE,
                "alive",
            ],
            requires_values=[
                PIPELINES.HEMOGLOBIN_EXPOSURE,
                PIPELINES.GESTATIONAL_AGE_EXPOSURE,
            ],
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            to_observe=self.to_observe,
            aggregator=self.calculate_anemia_ylds,
        )

    def to_observe(self, event: Event) -> bool:
        return self._sim_step_name() in [
            SIMULATION_EVENT_NAMES.FIRST_TRIMESTER_ANC,
            SIMULATION_EVENT_NAMES.LATER_PREGNANCY_INTERVENTION,
            SIMULATION_EVENT_NAMES.ULTRASOUND,
            SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY,
        ]

    def calculate_anemia_ylds(self, data: pd.DataFrame) -> float:
        """Calculate YLDs for anemia based on the current simulation event.

        For simulants who do not attend a visit, we use a stand-in value for
        the timing of their visit to simplify calculations. The stand-in values
        are 6 weeks for the first trimester visit and 12 weeks for the later pregnancy visit.

        Gestational age is used as a proxy for pregnancy duration during the
        later pregnancy intervention event because pregnancy duration has not been
        defined for live births and stillbirths by that point. This gestational
        age exposure will be modified by any iron interventions received at a
        first trimester ANC visit.
        """
        duration_calculators = {
            SIMULATION_EVENT_NAMES.FIRST_TRIMESTER_ANC: self._get_first_anc_interval,
            SIMULATION_EVENT_NAMES.LATER_PREGNANCY_INTERVENTION: self._get_later_anc_interval,
            SIMULATION_EVENT_NAMES.ULTRASOUND: self._get_later_anc_to_delivery_interval,
            SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY: lambda df: 6
            * 7
            * (1 / 365.25),  # 6 weeks in years
        }
        anemia_status = self.get_anemia_status_from_hemoglobin(self.hemoglobin(data.index))
        dw = self.get_disability_weight_from_anemia_status(anemia_status)
        duration_years = duration_calculators[self._sim_step_name()](data)
        ylds = dw * duration_years
        return ylds.sum()

    ##################
    # Helper methods #
    ##################

    # duration calculations

    def _get_first_anc_interval(self, data):
        # time from start of sim to first visit
        duration_years = data[COLUMNS.TIME_OF_FIRST_ANC_VISIT] / pd.Timedelta(days=365.25)
        duration_years = duration_years.fillna(self.STAND_IN_ANC1_YEARS)
        return duration_years

    def _get_later_anc_interval(self, data):
        # time from first visit to later visit
        later_visit_years = data[COLUMNS.TIME_OF_LATER_ANC_VISIT] / pd.Timedelta(days=365.25)
        later_visit_years = later_visit_years.fillna(self.STAND_IN_ANC_LATER_YEARS)
        duration_years = later_visit_years - self._get_first_anc_interval(data)
        return duration_years

    def _get_later_anc_to_delivery_interval(self, data):
        # time from later visit to delivery
        later_visit_years = data[COLUMNS.TIME_OF_LATER_ANC_VISIT] / pd.Timedelta(days=365.25)
        later_visit_years = later_visit_years.fillna(self.STAND_IN_ANC_LATER_YEARS)
        gestational_age_years = self.gestational_age(data.index) / 52
        return gestational_age_years - later_visit_years

    # helper methods for disability weights

    @staticmethod
    def get_anemia_status_from_hemoglobin(hemoglobin: pd.Series) -> pd.Series:
        anemia_status = (
            pd.cut(
                hemoglobin,
                bins=[-np.inf] + ANEMIA_THRESHOLDS,
                labels=["severe", "moderate", "mild"],
                right=False,
            )
            .astype("object")
            .fillna("not_anemic")
        )
        return anemia_status

    def get_disability_weight_from_anemia_status(self, anemia_status: pd.Series) -> pd.Series:
        # https://vivarium-research.readthedocs.io/en/latest/models/other_models/hemoglobin_anemia_and_iron_deficiency/anemia_impairment/index.html#id6
        anemia_status_to_dw = {
            "severe": 0.149,
            "moderate": 0.052,
            "mild": 0.004,
            "not_anemic": 0.0,
        }
        return anemia_status.map(anemia_status_to_dw)


class NeonatalCauseRelativeRiskObserver(PublicHealthObserver):
    def __init__(self):
        super().__init__()
        self.neonatal_causes = CAUSES_OF_NEONATAL_MORTALITY + ["all_causes"]

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()

    def get_configuration(self, builder: Builder) -> dict[str, Any]:
        return builder.configuration["stratification"][self.get_configuration_name()]

    def register_observations(self, builder: Builder) -> None:
        for cause in self.neonatal_causes:
            self.register_adding_observation(
                builder=builder,
                name=f"{cause}_relative_risk",
                pop_filter=f"{COLUMNS.PREGNANCY_OUTCOME} == '{PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME}'",
                requires_columns=[COLUMNS.PREGNANCY_OUTCOME],
                requires_values=[
                    f"effect_of_low_birth_weight_and_short_gestation_on_{cause}.relative_risk"
                ],
                additional_stratifications=self.configuration.include,
                excluded_stratifications=self.configuration.exclude,
                to_observe=self.to_observe,
            )

    def to_observe(self, event: Event) -> bool:
        return (self._sim_step_name() == SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY) or (
            self._sim_step_name() == SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY
        )


class InterventionObserver(PublicHealthObserver):
    @property
    def configuration_defaults(self) -> dict[str, Any]:
        """A dictionary containing the defaults for any configurations managed by
        this component.
        """
        return {
            "stratification": {
                f"intervention_{self.intervention}": super().configuration_defaults[
                    "stratification"
                ][self.get_configuration_name()]
            }
        }

    def __init__(self, intervention: str) -> None:
        super().__init__()
        self.intervention = intervention

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()

    def get_configuration(self, builder: Builder) -> LayeredConfigTree:
        """Get the stratification configuration for this observer.

        Parameters
        ----------
        builder
            The builder object for the simulation.

        Returns
        -------
            The stratification configuration for this observer.
        """
        return builder.configuration.stratification[
            f"{self.get_configuration_name()}_{self.intervention}"
        ]

    def register_observations(self, builder: Builder) -> None:
        pop_filter = f"{self.intervention}_available == True"
        if self.intervention in [
            INTERVENTIONS.CPAP,
            INTERVENTIONS.ANTIBIOTICS,
            INTERVENTIONS.PROBIOTICS,
        ]:
            pop_filter += (
                f" & {COLUMNS.PREGNANCY_OUTCOME} == '{PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME}'"
            )
        self.register_adding_observation(
            builder=builder,
            name=self.intervention,
            pop_filter=pop_filter,
            requires_columns=[f"{self.intervention}_available"],
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            to_observe=self.to_observe,
        )

    def to_observe(self, event: Event) -> bool:
        # Last time step
        return self._sim_step_name() == SIMULATION_EVENT_NAMES.POSTPARTUM_DEPRESSION


class PostpartumDepressionObserver(PublicHealthObserver):
    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            "stratification": {
                self.get_configuration_name(): {
                    "exclude": [],
                    "include": [],
                    "data_sources": {
                        "disability_weight": POSTPARTUM_DEPRESSION.DISABILITY_WEIGHT
                    },
                },
            },
        }

    @property
    def columns_required(self) -> list[str]:
        return [
            COLUMNS.POSTPARTUM_DEPRESSION,
            COLUMNS.POSTPARTUM_DEPRESSION_CASE_TYPE,
            COLUMNS.POSTPARTUM_DEPRESSION_CASE_DURATION,
            COLUMNS.MOTHER_ALIVE,
        ]

    def __init__(self) -> None:
        super().__init__()
        self.maternal_disorder = COLUMNS.POSTPARTUM_DEPRESSION

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()

    def register_observations(self, builder: Builder) -> None:
        pop_filter = f"{self.maternal_disorder} == True & {COLUMNS.MOTHER_ALIVE} == 'alive'"
        self.register_adding_observation(
            builder=builder,
            name=f"{self.maternal_disorder}_counts",
            pop_filter=pop_filter,
            requires_columns=[COLUMNS.MOTHER_ALIVE, COLUMNS.POSTPARTUM_DEPRESSION],
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            to_observe=self.to_observe,
        )
        self.register_adding_observation(
            builder=builder,
            name=f"{self.maternal_disorder}_ylds",
            pop_filter=pop_filter,
            requires_columns=self.columns_required,
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            to_observe=self.to_observe,
            aggregator=self.calculate_ylds,
        )

    def calculate_ylds(self, data: pd.DataFrame) -> float:
        """Calculate the YLDs for postpartum depression."""
        case_duration = data[COLUMNS.POSTPARTUM_DEPRESSION_CASE_DURATION]
        disability_weight = self.lookup_tables["disability_weight"](data.index)
        ylds = case_duration * disability_weight

        return ylds.sum()

    def to_observe(self, event: Event) -> bool:
        return self._sim_step_name() == SIMULATION_EVENT_NAMES.POSTPARTUM_DEPRESSION


def register_observations_of_continuous_quantity(
    observer: PublicHealthObserver,
    builder: Builder,
    columns_required: list[str],
    values_required: list[str],
    quantity_name: str,
    get_values: Callable[[pd.DataFrame], pd.Series],
):
    """
    Register observations that compute summary statistics about a continuous quantity:
    the number of values, the number of nonzero values, the sum of the values,
    and the sum of the squares of the values.
    These are "moments" with respect to the count measure, which:
    (a) aggregate additively, and
    (b) allow calculation of the mean and variance, as well as the mean among nonzero values
        and the variance among nonzero values.

    The quantity in question is called `quantity_name` in the resulting observations,
    and `get_values` is a callable that returns the values of this quantity.
    `get_values` will be passed a dataframe containing the `columns_required` from the state table
    and the `values_required` from pipelines.

    We use composition (this function which is passed an observer) rather than inheritance
    (the observer inheriting from a base class containing this functionality)
    because for one of our observers below, we want to register these observations for *each*
    cause.
    This is much easier to do with composition.
    """

    def count_values(data: pd.DataFrame) -> float:
        return len(data)

    def count_nonzero_values(data: pd.DataFrame) -> float:
        return (get_values(data) != 0).sum()

    def sum_values(data: pd.DataFrame) -> float:
        return get_values(data).sum()

    def sum_squared_values(data: pd.DataFrame) -> float:
        return (get_values(data) ** 2).sum()

    observer.register_adding_observation(
        builder=builder,
        name=f"neonatal_{quantity_name}_count",
        pop_filter=f"{COLUMNS.PREGNANCY_OUTCOME} == '{PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME}'",
        requires_columns=columns_required,
        requires_values=values_required,
        additional_stratifications=observer.configuration.include,
        excluded_stratifications=observer.configuration.exclude,
        to_observe=observer.to_observe,
        aggregator=count_values,
    )
    observer.register_adding_observation(
        builder=builder,
        name=f"neonatal_{quantity_name}_nonzero_count",
        pop_filter=f"{COLUMNS.PREGNANCY_OUTCOME} == '{PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME}'",
        requires_columns=columns_required,
        requires_values=values_required,
        additional_stratifications=observer.configuration.include,
        excluded_stratifications=observer.configuration.exclude,
        to_observe=observer.to_observe,
        aggregator=count_nonzero_values,
    )
    observer.register_adding_observation(
        builder=builder,
        name=f"neonatal_{quantity_name}_sum",
        pop_filter=f"{COLUMNS.PREGNANCY_OUTCOME} == '{PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME}'",
        requires_columns=columns_required,
        requires_values=values_required,
        additional_stratifications=observer.configuration.include,
        excluded_stratifications=observer.configuration.exclude,
        to_observe=observer.to_observe,
        aggregator=sum_values,
    )
    observer.register_adding_observation(
        builder=builder,
        name=f"neonatal_{quantity_name}_sum_of_squares",
        pop_filter=f"{COLUMNS.PREGNANCY_OUTCOME} == '{PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME}'",
        requires_columns=columns_required,
        requires_values=values_required,
        additional_stratifications=observer.configuration.include,
        excluded_stratifications=observer.configuration.exclude,
        to_observe=observer.to_observe,
        aggregator=sum_squared_values,
    )


class NeonatalObserver(PublicHealthObserver, ABC):
    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()

    @abstractmethod
    def register_observations(self, builder: Builder) -> None:
        pass

    def to_observe(self, event: Event) -> bool:
        return self._sim_step_name() in (
            SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY,
            SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY,
        )


class NeonatalACMRiskObserver(NeonatalObserver):
    def register_observations(self, builder: Builder):
        register_observations_of_continuous_quantity(
            self,
            builder,
            columns_required=[COLUMNS.PREGNANCY_OUTCOME],
            values_required=[PIPELINES.DEATH_IN_AGE_GROUP_PROBABILITY],
            quantity_name="acmrisk",
            get_values=lambda data: data[PIPELINES.DEATH_IN_AGE_GROUP_PROBABILITY],
        )


class NeonatalCSMRiskObserver(NeonatalObserver):
    def __init__(self) -> None:
        super().__init__()
        self.neonatal_causes = CAUSES_OF_NEONATAL_MORTALITY

    def register_observations(self, builder: Builder):
        for cause in self.neonatal_causes:
            register_observations_of_continuous_quantity(
                self,
                builder,
                columns_required=[COLUMNS.PREGNANCY_OUTCOME],
                values_required=[f"{cause}.csmr"],
                quantity_name=f"{cause}_csmrisk",
                get_values=lambda data, cause=cause: data[f"{cause}.csmr"],
            )


class ImpossibleNeonatalCSMRiskObserver(NeonatalObserver):
    def register_observations(self, builder: Builder):
        register_observations_of_continuous_quantity(
            self,
            builder,
            columns_required=[COLUMNS.PREGNANCY_OUTCOME],
            values_required=[
                PIPELINES.DEATH_IN_AGE_GROUP_PROBABILITY,
                PIPELINES.PRETERM_WITH_RDS_FINAL_CSMR,
                PIPELINES.PRETERM_WITHOUT_RDS_FINAL_CSMR,
                PIPELINES.NEONATAL_SEPSIS_FINAL_CSMR,
                PIPELINES.NEONATAL_ENCEPHALOPATHY_FINAL_CSMR,
            ],
            quantity_name="impossible_csmrisk",
            get_values=self.get_values,
        )

    def get_values(self, data: pd.DataFrame) -> pd.Series:
        total_csmrisk = data[
            [
                PIPELINES.PRETERM_WITH_RDS_FINAL_CSMR,
                PIPELINES.PRETERM_WITHOUT_RDS_FINAL_CSMR,
                PIPELINES.NEONATAL_SEPSIS_FINAL_CSMR,
                PIPELINES.NEONATAL_ENCEPHALOPATHY_FINAL_CSMR,
            ]
        ].sum(axis=1)
        return ((total_csmrisk / data[PIPELINES.DEATH_IN_AGE_GROUP_PROBABILITY]) - 1).clip(0)
