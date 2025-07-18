from abc import abstractmethod
from functools import partial
from typing import Any

import pandas as pd
from layered_config_tree import LayeredConfigTree
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.results import Observer
from vivarium_public_health.results import COLUMNS
from vivarium_public_health.results import ResultsStratifier as ResultsStratifier_

from vivarium_gates_mncnh.constants.data_keys import POSTPARTUM_DEPRESSION
from vivarium_gates_mncnh.constants.data_values import (
    CAUSES_OF_NEONATAL_MORTALITY,
    COLUMNS,
    DELIVERY_FACILITY_TYPES,
    INTERVENTIONS,
    MATERNAL_DISORDERS,
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
            [True, False],
            is_vectorized=True,
            requires_columns=[COLUMNS.ATTENDED_CARE_FACILITY],
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
            "preterm_birth",
            [True, False],
            mapper=self.map_preterm_birth,
            is_vectorized=True,
            requires_columns=[COLUMNS.GESTATIONAL_AGE_EXPOSURE],
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


class BirthObserver(Observer):
    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()

    def get_configuration(self, builder: Builder) -> dict[str, Any]:
        return builder.configuration["stratification"][self.get_configuration_name()]

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_adding_observation(
            name="births",
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            to_observe=self.to_observe,
        )

    def to_observe(self, event: Event) -> bool:
        return self._sim_step_name() == SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY


class ANCObserver(Observer):
    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()

    def get_configuration(self, builder: Builder) -> dict[str, Any]:
        return builder.configuration["stratification"][self.get_configuration_name()]

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_adding_observation(
            name="anc",
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            to_observe=self.to_observe,
        )

    def to_observe(self, event: Event) -> bool:
        return self._sim_step_name() == SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY


class BurdenObserver(Observer):
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

        builder.results.register_adding_observation(
            name=f"{self.name}_disorder_deaths",
            pop_filter=dead_pop_filter,
            requires_columns=[self.alive_column],
            additional_stratifications=self.configuration.include
            + [f"{self.name}_cause_of_death"],
            excluded_stratifications=self.configuration.exclude + self.excluded_causes,
            to_observe=self.to_observe,
        )
        builder.results.register_adding_observation(
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
            builder.results.register_adding_observation(
                name=f"{cause}_counts",
                pop_filter=f"{cause} == True",
                requires_columns=[cause],
                additional_stratifications=self.configuration.include,
                excluded_stratifications=self.configuration.exclude,
                to_observe=self.to_observe,
            )
            builder.results.register_adding_observation(
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
        incidence_rate = builder.data.load(f"cause.{cause}.incidence_rate").set_index(
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
            builder.results.register_adding_observation(
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


class NeonatalCauseRelativeRiskObserver(Observer):
    def __init__(self):
        super().__init__()
        self.neonatal_causes = CAUSES_OF_NEONATAL_MORTALITY + ["all_causes"]

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()

    def get_configuration(self, builder: Builder) -> dict[str, Any]:
        return builder.configuration["stratification"][self.get_configuration_name()]

    def register_observations(self, builder: Builder) -> None:
        for cause in self.neonatal_causes:
            builder.results.register_adding_observation(
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


class InterventionObserver(Observer):
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
            pop_filter += f" & pregnancy_outcome == '{PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME}'"
        builder.results.register_adding_observation(
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


class PostpartumDepressionObserver(Observer):
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
        builder.results.register_adding_observation(
            name=f"{self.maternal_disorder}_counts",
            pop_filter=pop_filter,
            requires_columns=[COLUMNS.MOTHER_ALIVE, COLUMNS.POSTPARTUM_DEPRESSION],
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            to_observe=self.to_observe,
        )
        builder.results.register_adding_observation(
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
