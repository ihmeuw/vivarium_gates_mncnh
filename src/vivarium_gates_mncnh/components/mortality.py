from __future__ import annotations

from functools import partial
from typing import Any

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.values import Pipeline, list_combiner, union_post_processor

from vivarium_gates_mncnh.constants.data_keys import POPULATION
from vivarium_gates_mncnh.constants.data_values import (
    CAUSES_OF_NEONATAL_MORTALITY,
    CHILD_LOOKUP_COLUMN_MAPPER,
    COLUMNS,
    MATERNAL_DISORDERS,
    NEONATAL_CAUSES,
    PIPELINES,
    PREGNANCY_OUTCOMES,
    SIMULATION_EVENT_NAMES,
)
from vivarium_gates_mncnh.constants.metadata import ARTIFACT_INDEX_COLUMNS
from vivarium_gates_mncnh.utilities import get_location, rate_to_probability


class MaternalDisordersBurden(Component):
    """A component to handle morbidity and mortality caused by the modeled maternal disorders."""

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            self.name: {
                "data_sources": {
                    **{
                        "life_expectancy": "population.theoretical_minimum_risk_life_expectancy"
                    },
                    **{
                        f"{cause}_case_fatality_rate": partial(
                            self.load_cfr_data, cause=cause
                        )
                        for cause in self.maternal_disorders
                    },
                },
            },
        }

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self) -> None:
        super().__init__()
        self.maternal_disorders = MATERNAL_DISORDERS

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.location = get_location(builder)
        self.life_expectancy_table = self.build_lookup_table(builder, "life_expectancy")
        self.cfr_tables = {
            cause: self.build_lookup_table(builder, f"{cause}_case_fatality_rate")
            for cause in self.maternal_disorders
        }
        builder.population.register_initializer(
            self.initialize_burden,
            columns=[
                COLUMNS.MOTHER_IS_ALIVE,
                COLUMNS.MOTHER_CAUSE_OF_DEATH,
                COLUMNS.MOTHER_YEARS_OF_LIFE_LOST,
            ],
        )

    ########################
    # Event-driven methods #
    ########################

    def initialize_burden(self, pop_data: SimulantData) -> None:
        pop_update = pd.DataFrame(
            {
                COLUMNS.MOTHER_IS_ALIVE: True,
                COLUMNS.MOTHER_CAUSE_OF_DEATH: "not_dead",
                COLUMNS.MOTHER_YEARS_OF_LIFE_LOST: 0.0,
            },
            index=pop_data.index,
        )
        self.population_view.update(pop_update)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.MORTALITY:
            return

        has_maternal_disorders = self.population_view.get_attributes(
            event.index, self.maternal_disorders
        )
        has_maternal_disorders = has_maternal_disorders.loc[
            has_maternal_disorders.any(axis=1)
        ]

        # Get raw and conditional case fatality rates for each simulant
        choice_data = has_maternal_disorders.copy()
        choice_data = self.calculate_case_fatality_rates(choice_data)

        # Decide what simulants die from what maternal disorders
        dead_idx = self.randomness.filter_for_probability(
            choice_data.index,
            choice_data["mortality_probability"],
            "mortality_choice",
        )

        # Update metadata for simulants that died
        if not dead_idx.empty:
            mother_is_alive = self.population_view.get_attributes(
                event.index, COLUMNS.MOTHER_IS_ALIVE
            )
            mother_is_alive.loc[dead_idx] = False
            self.population_view.update(mother_is_alive)

            cause_of_death = self.population_view.get_private_columns(
                event.index, COLUMNS.MOTHER_CAUSE_OF_DEATH
            )
            ylls = self.population_view.get_private_columns(
                event.index, COLUMNS.MOTHER_YEARS_OF_LIFE_LOST
            )

            # Get maternal disorders each simulant is affect by
            chosen_cause = self.randomness.choice(
                index=dead_idx,
                choices=self.maternal_disorders,
                p=choice_data.loc[
                    dead_idx,
                    [f"{disorder}_proportional_cfr" for disorder in self.maternal_disorders],
                ],
                additional_key="cause_of_death",
            )
            cause_of_death.loc[dead_idx] = chosen_cause
            ylls.loc[dead_idx] = self.life_expectancy_table(dead_idx)

            self.population_view.update(cause_of_death)
            self.population_view.update(ylls)

    ##################
    # Helper methods #
    ##################

    def load_cfr_data(self, builder: Builder, cause: str) -> pd.DataFrame:
        """Load case fatality rate data for maternal disorders."""
        csmr = builder.data.load(f"cause.{cause}.cause_specific_mortality_rate").set_index(
            ARTIFACT_INDEX_COLUMNS
        )
        special_incidence_rates = {"residual_maternal_disorders": "population.birth_rate"}
        incidence_rate_key = special_incidence_rates.get(
            cause, f"cause.{cause}.incidence_rate"
        )
        incidence_rate = builder.data.load(incidence_rate_key).set_index(
            ARTIFACT_INDEX_COLUMNS
        )
        cfr = (csmr / incidence_rate).fillna(0).reset_index()

        return cfr

    def calculate_case_fatality_rates(self, simulants: pd.DataFrame) -> pd.DataFrame:
        """Calculate the total and proportional case fatality rate for each simulant."""

        # Simulants is a boolean dataframe of whether or not a simulant has each maternal disorder.
        for cause in self.maternal_disorders:
            simulants[cause] = simulants[cause] * self.cfr_tables[cause](simulants.index)
        simulants["mortality_probability"] = simulants[self.maternal_disorders].sum(axis=1)
        cfr_data = self.get_proportional_case_fatality_rates(simulants)

        return cfr_data

    def get_proportional_case_fatality_rates(self, simulants: pd.DataFrame) -> pd.DataFrame:
        """Calculate the proportional case fatality rates for each maternal disorder."""

        for cause in self.maternal_disorders:
            simulants[f"{cause}_proportional_cfr"] = (
                simulants[cause] / simulants["mortality_probability"]
            )

        return simulants


class NeonatalMortality(Component):
    """A component to handle neonatal mortality."""

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            self.name: {
                "data_sources": {
                    "all_cause_mortality_risk": self.load_all_causes_mortality_data,
                    "life_expectancy": self.load_life_expectancy_data,
                }
            }
        }

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name, self)
        self.causes_of_death = CAUSES_OF_NEONATAL_MORTALITY + ["other_causes"]

        self.acmr_table = self.build_lookup_table(builder, "all_cause_mortality_risk")
        self.life_expectancy_table = self.build_lookup_table(builder, "life_expectancy")

        # Get neonatal csmr pipelines
        # CSMR pipeline names (now attribute pipelines)
        self.csmr_pipeline_names = [
            PIPELINES.PRETERM_WITH_RDS_FINAL_CSMR,
            PIPELINES.PRETERM_WITHOUT_RDS_FINAL_CSMR,
            PIPELINES.NEONATAL_SEPSIS_FINAL_CSMR,
            PIPELINES.NEONATAL_ENCEPHALOPATHY_FINAL_CSMR,
        ]

        # Register pipelines
        self.get_acmr_paf_pipeline(builder)

        builder.value.register_attribute_producer(
            PIPELINES.ACMR,
            source=self.get_acmr_pipeline,
            required_resources=[PIPELINES.ACMR_PAF],
        )
        # Modify ACMR pipeline with CSMR for neonatal causes
        builder.value.register_attribute_producer(
            PIPELINES.DEATH_IN_AGE_GROUP_PROBABILITY,
            source=self.get_death_in_age_group_probability,
            required_resources=[PIPELINES.ACMR],
        )

        builder.population.register_initializer(
            self.initialize_neonatal_mortality,
            columns=[
                COLUMNS.CHILD_CAUSE_OF_DEATH,
                COLUMNS.CHILD_YEARS_OF_LIFE_LOST,
            ],
            required_resources=[COLUMNS.PREGNANCY_OUTCOME],
        )

        # Register an attribute modifier for child_alive so that Children
        # (the column owner) can read the modified value and update its
        # own private column — mirroring the exit_time / Mortality pattern.
        builder.value.register_attribute_modifier(
            COLUMNS.CHILD_ALIVE, self.modify_child_alive
        )

    def initialize_neonatal_mortality(self, pop_data: SimulantData) -> None:
        pop_update = pd.DataFrame(
            {
                COLUMNS.CHILD_CAUSE_OF_DEATH: "not_dead",
                COLUMNS.CHILD_YEARS_OF_LIFE_LOST: 0.0,
            },
            index=pop_data.index,
        )
        pregnancy_outcomes = self.population_view.get_attributes(
            pop_data.index, COLUMNS.PREGNANCY_OUTCOME
        )
        for outcome in [
            PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME,
            PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME,
        ]:
            outcome_idx = pregnancy_outcomes.index[pregnancy_outcomes == outcome]
            pop_update.loc[outcome_idx, COLUMNS.CHILD_CAUSE_OF_DEATH] = outcome
        self.population_view.update(pop_update)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() not in [
            SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY,
            SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY,
        ]:
            return

        child_alive = self.population_view.get_attributes(event.index, COLUMNS.CHILD_ALIVE)
        alive_idx = child_alive.index[child_alive == "alive"]
        mortality_risk = self.population_view.get_attributes(
            alive_idx, PIPELINES.DEATH_IN_AGE_GROUP_PROBABILITY
        )

        # Determine which neonates die and update metadata
        dead_idx = self.randomness.filter_for_probability(
            alive_idx,
            mortality_risk,
            f"{self._sim_step_name()}_choice",
        )
        if not dead_idx.empty:
            # Store dead indices so the attribute modifier can mark them dead
            # when Children reads child_alive via get_attributes.
            self._newly_dead_idx = dead_idx

            cause_of_death = self.population_view.get_private_columns(
                event.index, COLUMNS.CHILD_CAUSE_OF_DEATH
            )
            ylls = self.population_view.get_private_columns(
                event.index, COLUMNS.CHILD_YEARS_OF_LIFE_LOST
            )
            cause_of_death.loc[dead_idx] = self.determine_cause_of_death(dead_idx)
            ylls.loc[dead_idx] = self.life_expectancy_table(dead_idx)

            self.population_view.update(cause_of_death)
            self.population_view.update(ylls)

    def modify_child_alive(self, index: pd.Index, child_alive: pd.Series) -> pd.Series:
        """Attribute modifier for child_alive; marks newly dead neonates."""
        dead_idx = getattr(self, "_newly_dead_idx", pd.Index([]))
        overlap = dead_idx.intersection(index)
        if not overlap.empty:
            child_alive.loc[overlap] = "dead"
        return child_alive

    ##################
    # Helper methods #
    ##################

    def load_all_causes_mortality_data(self, builder: Builder) -> pd.DataFrame:
        """Load all-cause mortality rate data."""
        acmrisk = builder.data.load(POPULATION.ALL_CAUSES_MORTALITY_RISK)
        child_acmrisk = acmrisk.rename(columns=CHILD_LOOKUP_COLUMN_MAPPER)
        return child_acmrisk

    def load_life_expectancy_data(self, builder: Builder) -> pd.DataFrame:
        """Load life expectancy data."""
        life_expectancy = builder.data.load(
            "population.theoretical_minimum_risk_life_expectancy"
        )
        # This needs to remain here since it gets used for both maternal and neonatal mortality
        child_life_expectancy = life_expectancy.rename(columns=CHILD_LOOKUP_COLUMN_MAPPER)
        return child_life_expectancy

    def determine_cause_of_death(self, simulant_idx: pd.Index) -> pd.Series:
        """Determine the cause of death for neonates."""
        choices = pd.DataFrame(index=simulant_idx)
        mortality_data = self.population_view.get_attributes(
            simulant_idx,
            [PIPELINES.DEATH_IN_AGE_GROUP_PROBABILITY] + self.csmr_pipeline_names,
        )
        all_causes_death_rate = mortality_data[PIPELINES.DEATH_IN_AGE_GROUP_PROBABILITY]
        csmr_data = mortality_data[self.csmr_pipeline_names]
        neonatal_cause_dict = {
            NEONATAL_CAUSES.PRETERM_BIRTH_WITH_RDS: csmr_data[
                PIPELINES.PRETERM_WITH_RDS_FINAL_CSMR
            ],
            NEONATAL_CAUSES.PRETERM_BIRTH_WITHOUT_RDS: csmr_data[
                PIPELINES.PRETERM_WITHOUT_RDS_FINAL_CSMR
            ],
            NEONATAL_CAUSES.NEONATAL_SEPSIS: csmr_data[PIPELINES.NEONATAL_SEPSIS_FINAL_CSMR],
            NEONATAL_CAUSES.NEONATAL_ENCEPHALOPATHY: csmr_data[
                PIPELINES.NEONATAL_ENCEPHALOPATHY_FINAL_CSMR
            ],
        }

        # Calculate proportional cause of death for each neonatal cause
        for cause, pipeline in neonatal_cause_dict.items():
            choices[cause] = pipeline / all_causes_death_rate
        choices["other_causes"] = 1 - choices.sum(axis=1)
        # TODO: fix temporary hack for negative other_causes probabilities
        if (choices["other_causes"] < 0).any():
            negative_idx = choices["other_causes"] < 0
            choices.loc[negative_idx, "other_causes"] = 0
            # Scale each cause of death by the sum of the positive probabilities
            choices.loc[negative_idx] = choices.loc[negative_idx].div(
                choices.loc[negative_idx].sum(axis=1), axis=0
            )

        # Choose cause of death for each neonate
        cause_of_death = self.randomness.choice(
            index=simulant_idx,
            choices=self.causes_of_death,
            p=choices,
            additional_key="cause_of_death",
        )

        return cause_of_death

    def get_acmr_pipeline(self, index: pd.Index) -> Pipeline:
        # NOTE: This will be modified by the LBWSGRiskEffect
        acmr = self.acmr_table(index)
        paf = self.population_view.get_attributes(index, PIPELINES.ACMR_PAF)
        return acmr * (1 - paf)

    def get_death_in_age_group_probability(self, index: pd.Index) -> pd.Series:
        return self.population_view.get_attributes(index, PIPELINES.ACMR)

    def get_acmr_paf_pipeline(self, builder: Builder) -> None:
        acmr_paf = builder.lookup.build_table(0)
        builder.value.register_attribute_producer(
            PIPELINES.ACMR_PAF,
            source=lambda index: [acmr_paf(index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )
