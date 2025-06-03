from __future__ import annotations

from functools import partial
from typing import Any

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.values import Pipeline, list_combiner, union_post_processor
from vivarium_public_health.utilities import get_lookup_columns

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
from vivarium_gates_mncnh.constants.data_keys import POPULATION
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

    @property
    def columns_created(self) -> list[str]:
        return [COLUMNS.MOTHER_CAUSE_OF_DEATH, COLUMNS.MOTHER_YEARS_OF_LIFE_LOST]

    @property
    def columns_required(self) -> list[str]:
        return [
            COLUMNS.MOTHER_ALIVE,
            COLUMNS.EXIT_TIME,
            COLUMNS.MOTHER_AGE,
            COLUMNS.MOTHER_SEX,
        ] + self.maternal_disorders

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

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop_update = pd.DataFrame(
            {
                COLUMNS.MOTHER_CAUSE_OF_DEATH: "not_dead",
                COLUMNS.MOTHER_YEARS_OF_LIFE_LOST: 0.0,
            },
            index=pop_data.index,
        )
        self.population_view.update(pop_update)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.MORTALITY:
            return

        pop = self.population_view.get(event.index)
        has_maternal_disorders = pop[self.maternal_disorders]
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
            pop.loc[dead_idx, COLUMNS.MOTHER_ALIVE] = "dead"

            # Get maternal disorders each simulant is affect by
            cause_of_death = self.randomness.choice(
                index=dead_idx,
                choices=self.maternal_disorders,
                p=choice_data.loc[
                    dead_idx,
                    [f"{disorder}_proportional_cfr" for disorder in self.maternal_disorders],
                ],
                additional_key="cause_of_death",
            )
            pop.loc[dead_idx, COLUMNS.MOTHER_CAUSE_OF_DEATH] = cause_of_death
            pop.loc[dead_idx, COLUMNS.MOTHER_YEARS_OF_LIFE_LOST] = self.lookup_tables[
                "life_expectancy"
            ](dead_idx)

        self.population_view.update(pop)

    ##################
    # Helper methods #
    ##################

    def load_cfr_data(self, builder: Builder, cause: str) -> pd.DataFrame:
        """Load case fatality rate data for maternal disorders."""
        incidence_rate = builder.data.load(f"cause.{cause}.incidence_rate").set_index(
            ARTIFACT_INDEX_COLUMNS
        )
        csmr = builder.data.load(f"cause.{cause}.cause_specific_mortality_rate").set_index(
            ARTIFACT_INDEX_COLUMNS
        )
        cfr = (csmr / incidence_rate).fillna(0).reset_index()

        return cfr

    def calculate_case_fatality_rates(self, simulants: pd.DataFrame) -> pd.DataFrame:
        """Calculate the total and proportional case fatality rate for each simulant."""

        # Simulants is a boolean dataframe of whether or not a simulant has each maternal disorder.
        for cause in self.maternal_disorders:
            simulants[cause] = simulants[cause] * self.lookup_tables[
                f"{cause}_case_fatality_rate"
            ](simulants.index)
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
                    "all_cause_mortality_rate": self.load_all_causes_mortality_data,
                    "life_expectancy": self.load_life_expectancy_data,
                }
            }
        }

    @property
    def columns_created(self) -> list[str]:
        return [COLUMNS.CHILD_CAUSE_OF_DEATH, COLUMNS.CHILD_YEARS_OF_LIFE_LOST]

    @property
    def columns_required(self) -> list[str]:
        return [
            COLUMNS.CHILD_AGE,
            COLUMNS.CHILD_ALIVE,
            COLUMNS.PREGNANCY_OUTCOME,
        ]

    @property
    def initialization_requirements(self):
        return [
            COLUMNS.PREGNANCY_OUTCOME,
        ]

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name, self)
        self.causes_of_death = CAUSES_OF_NEONATAL_MORTALITY + ["other_causes"]

        # Get neonatal csmr pipelines
        self.preterm_with_rds_csmr = builder.value.get_value(
            PIPELINES.PRETERM_WITH_RDS_FINAL_CSMR
        )
        self.preterm_without_rds_csmr = builder.value.get_value(
            PIPELINES.PRETERM_WITHOUT_RDS_FINAL_CSMR
        )
        self.sepsis_csmr = builder.value.get_value(PIPELINES.NEONATAL_SEPSIS_FINAL_CSMR)
        self.encephalopathy_csmr = builder.value.get_value(
            PIPELINES.NEONATAL_ENCEPHALOPATHY_FINAL_CSMR
        )

        # Register pipelines
        self.acmr_paf = self.get_acmr_paf_pipeline(builder)

        self.all_cause_mortality_rate = builder.value.register_value_producer(
            PIPELINES.ACMR,
            source=self.get_acmr_pipeline,
            component=self,
            required_resources=get_lookup_columns(
                [self.lookup_tables["all_cause_mortality_rate"]]
            )
            + [self.acmr_paf],
        )
        # Modify ACMR pipeline with CSMR for neonatal causes
        self.death_in_age_group = builder.value.register_value_producer(
            PIPELINES.DEATH_IN_AGE_GROUP_PROBABILITY,
            source=self.all_cause_mortality_rate,
            component=self,
            required_resources=[self.all_cause_mortality_rate],
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop_update = pd.DataFrame(
            {
                COLUMNS.CHILD_CAUSE_OF_DEATH: "not_dead",
                COLUMNS.CHILD_YEARS_OF_LIFE_LOST: 0.0,
            },
            index=pop_data.index,
        )
        pregnancy_outcomes = self.population_view.subview([COLUMNS.PREGNANCY_OUTCOME]).get(
            pop_data.index
        )
        for outcome in [
            PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME,
            PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME,
        ]:
            outcome_idx = pregnancy_outcomes.index[
                pregnancy_outcomes[COLUMNS.PREGNANCY_OUTCOME] == outcome
            ]
            pop_update.loc[outcome_idx, COLUMNS.CHILD_CAUSE_OF_DEATH] = outcome
        self.population_view.update(pop_update)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() not in [
            SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY,
            SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY,
        ]:
            return

        pop = self.population_view.get(event.index)
        alive_idx = pop.index[pop[COLUMNS.CHILD_ALIVE] == "alive"]
        mortality_rates = self.death_in_age_group(alive_idx)
        # Convert to rates to probability
        if self._sim_step_name() == SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY:
            duration = 7 / 365.0
        else:
            duration = 21 / 365.0
        mortality_risk = rate_to_probability(mortality_rates, duration)

        # Determine which neonates die and update metadata
        dead_idx = self.randomness.filter_for_probability(
            alive_idx,
            mortality_risk,
            f"{self._sim_step_name()}_choice",
        )
        if not dead_idx.empty:
            pop.loc[dead_idx, COLUMNS.CHILD_ALIVE] = "dead"
            pop.loc[dead_idx, COLUMNS.CHILD_CAUSE_OF_DEATH] = self.determine_cause_of_death(
                dead_idx
            )
            pop.loc[dead_idx, COLUMNS.CHILD_YEARS_OF_LIFE_LOST] = self.lookup_tables[
                "life_expectancy"
            ](dead_idx)

        self.population_view.update(pop)

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
        all_causes_death_rate = self.death_in_age_group(simulant_idx)
        neonatal_cause_dict = {
            NEONATAL_CAUSES.PRETERM_BIRTH_WITH_RDS: self.preterm_with_rds_csmr(simulant_idx),
            NEONATAL_CAUSES.PRETERM_BIRTH_WITHOUT_RDS: self.preterm_without_rds_csmr(
                simulant_idx
            ),
            NEONATAL_CAUSES.NEONATAL_SEPSIS: self.sepsis_csmr(simulant_idx),
            NEONATAL_CAUSES.NEONATAL_ENCEPHALOPATHY: self.encephalopathy_csmr(simulant_idx),
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
        acmr = self.lookup_tables["all_cause_mortality_rate"](index)
        paf = self.acmr_paf(index)
        return acmr * (1 - paf)

    def get_acmr_paf_pipeline(self, builder: Builder) -> Pipeline:
        acmr_paf = builder.lookup.build_table(0)
        return builder.value.register_value_producer(
            PIPELINES.ACMR_PAF,
            source=lambda index: [acmr_paf(index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )
