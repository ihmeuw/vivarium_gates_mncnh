from __future__ import annotations

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.values import list_combiner, union_post_processor
from vivarium_public_health.utilities import get_lookup_columns

from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.constants.data_values import (
    COLUMNS,
    POSTPARTUM_DEPRESSION_CASE_TYPES,
    PREGNANCY_OUTCOMES,
)
from vivarium_gates_mncnh.constants.metadata import ARTIFACT_INDEX_COLUMNS
from vivarium_gates_mncnh.utilities import get_location


class MaternalDisorder(Component):
    @property
    def configuration_defaults(self) -> dict:
        return {self.name: {"data_sources": {"incidence_risk": self.load_incidence_risk}}}

    @property
    def columns_created(self):
        return [self.maternal_disorder]

    @property
    def columns_required(self):
        return [COLUMNS.PREGNANCY_OUTCOME]

    def __init__(self, maternal_disorder: str) -> None:
        super().__init__()
        self.maternal_disorder = maternal_disorder

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.incidence_risk = builder.value.register_value_producer(
            f"{self.maternal_disorder}.incidence_risk",
            source=self.calculate_risk_deleted_incidence,
            component=self,
            required_resources=get_lookup_columns([self.lookup_tables["incidence_risk"]]),
        )
        paf = builder.lookup.build_table(0)
        self.joint_paf = builder.value.register_value_producer(
            f"{self.maternal_disorder}.incidence_risk.paf",
            source=lambda index: [paf(index)],
            component=self,
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        anc_data = pd.DataFrame(
            {
                self.maternal_disorder: False,
            },
            index=pop_data.index,
        )

        self.population_view.update(anc_data)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != self.maternal_disorder:
            return

        pop = self.population_view.get(event.index)
        full_term = pop.loc[
            pop[COLUMNS.PREGNANCY_OUTCOME].isin(
                [PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME, PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME]
            )
        ]
        incidence_risk = self.incidence_risk(full_term.index)
        got_disorder = self.randomness.filter_for_probability(
            full_term.index,
            incidence_risk,
            f"got_{self.maternal_disorder}_choice",
        )
        pop.loc[got_disorder, self.maternal_disorder] = True
        self.population_view.update(pop)

    def load_incidence_risk(self, builder: Builder) -> pd.DataFrame:
        artifact_key = "cause." + self.maternal_disorder + ".incidence_rate"
        raw_incidence = builder.data.load(artifact_key).set_index(ARTIFACT_INDEX_COLUMNS)
        asfr = builder.data.load(data_keys.PREGNANCY.ASFR).set_index(ARTIFACT_INDEX_COLUMNS)
        sbr = (
            builder.data.load(data_keys.PREGNANCY.SBR)
            .set_index("year_start")
            .drop(columns=["year_end"])
            .reindex(asfr.index, level="year_start")
        )
        birth_rate = (sbr + 1) * asfr
        incidence_risk = (raw_incidence / birth_rate).fillna(0.0)
        return incidence_risk.reset_index()

    def calculate_risk_deleted_incidence(self, index: pd.Index) -> pd.Series:
        incidence_risk = self.lookup_tables["incidence_risk"](index)
        joint_paf = self.joint_paf(index)
        return incidence_risk * (1 - joint_paf)


class PostpartumDepression(MaternalDisorder):
    @property
    def configuration_defaults(self) -> dict:
        return {
            self.name: {
                "data_sources": {
                    "incidence_risk": data_keys.POSTPARTUM_DEPRESSION.INCIDENCE_RISK,
                    "case_fatality_rate": data_keys.POSTPARTUM_DEPRESSION.CASE_FATALITY_RATE,
                    "case_duration": data_keys.POSTPARTUM_DEPRESSION.CASE_DURATION,
                    "disability_weight": data_keys.POSTPARTUM_DEPRESSION.DISABILITY_WEIGHT,
                }
            }
        }

    @property
    def columns_created(self) -> list[str]:
        return [
            self.maternal_disorder,
            COLUMNS.POSTPARTUM_DEPRESSION_CASE_TYPE,
            COLUMNS.POSTPARTUM_DEPRESSION_CASE_DURATION,
        ]

    @property
    def columns_required(self) -> list[str]:
        return super().columns_required + [
            COLUMNS.MOTHER_ALIVE,
        ]

    def __init__(self) -> None:
        super().__init__(COLUMNS.POSTPARTUM_DEPRESSION)

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.case_severity_probability = builder.data.load(
            data_keys.POSTPARTUM_DEPRESSION.CASE_SEVERITY
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        anc_data = pd.DataFrame(
            {
                self.maternal_disorder: False,
                COLUMNS.POSTPARTUM_DEPRESSION_CASE_TYPE: POSTPARTUM_DEPRESSION_CASE_TYPES.NONE,
                COLUMNS.POSTPARTUM_DEPRESSION_CASE_DURATION: np.nan,
            },
            index=pop_data.index,
        )

        self.population_view.update(anc_data)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != self.maternal_disorder:
            return

        pop = self.population_view.get(event.index)
        alive = pop.loc[
            (pop[COLUMNS.PREGNANCY_OUTCOME] != PREGNANCY_OUTCOMES.INVALID_OUTCOME)
            & (pop[COLUMNS.MOTHER_ALIVE] == "alive")
        ]
        # Choose who gets PPD
        got_disorder_idx = self.randomness.filter_for_probability(
            alive.index,
            self.incidence_risk(alive.index),
            f"got_{self.maternal_disorder}_choice",
        )
        pop.loc[got_disorder_idx, self.maternal_disorder] = True
        # PPD case type
        case_type = self.randomness.choice(
            got_disorder_idx,
            list(self.case_severity_probability.keys()),
            list(self.case_severity_probability.values()),
            f"{self.maternal_disorder}_case_type_choice",
        )
        pop.loc[got_disorder_idx, COLUMNS.POSTPARTUM_DEPRESSION_CASE_TYPE] = case_type
        # PPD case duration
        pop.loc[
            got_disorder_idx, COLUMNS.POSTPARTUM_DEPRESSION_CASE_DURATION
        ] = self.lookup_tables["case_duration"](got_disorder_idx)

        self.population_view.update(pop)


class AbortionMiscarriageEctopicPregnancy(MaternalDisorder):
    def __init__(self) -> None:
        super().__init__(COLUMNS.ABORTION_MISCARRIAGE_ECTOPIC_PREGNANCY)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != self.maternal_disorder:
            return

        pop = self.population_view.get(event.index)
        pop.loc[
            pop[COLUMNS.PREGNANCY_OUTCOME] == PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME,
            self.maternal_disorder,
        ] = True
        self.population_view.update(pop)


class ResidualMaternalDisorders(MaternalDisorder):
    def __init__(self) -> None:
        super().__init__(COLUMNS.RESIDUAL_MATERNAL_DISORDERS)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != self.maternal_disorder:
            return

        pop = self.population_view.get(event.index)
        pop.loc[
            pop[COLUMNS.PREGNANCY_OUTCOME].isin(
                [PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME, PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME]
            ),
            self.maternal_disorder,
        ] = True
        self.population_view.update(pop)
