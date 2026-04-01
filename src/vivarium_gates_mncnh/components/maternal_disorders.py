from __future__ import annotations

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.values import list_combiner, union_post_processor

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
        return {
            self.name: {"data_sources": {"incidence_risk_data": self.load_incidence_risk}}
        }

    def __init__(self, maternal_disorder: str) -> None:
        super().__init__()
        self.maternal_disorder = maternal_disorder

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)

        self.incidence_risk_table = self.build_lookup_table(builder, "incidence_risk_data")

        self.incidence_risk_pipeline_name = f"{self.maternal_disorder}.incidence_risk"
        builder.value.register_attribute_producer(
            self.incidence_risk_pipeline_name,
            source=self.calculate_risk_deleted_incidence,
        )

        paf = builder.lookup.build_table(0)
        self.joint_paf_pipeline_name = f"{self.maternal_disorder}.incidence_risk.paf"
        builder.value.register_attribute_producer(
            self.joint_paf_pipeline_name,
            source=lambda index: [paf(index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )

        builder.population.register_initializer(
            self.initialize_maternal_disorder,
            columns=[self.maternal_disorder],
            required_resources=[COLUMNS.PREGNANCY_OUTCOME],
        )

    def initialize_maternal_disorder(self, pop_data: SimulantData) -> None:
        self.population_view.initialize(
            pd.DataFrame(
                {self.maternal_disorder: False},
                index=pop_data.index,
            )
        )

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != self.maternal_disorder:
            return

        pop = self.population_view.get(event.index, [COLUMNS.PREGNANCY_OUTCOME])
        full_term = pop.loc[
            pop[COLUMNS.PREGNANCY_OUTCOME].isin(
                [PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME, PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME]
            )
        ]
        incidence_risk = self.population_view.get(
            full_term.index, self.incidence_risk_pipeline_name
        )
        got_disorder = self.randomness.filter_for_probability(
            full_term.index,
            incidence_risk,
            f"got_{self.maternal_disorder}_choice",
        )

        self.population_view.update(
            self.maternal_disorder,
            lambda col: col.where(~col.index.isin(got_disorder), True),
        )

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
        incidence_risk = self.incidence_risk_table(index)
        joint_paf = self.population_view.get(index, self.joint_paf_pipeline_name)
        return incidence_risk * (1 - joint_paf)


class PostpartumDepression(MaternalDisorder):
    @property
    def configuration_defaults(self) -> dict:
        return {
            self.name: {
                "data_sources": {
                    "incidence_risk_data": data_keys.POSTPARTUM_DEPRESSION.INCIDENCE_RISK,
                    "case_fatality_rate": data_keys.POSTPARTUM_DEPRESSION.CASE_FATALITY_RATE,
                    "case_duration": data_keys.POSTPARTUM_DEPRESSION.CASE_DURATION,
                    "disability_weight": data_keys.POSTPARTUM_DEPRESSION.DISABILITY_WEIGHT,
                }
            }
        }

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
        self.case_fatality_rate_table = self.build_lookup_table(builder, "case_fatality_rate")
        self.case_duration_table = self.build_lookup_table(builder, "case_duration")
        self.disability_weight_table = self.build_lookup_table(builder, "disability_weight")

        builder.population.register_initializer(
            self.initialize_ppd_columns,
            columns=[
                COLUMNS.POSTPARTUM_DEPRESSION_CASE_TYPE,
                COLUMNS.POSTPARTUM_DEPRESSION_CASE_DURATION,
            ],
            required_resources=[COLUMNS.MOTHER_ALIVE],
        )

    def initialize_ppd_columns(self, pop_data: SimulantData) -> None:
        self.population_view.initialize(
            pd.DataFrame(
                {
                    COLUMNS.POSTPARTUM_DEPRESSION_CASE_TYPE: POSTPARTUM_DEPRESSION_CASE_TYPES.NONE,
                    COLUMNS.POSTPARTUM_DEPRESSION_CASE_DURATION: np.nan,
                },
                index=pop_data.index,
            )
        )

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != self.maternal_disorder:
            return

        pop = self.population_view.get(
            event.index, [COLUMNS.PREGNANCY_OUTCOME, COLUMNS.MOTHER_ALIVE]
        )
        alive = pop.loc[
            (pop[COLUMNS.PREGNANCY_OUTCOME] != PREGNANCY_OUTCOMES.INVALID_OUTCOME)
            & pop[COLUMNS.MOTHER_ALIVE]
        ]
        # Choose who gets PPD
        incidence_risk = self.population_view.get(
            alive.index, self.incidence_risk_pipeline_name
        )
        got_disorder_idx = self.randomness.filter_for_probability(
            alive.index,
            incidence_risk,
            f"got_{self.maternal_disorder}_choice",
        )
        # PPD case type
        case_type = self.randomness.choice(
            got_disorder_idx,
            list(self.case_severity_probability.keys()),
            list(self.case_severity_probability.values()),
            f"{self.maternal_disorder}_case_type_choice",
        )
        # PPD case duration
        case_duration = self.case_duration_table(got_disorder_idx)

        # Update columns
        def _set_disorder(col: pd.Series) -> pd.Series:
            result = col.copy()
            result.loc[got_disorder_idx] = True
            return result

        def _set_case_type(col: pd.Series) -> pd.Series:
            result = col.copy()
            result.loc[got_disorder_idx] = case_type
            return result

        def _set_case_duration(col: pd.Series) -> pd.Series:
            result = col.copy()
            result.loc[got_disorder_idx] = case_duration
            return result

        self.population_view.update(self.maternal_disorder, _set_disorder)
        self.population_view.update(COLUMNS.POSTPARTUM_DEPRESSION_CASE_TYPE, _set_case_type)
        self.population_view.update(
            COLUMNS.POSTPARTUM_DEPRESSION_CASE_DURATION, _set_case_duration
        )


class AbortionMiscarriageEctopicPregnancy(MaternalDisorder):
    def __init__(self) -> None:
        super().__init__(COLUMNS.ABORTION_MISCARRIAGE_ECTOPIC_PREGNANCY)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != self.maternal_disorder:
            return

        pop = self.population_view.get(event.index, [COLUMNS.PREGNANCY_OUTCOME])
        partial_term = pop.loc[
            pop[COLUMNS.PREGNANCY_OUTCOME] == PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME
        ].index

        self.population_view.update(
            self.maternal_disorder,
            lambda col: col.where(~col.index.isin(partial_term), True),
        )


class ResidualMaternalDisorders(MaternalDisorder):
    @property
    def configuration_defaults(self) -> dict:
        # Adding this to circumvent incidence rate pull
        return {self.name: {"data_sources": {"incidence_risk_data": 1.0}}}

    def __init__(self) -> None:
        super().__init__(COLUMNS.RESIDUAL_MATERNAL_DISORDERS)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != self.maternal_disorder:
            return

        pop = self.population_view.get(event.index, [COLUMNS.PREGNANCY_OUTCOME])
        full_term = pop.loc[
            pop[COLUMNS.PREGNANCY_OUTCOME].isin(
                [PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME, PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME]
            )
        ].index

        self.population_view.update(
            self.maternal_disorder,
            lambda col: col.where(~col.index.isin(full_term), True),
        )
