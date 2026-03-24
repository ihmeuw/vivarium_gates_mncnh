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
        return {self.name: {"data_sources": {"raw_incidence_risk": self.load_incidence_risk}}}

    def __init__(self, maternal_disorder: str) -> None:
        super().__init__()
        self.maternal_disorder = maternal_disorder

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)

        self.incidence_risk_table = self.build_lookup_table(builder, "raw_incidence_risk")
        builder.value.register_attribute_producer(
            f"{self.maternal_disorder}.incidence_risk",
            source=self.calculate_risk_deleted_incidence,
            required_resources=[self.incidence_risk_table],
        )
        paf = builder.lookup.build_table(0)
        builder.value.register_attribute_producer(
            f"{self.maternal_disorder}.incidence_risk.paf",
            source=lambda index: [paf(index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )
        builder.population.register_initializer(
            self.initialize_disorder,
            columns=[self.maternal_disorder],
        )

    def initialize_disorder(self, pop_data: SimulantData) -> None:
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

        pregnancy_outcome = self.population_view.get_attributes(
            event.index, COLUMNS.PREGNANCY_OUTCOME
        )
        full_term_idx = pregnancy_outcome.index[
            pregnancy_outcome.isin(
                [PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME, PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME]
            )
        ]
        incidence_risk = self.population_view.get_attributes(
            full_term_idx, f"{self.maternal_disorder}.incidence_risk"
        )
        got_disorder = self.randomness.filter_for_probability(
            full_term_idx,
            incidence_risk,
            f"got_{self.maternal_disorder}_choice",
        )
        disorder_col = self.population_view.get_private_columns(
            event.index, self.maternal_disorder
        )
        disorder_col.loc[got_disorder] = True
        self.population_view.update(disorder_col)

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
        joint_paf = self.population_view.get_attributes(
            index, f"{self.maternal_disorder}.incidence_risk.paf"
        )
        return incidence_risk * (1 - joint_paf)


class PostpartumDepression(MaternalDisorder):
    @property
    def configuration_defaults(self) -> dict:
        return {
            self.name: {
                "data_sources": {
                    "raw_incidence_risk": data_keys.POSTPARTUM_DEPRESSION.INCIDENCE_RISK,
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
        self.case_duration_table = self.build_lookup_table(builder, "case_duration")
        builder.population.register_initializer(
            self.initialize_ppd,
            columns=[
                COLUMNS.POSTPARTUM_DEPRESSION_CASE_TYPE,
                COLUMNS.POSTPARTUM_DEPRESSION_CASE_DURATION,
            ],
        )

    def initialize_ppd(self, pop_data: SimulantData) -> None:
        anc_data = pd.DataFrame(
            {
                COLUMNS.POSTPARTUM_DEPRESSION_CASE_TYPE: POSTPARTUM_DEPRESSION_CASE_TYPES.NONE,
                COLUMNS.POSTPARTUM_DEPRESSION_CASE_DURATION: np.nan,
            },
            index=pop_data.index,
        )
        self.population_view.update(anc_data)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != self.maternal_disorder:
            return

        pop = self.population_view.get_attributes(
            event.index,
            [COLUMNS.PREGNANCY_OUTCOME, COLUMNS.MOTHER_IS_ALIVE],
        )
        alive_idx = pop.index[
            (pop[COLUMNS.PREGNANCY_OUTCOME] != PREGNANCY_OUTCOMES.INVALID_OUTCOME)
            & (pop[COLUMNS.MOTHER_IS_ALIVE] == True)
        ]
        # Choose who gets PPD
        incidence_risk = self.population_view.get_attributes(
            alive_idx, f"{self.maternal_disorder}.incidence_risk"
        )
        got_disorder_idx = self.randomness.filter_for_probability(
            alive_idx,
            incidence_risk,
            f"got_{self.maternal_disorder}_choice",
        )

        disorder_col = self.population_view.get_private_columns(
            event.index, self.maternal_disorder
        )
        disorder_col.loc[got_disorder_idx] = True
        self.population_view.update(disorder_col)

        # PPD case type
        case_type = self.randomness.choice(
            got_disorder_idx,
            list(self.case_severity_probability.keys()),
            list(self.case_severity_probability.values()),
            f"{self.maternal_disorder}_case_type_choice",
        )
        case_type_col = pd.Series(
            case_type, index=got_disorder_idx, name=COLUMNS.POSTPARTUM_DEPRESSION_CASE_TYPE
        )
        self.population_view.update(case_type_col)

        # PPD case duration
        case_duration = pd.Series(
            self.case_duration_table(got_disorder_idx),
            index=got_disorder_idx,
            name=COLUMNS.POSTPARTUM_DEPRESSION_CASE_DURATION,
        )
        self.population_view.update(case_duration)


class AbortionMiscarriageEctopicPregnancy(MaternalDisorder):
    def __init__(self) -> None:
        super().__init__(COLUMNS.ABORTION_MISCARRIAGE_ECTOPIC_PREGNANCY)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != self.maternal_disorder:
            return

        pregnancy_outcome = self.population_view.get_attributes(
            event.index, COLUMNS.PREGNANCY_OUTCOME
        )
        partial_term_idx = pregnancy_outcome.index[
            pregnancy_outcome == PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME
        ]
        disorder_col = self.population_view.get_private_columns(
            event.index, self.maternal_disorder
        )
        disorder_col.loc[partial_term_idx] = True
        self.population_view.update(disorder_col)


class ResidualMaternalDisorders(MaternalDisorder):
    @property
    def configuration_defaults(self) -> dict:
        # Adding this to circumvent incidence rate pull
        return {self.name: {"data_sources": {"raw_incidence_risk": 1.0}}}

    def __init__(self) -> None:
        super().__init__(COLUMNS.RESIDUAL_MATERNAL_DISORDERS)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != self.maternal_disorder:
            return

        pregnancy_outcome = self.population_view.get_attributes(
            event.index, COLUMNS.PREGNANCY_OUTCOME
        )
        full_term_idx = pregnancy_outcome.index[
            pregnancy_outcome.isin(
                [PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME, PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME]
            )
        ]
        disorder_col = self.population_view.get_private_columns(
            event.index, self.maternal_disorder
        )
        disorder_col.loc[full_term_idx] = True
        self.population_view.update(disorder_col)
