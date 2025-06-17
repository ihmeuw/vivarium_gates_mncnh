from __future__ import annotations

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium_public_health.utilities import get_lookup_columns

from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.constants.data_values import COLUMNS, PREGNANCY_OUTCOMES
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
            self.lookup_tables["incidence_risk"],
            component=self,
            required_resources=get_lookup_columns([self.lookup_tables["incidence_risk"]]),
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


class PostpartumDepression(MaternalDisorder):
    pass