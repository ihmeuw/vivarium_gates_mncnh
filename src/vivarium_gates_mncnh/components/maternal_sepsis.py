from __future__ import annotations

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.constants.data_values import COLUMNS, SIMULATION_EVENT_NAMES
from vivarium_gates_mncnh.constants.metadata import ARTIFACT_INDEX_COLUMNS
from vivarium_gates_mncnh.utilities import get_location


class MaternalSepsis(Component):
    @property
    def configuration_defaults(self) -> dict:
        return {self.name: {"data_sources": {"incidence_risk": self.load_incidence_risk}}}

    @property
    def columns_created(self):
        return [COLUMNS.MATERNAL_SEPSIS]

    @property
    def columns_required(self):
        return [COLUMNS.PREGNANCY_OUTCOME]

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.location = get_location(builder)

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        anc_data = pd.DataFrame(
            {
                COLUMNS.MATERNAL_SEPSIS: False,
            },
            index=pop_data.index,
        )

        self.population_view.update(anc_data)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.MATERNAL_SEPSIS:
            return

        pop = self.population_view.get(event.index)
        full_term = pop.loc[pop[COLUMNS.PREGNANCY_OUTCOME].isin(["live_birth", "stillbirth"])]
        sepsis_risk = self.lookup_tables["incidence_risk"](full_term.index)
        got_sepsis = self.randomness.filter_for_probability(
            full_term.index,
            sepsis_risk,
            "got_sepsis_choice",
        )
        pop.loc[got_sepsis, COLUMNS.MATERNAL_SEPSIS] = True
        self.population_view.update(pop)

    def load_incidence_risk(self, builder: Builder) -> pd.DataFrame:
        raw_incidence = builder.data.load(
            data_keys.MATERNAL_SEPSIS.RAW_INCIDENCE_RATE
        ).set_index(ARTIFACT_INDEX_COLUMNS)
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
