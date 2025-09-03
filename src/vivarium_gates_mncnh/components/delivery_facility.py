import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_gates_mncnh.constants.data_keys import FACILITY_CHOICE
from vivarium_gates_mncnh.constants.data_values import (
    COLUMNS,
    DELIVERY_FACILITY_TYPES,
    PREGNANCY_OUTCOMES,
    SIMULATION_EVENT_NAMES,
)
from vivarium_gates_mncnh.utilities import get_location


class DeliveryFacility(Component):
    """ "Component that stores functionality for the delivery facility choice model."""

    @property
    def columns_created(self) -> list[str]:
        return [COLUMNS.DELIVERY_FACILITY_TYPE]

    @property
    def columns_required(self) -> list[str]:
        return [COLUMNS.PREGNANCY_OUTCOME, COLUMNS.STATED_GESTATIONAL_AGE]

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.location = get_location(builder)
        self.delivery_facility_probabilities = {
            FACILITY_CHOICE.P_HOME_PRETERM: builder.data.load(FACILITY_CHOICE.P_HOME_PRETERM),
            FACILITY_CHOICE.P_HOME_FULL_TERM: builder.data.load(
                FACILITY_CHOICE.P_HOME_FULL_TERM
            ),
            FACILITY_CHOICE.BEmONC_FACILITY_FRACTION: builder.data.load(
                FACILITY_CHOICE.BEmONC_FACILITY_FRACTION
            ),
        }
        self.propensity = builder.value.get_value(f"{self.name}.correlated_propensity")

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        anc_data = pd.DataFrame(
            {
                COLUMNS.DELIVERY_FACILITY_TYPE: DELIVERY_FACILITY_TYPES.NONE,
            },
            index=pop_data.index,
        )
        self.population_view.update(anc_data)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.DELIVERY_FACILITY:
            return

        pop = self.population_view.get(event.index)
        # Choose delivery facility type
        not_partial_term = (
            pop[COLUMNS.PREGNANCY_OUTCOME] != PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME
        )
        pop = pop[not_partial_term]
        propensity = self.propensity(pop.index)
        is_believed_preterm = pop[COLUMNS.STATED_GESTATIONAL_AGE] < 37

        # home births for lower propensities, facility for higher
        assigned_home_preterm = (
            propensity[is_believed_preterm]
            < self.delivery_facility_probabilities[FACILITY_CHOICE.P_HOME_PRETERM]
        )
        assigned_home_full_term = (
            propensity[~is_believed_preterm]
            < self.delivery_facility_probabilities[FACILITY_CHOICE.P_HOME_FULL_TERM]
        )
        assigned_home = pd.concat(
            [assigned_home_preterm, assigned_home_full_term]
        ).sort_index()
        pop.loc[assigned_home, COLUMNS.DELIVERY_FACILITY_TYPE] = DELIVERY_FACILITY_TYPES.HOME

        # BEmONC for lower propensities, CEmonC for higher
        pop.loc[
            ~assigned_home, COLUMNS.DELIVERY_FACILITY_TYPE
        ] = DELIVERY_FACILITY_TYPES.CEmONC
        is_bemonc = (
            self.randomness.get_draw(pop.index)
            < self.delivery_facility_probabilities[FACILITY_CHOICE.BEmONC_FACILITY_FRACTION]
        )
        pop.loc[
            is_bemonc & ~assigned_home, COLUMNS.DELIVERY_FACILITY_TYPE
        ] = DELIVERY_FACILITY_TYPES.BEmONC

        self.population_view.update(pop)
