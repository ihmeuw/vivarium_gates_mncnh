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

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.location = get_location(builder)
        self.delivery_facility_probabilities = {
            FACILITY_CHOICE.P_HOME_PRETERM: builder.data.load(FACILITY_CHOICE.P_HOME_PRETERM),
            FACILITY_CHOICE.P_HOME_FULL_TERM: builder.data.load(
                FACILITY_CHOICE.P_HOME_FULL_TERM
            ),
        }
        self.propensity_name = f"{self.name}.correlated_propensity"

        self.bemonc_facility_fraction_table = self.build_lookup_table(
            builder,
            FACILITY_CHOICE.BEmONC_FACILITY_FRACTION,
            data_source=builder.data.load(FACILITY_CHOICE.BEmONC_FACILITY_FRACTION),
            value_columns="value",
        )

        builder.population.register_initializer(
            self._initialize_delivery_facility,
            columns=[COLUMNS.DELIVERY_FACILITY_TYPE],
        )

    def _initialize_delivery_facility(self, pop_data: SimulantData) -> None:
        anc_data = pd.DataFrame(
            {
                COLUMNS.DELIVERY_FACILITY_TYPE: DELIVERY_FACILITY_TYPES.NONE,
            },
            index=pop_data.index,
        )
        self.population_view.initialize(anc_data)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.DELIVERY_FACILITY:
            return

        pop = self.population_view.get(
            event.index,
            [COLUMNS.PREGNANCY_OUTCOME, COLUMNS.STATED_GESTATIONAL_AGE],
        )
        # Choose delivery facility type
        not_partial_term = (
            pop[COLUMNS.PREGNANCY_OUTCOME] != PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME
        )
        pop = pop[not_partial_term]
        propensity = self.population_view.get(pop.index, self.propensity_name)
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

        # BEmONC for lower propensities, CEmONC for higher
        facility_type = pd.Series(DELIVERY_FACILITY_TYPES.CEmONC, index=pop.index)
        facility_type[assigned_home] = DELIVERY_FACILITY_TYPES.HOME
        is_bemonc = self.randomness.get_draw(pop.index) < self.bemonc_facility_fraction_table(
            pop.index
        )
        facility_type[is_bemonc & ~assigned_home] = DELIVERY_FACILITY_TYPES.BEmONC

        self.population_view.update(COLUMNS.DELIVERY_FACILITY_TYPE, lambda _: facility_type)
