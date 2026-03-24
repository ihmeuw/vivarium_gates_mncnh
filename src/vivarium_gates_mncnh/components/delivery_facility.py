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
        self.propensity_pipeline_name = f"{self.name}.correlated_propensity"
        self.bemonc_fraction_table = self.build_lookup_table(
            builder,
            FACILITY_CHOICE.BEmONC_FACILITY_FRACTION,
            data_source=builder.data.load(FACILITY_CHOICE.BEmONC_FACILITY_FRACTION),
            value_columns="value",
        )
        builder.population.register_initializer(
            self.initialize_delivery_facility,
            columns=[COLUMNS.DELIVERY_FACILITY_TYPE],
        )

    def initialize_delivery_facility(self, pop_data: SimulantData) -> None:
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

        pregnancy_outcome = self.population_view.get_attributes(
            event.index, COLUMNS.PREGNANCY_OUTCOME
        )
        not_partial_term = pregnancy_outcome != PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME
        active_idx = pregnancy_outcome.index[not_partial_term]

        stated_ga = self.population_view.get_attributes(
            active_idx, COLUMNS.STATED_GESTATIONAL_AGE
        )
        propensity = self.population_view.get_attributes(
            active_idx, self.propensity_pipeline_name
        )
        is_believed_preterm = stated_ga < 37
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

        facility_type = self.population_view.get_private_columns(
            event.index, COLUMNS.DELIVERY_FACILITY_TYPE
        )
        facility_type.loc[assigned_home.index[assigned_home]] = DELIVERY_FACILITY_TYPES.HOME

        # BEmONC for lower propensities, CEmonC for higher
        facility_type.loc[
            assigned_home.index[~assigned_home]
        ] = DELIVERY_FACILITY_TYPES.CEmONC
        is_bemonc = self.randomness.get_draw(active_idx) < self.bemonc_fraction_table(
            active_idx
        )
        facility_type.loc[
            is_bemonc.index[is_bemonc & ~assigned_home]
        ] = DELIVERY_FACILITY_TYPES.BEmONC

        self.population_view.update(facility_type)
