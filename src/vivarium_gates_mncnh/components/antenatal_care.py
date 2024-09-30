import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_gates_mncnh.constants.data_values import (
    ANC_RATES,
    COLUMNS,
    SIMULATION_EVENT_NAMES,
)


class AntenatalCare(Component):
    @property
    def columns_created(self):
        return [
            COLUMNS.ATTENDED_CARE_FACILITY,
            COLUMNS.RECEIVED_ULTRASOUND,
            COLUMNS.ULTRASOUND_TYPE,
            COLUMNS.STATED_GESTATIONAL_AGE,
            COLUMNS.SUCCESSFUL_LBW_IDENTIFICATION,
        ]

    @property
    def columns_required(self):
        return [
            COLUMNS.GESTATIONAL_AGE,
            COLUMNS.BIRTH_WEIGHT,
            COLUMNS.SEX_OF_CHILD,
        ]

    def setup(self, builder: Builder):
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.location = self._get_location(builder)

    def build_all_lookup_tables(self, builder: Builder) -> None:
        # TODO: I don't think we need this
        pass

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        # TODO: is this the best data to initialize these columns with?
        anc_data = pd.DataFrame(
            {
                COLUMNS.ATTENDED_CARE_FACILITY: False,
                COLUMNS.RECEIVED_ULTRASOUND: False,
                COLUMNS.ULTRASOUND_TYPE: "no_ultrasound",
                COLUMNS.STATED_GESTATIONAL_AGE: np.nan,
                COLUMNS.SUCCESSFUL_LBW_IDENTIFICATION: np.nan,
            },
            index=pop_data.index,
        )
        self.population_view.update(anc_data)

    def on_time_step(self, event: Event) -> None:
        """We need to make 3 decisions in this method and record data based on that decision tree.
        First, we need to decide whether simulants go to a care facility for ANC. Second, we need
        to decide if the simulants that attend a care facility get an ultrasound. Finally, we need
        to determine what type of ultrasound a simulant received if they got an ultrasound. We also
        need to record the stated gestational age and whether the child was successfully identified
        as low birth weight. Because we are walking through a decision tree, oder of operations is
        very important.
        """
        # TODO: is this necessary?
        if self._sim_step_name != SIMULATION_EVENT_NAMES.PREGNANCY:
            pass
        population = self.population_view.get(event.index)

        # Decide if simulants went to a facility for ANC
        population[COLUMNS.ATTENDED_CARE_FACILITY] = self.randomness.choice(
            index=population.index,
            choices=[True, False],
            p=[
                ANC_RATES.ATTENDED_CARE_FACILITY[self.location],
                (1 - ANC_RATES.ATTENDED_CARE_FACILITY[self.location]),
            ],
            additional_key="attended_care_facility",
        )
        # Decide if simulants that went to a facility got an ultrasound
        anc_index = population.index[population[COLUMNS.ATTENDED_CARE_FACILITY] == True]
        ultrasound_decision = self.randomness.choice(
            index=anc_index,
            choices=[True, False],
            p=[
                ANC_RATES.RECEIVED_ULTRASOUND[self.location],
                (1 - ANC_RATES.RECEIVED_ULTRASOUND[self.location]),
            ],
            additional_key="received_ultrasound",
        )
        population.loc[anc_index, COLUMNS.RECEIVED_ULTRASOUND] = ultrasound_decision

        # Decide what type of ultrasound a simulant received
        ultrasound_index = population.index[population[COLUMNS.RECEIVED_ULTRASOUND] == True]
        # TODO: update data
        ultrasound_type = self.randomness.choice(
            index=ultrasound_index,
            choices=["standard", "AI_assisted"],
            p=[
                ANC_RATES.ULTRASOUND_TYPE["standard"],
                ANC_RATES.ULTRASOUND_TYPE["AI_assisted"],
            ],
        )
        population.loc[ultrasound_index, COLUMNS.ULTRASOUND_TYPE] = ultrasound_type

        # Get index groups for each ultrasound type
        no_ultrasound_index = population.index[
            population[COLUMNS.ULTRASOUND_TYPE] == "no_ultrasound"
        ]
        standard_ultrasound_index = population.index[
            population[COLUMNS.ULTRASOUND_TYPE] == "standard"
        ]
        ai_ultrasound_index = population.index[
            population[COLUMNS.ULTRASOUND_TYPE] == "AI_assisted"
        ]

        # Record the stated gestational age

        # Record whether the child was successfully identified as low birth weight

    def _get_location(self, builder: Builder) -> str:
        return builder.data.load("population.location")
