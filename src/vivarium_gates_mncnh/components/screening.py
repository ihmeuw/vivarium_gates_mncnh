import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.constants.data_values import (
    ANC_ATTENDANCE_TYPES,
    ANEMIA_THRESHOLDS,
    COLUMNS,
    HEMOGLOBIN_TEST_RESULTS,
    HEMOGLOBIN_TEST_SENSITIVITY,
    HEMOGLOBIN_TEST_SPECIFICITY,
    LOW_HEMOGLOBIN_THRESHOLD,
    PIPELINES,
    SIMULATION_EVENT_NAMES,
)
from vivarium_gates_mncnh.constants.scenarios import INTERVENTION_SCENARIOS


class AnemiaScreening(Component):
    @property
    def configuration_defaults(self) -> dict:
        return {
            self.name: {
                "data_sources": {
                    "low_ferritin_probability": data_keys.FERRITIN.PROBABILITY_LOW_FERRITIN
                }
            }
        }

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.scenario = builder.configuration.intervention.scenario
        self.hemoglobin_screening_coverage = (
            builder.data.load(data_keys.HEMOGLOBIN.SCREENING_COVERAGE)
            .query("parameter=='cat2'")
            .reset_index()
            .value[0]
        )
        self.low_ferritin_probability = self.build_lookup_table(
            builder, "low_ferritin_probability"
        )
        builder.population.register_initializer(
            self.initialize_screening,
            columns=[
                COLUMNS.ANEMIA_STATUS_DURING_PREGNANCY,
                COLUMNS.HEMOGLOBIN_SCREENING_COVERAGE,
                COLUMNS.FERRITIN_SCREENING_COVERAGE,
                COLUMNS.TESTED_HEMOGLOBIN,
                COLUMNS.TESTED_FERRITIN,
            ],
            required_resources=[
                COLUMNS.ANC_ATTENDANCE,
                COLUMNS.ANEMIA_INTERVENTION_PROPENSITY,
            ],
        )

    def initialize_screening(self, pop_data: SimulantData) -> None:
        anemia_screening_data = pd.DataFrame(
            {
                COLUMNS.ANEMIA_STATUS_DURING_PREGNANCY: pd.NA,
                COLUMNS.HEMOGLOBIN_SCREENING_COVERAGE: False,
                COLUMNS.FERRITIN_SCREENING_COVERAGE: False,
                COLUMNS.TESTED_HEMOGLOBIN: "not_tested",
                COLUMNS.TESTED_FERRITIN: "not_tested",
            },
            index=pop_data.index,
        )

        self.population_view.update(anemia_screening_data)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.LATER_PREGNANCY_SCREENING:
            return
        pop = self.population_view.get_private_columns(event.index)
        anc_attendance = self.population_view.get_attributes(
            event.index, COLUMNS.ANC_ATTENDANCE
        )
        propensity = self.population_view.get_attributes(
            event.index, COLUMNS.ANEMIA_INTERVENTION_PROPENSITY
        )
        # subset to pop who gets ANC in later pregnancy
        attends_later_anc = anc_attendance.isin(
            [
                ANC_ATTENDANCE_TYPES.LATER_PREGNANCY_ONLY,
                ANC_ATTENDANCE_TYPES.FIRST_TRIMESTER_AND_LATER_PREGNANCY,
            ]
        )
        later_anc_idx = attends_later_anc.index[attends_later_anc]

        # determine anemia status during pregnancy
        hemoglobin = self.population_view.get_attributes(
            later_anc_idx, PIPELINES.HEMOGLOBIN_EXPOSURE
        )
        pop.loc[later_anc_idx, COLUMNS.ANEMIA_STATUS_DURING_PREGNANCY] = (
            pd.cut(
                hemoglobin,
                bins=[-np.inf] + ANEMIA_THRESHOLDS,
                labels=["severe", "moderate", "mild"],
                right=False,
            )
            .astype("object")
            .fillna("not_anemic")
        )

        # determine who gets hemoglobin screening
        if INTERVENTION_SCENARIOS[self.scenario].hemoglobin_screening_coverage == "baseline":
            screen_for_hemoglobin = (
                propensity[later_anc_idx] < self.hemoglobin_screening_coverage
            )
        else:
            # screen all eligible simulants
            screen_for_hemoglobin = pd.Series(False, index=pop.index)
            screen_for_hemoglobin.loc[later_anc_idx] = True

        pop.loc[
            later_anc_idx, COLUMNS.HEMOGLOBIN_SCREENING_COVERAGE
        ] = screen_for_hemoglobin

        # subset to screened population and determine hemoglobin test results (low or adequate)
        screened_pop = pop.loc[pop[COLUMNS.HEMOGLOBIN_SCREENING_COVERAGE]]
        true_hemoglobin_is_low = hemoglobin[screened_pop.index] < LOW_HEMOGLOBIN_THRESHOLD

        test_results_for_truly_low = self.randomness.choice(
            index=screened_pop.index[true_hemoglobin_is_low],
            choices=[HEMOGLOBIN_TEST_RESULTS.LOW, HEMOGLOBIN_TEST_RESULTS.ADEQUATE],
            p=[HEMOGLOBIN_TEST_SENSITIVITY, 1 - HEMOGLOBIN_TEST_SENSITIVITY],
            additional_key="low_hemoglobin_test_result",
        )
        test_results_for_truly_adequate = self.randomness.choice(
            index=screened_pop.index[~true_hemoglobin_is_low],
            choices=[HEMOGLOBIN_TEST_RESULTS.ADEQUATE, HEMOGLOBIN_TEST_RESULTS.LOW],
            p=[HEMOGLOBIN_TEST_SPECIFICITY, 1 - HEMOGLOBIN_TEST_SPECIFICITY],
            additional_key="adequate_hemoglobin_test_result",
        )

        pop.loc[
            test_results_for_truly_low.index, COLUMNS.TESTED_HEMOGLOBIN
        ] = test_results_for_truly_low
        pop.loc[
            test_results_for_truly_adequate.index, COLUMNS.TESTED_HEMOGLOBIN
        ] = test_results_for_truly_adequate

        # subset to those who tested low for hemoglobin and determine ferritin test results
        # in scenarios with ferritin screening (scenarios are either 0% or 100% coverage)
        tested_low_idx = pop[
            pop[COLUMNS.TESTED_HEMOGLOBIN] == HEMOGLOBIN_TEST_RESULTS.LOW
        ].index
        if INTERVENTION_SCENARIOS[self.scenario].ferritin_screening_coverage == "full":
            low_ferritin_probabilities = self.low_ferritin_probability(
                tested_low_idx
            )
            propensities = self.randomness.get_draw(
                index=tested_low_idx, additional_key="tested_ferritin"
            )
            pop.loc[tested_low_idx, COLUMNS.TESTED_FERRITIN] = np.where(
                propensities < low_ferritin_probabilities,
                HEMOGLOBIN_TEST_RESULTS.LOW,
                HEMOGLOBIN_TEST_RESULTS.ADEQUATE,
            )

        self.population_view.update(pop)
