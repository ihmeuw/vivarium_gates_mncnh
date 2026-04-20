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
        self.hemoglobin_pipeline_name = PIPELINES.HEMOGLOBIN_EXPOSURE
        self.low_ferritin_probability = self.build_lookup_table(
            builder, "low_ferritin_probability"
        )

        builder.population.register_initializer(
            self.on_initialize_simulants,
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

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
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
        self.population_view.initialize(anemia_screening_data)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.LATER_PREGNANCY_SCREENING:
            return

        # Read input data
        pop = self.population_view.get(
            event.index,
            [COLUMNS.ANC_ATTENDANCE, COLUMNS.ANEMIA_INTERVENTION_PROPENSITY],
        )
        propensity = pop[COLUMNS.ANEMIA_INTERVENTION_PROPENSITY]

        # Subset to pop who gets ANC in later pregnancy
        attends_later_anc = pop[COLUMNS.ANC_ATTENDANCE].isin(
            [
                ANC_ATTENDANCE_TYPES.LATER_PREGNANCY_ONLY,
                ANC_ATTENDANCE_TYPES.FIRST_TRIMESTER_AND_LATER_PREGNANCY,
            ]
        )
        later_anc_idx = pop.index[attends_later_anc]

        # Determine anemia status during pregnancy
        hemoglobin = self.population_view.get(later_anc_idx, self.hemoglobin_pipeline_name)
        anemia_status = (
            pd.cut(
                hemoglobin,
                bins=[-np.inf] + ANEMIA_THRESHOLDS,
                labels=["severe", "moderate", "mild"],
                right=False,
            )
            .astype("object")
            .fillna("not_anemic")
        )
        anemia_status = anemia_status.rename(COLUMNS.ANEMIA_STATUS_DURING_PREGNANCY)
        self.population_view.update(
            COLUMNS.ANEMIA_STATUS_DURING_PREGNANCY,
            lambda _: anemia_status,
        )

        # Determine who gets hemoglobin screening
        if INTERVENTION_SCENARIOS[self.scenario].hemoglobin_screening_coverage == "baseline":
            screened_idx = later_anc_idx[
                propensity[later_anc_idx] < self.hemoglobin_screening_coverage
            ]
        else:
            # Screen all eligible simulants
            screened_idx = later_anc_idx

        hemoglobin_coverage = pd.Series(False, index=event.index)
        hemoglobin_coverage.loc[screened_idx] = True

        # Determine hemoglobin test results (low or adequate) for screened population
        tested_hemoglobin = pd.Series("not_tested", index=event.index)
        if len(screened_idx) > 0:
            true_hemoglobin_is_low = hemoglobin[screened_idx] < LOW_HEMOGLOBIN_THRESHOLD

            test_results_for_truly_low = self.randomness.choice(
                index=screened_idx[true_hemoglobin_is_low],
                choices=[HEMOGLOBIN_TEST_RESULTS.LOW, HEMOGLOBIN_TEST_RESULTS.ADEQUATE],
                p=[HEMOGLOBIN_TEST_SENSITIVITY, 1 - HEMOGLOBIN_TEST_SENSITIVITY],
                additional_key="low_hemoglobin_test_result",
            )
            test_results_for_truly_adequate = self.randomness.choice(
                index=screened_idx[~true_hemoglobin_is_low],
                choices=[HEMOGLOBIN_TEST_RESULTS.ADEQUATE, HEMOGLOBIN_TEST_RESULTS.LOW],
                p=[HEMOGLOBIN_TEST_SPECIFICITY, 1 - HEMOGLOBIN_TEST_SPECIFICITY],
                additional_key="adequate_hemoglobin_test_result",
            )
            tested_hemoglobin.loc[
                test_results_for_truly_low.index
            ] = test_results_for_truly_low
            tested_hemoglobin.loc[
                test_results_for_truly_adequate.index
            ] = test_results_for_truly_adequate

        # Determine ferritin test results for those who tested low for hemoglobin
        # (scenarios are either 0% or 100% coverage)
        tested_ferritin = pd.Series("not_tested", index=event.index)
        tested_low_idx = tested_hemoglobin[
            tested_hemoglobin == HEMOGLOBIN_TEST_RESULTS.LOW
        ].index
        if (
            INTERVENTION_SCENARIOS[self.scenario].ferritin_screening_coverage == "full"
            and len(tested_low_idx) > 0
        ):
            low_ferritin_probabilities = self.low_ferritin_probability(tested_low_idx)
            propensities = self.randomness.get_draw(
                index=tested_low_idx, additional_key="tested_ferritin"
            )
            tested_ferritin.loc[tested_low_idx] = np.where(
                propensities < low_ferritin_probabilities,
                HEMOGLOBIN_TEST_RESULTS.LOW,
                HEMOGLOBIN_TEST_RESULTS.ADEQUATE,
            )

        # Update all screening result columns
        self.population_view.update(
            [
                COLUMNS.HEMOGLOBIN_SCREENING_COVERAGE,
                COLUMNS.TESTED_HEMOGLOBIN,
                COLUMNS.TESTED_FERRITIN,
            ],
            lambda _: pd.DataFrame(
                {
                    COLUMNS.HEMOGLOBIN_SCREENING_COVERAGE: hemoglobin_coverage,
                    COLUMNS.TESTED_HEMOGLOBIN: tested_hemoglobin,
                    COLUMNS.TESTED_FERRITIN: tested_ferritin,
                },
                index=event.index,
            ),
        )
