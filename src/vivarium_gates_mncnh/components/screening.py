import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.constants.data_values import (
    ANEMIA_THRESHOLDS,
    COLUMNS,
    HEMOGLOBIN_TEST_SENSITIVITY,
    HEMOGLOBIN_TEST_SPECIFICITY,
    LOW_HEMOGLOBIN_THRESHOLD,
    PIPELINES,
    SIMULATION_EVENT_NAMES,
)
from vivarium_gates_mncnh.constants.scenarios import INTERVENTION_SCENARIOS


class AnemiaScreening(Component):
    @property
    def columns_created(self):
        return [
            COLUMNS.ANEMIA_SEVERITY,
            COLUMNS.HEMOGLOBIN_SCREENING_COVERAGE,
            COLUMNS.FERRITIN_SCREENING_COVERAGE,
            COLUMNS.TESTED_HEMOGLOBIN,
            COLUMNS.TESTED_FERRITIN,
        ]

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.scenario = builder.configuration.intervention.scenario
        self.hemoglobin_screening_coverage = builder.data.load(
            data_keys.HEMOGLOBIN.SCREENING_COVERAGE
        ).value[0]
        self.ifa_deleted_hemoglobin = builder.value.get_value(
            PIPELINES.IFA_DELETED_HEMOGLOBIN_EXPOSURE
        )

    def build_all_lookup_tables(self, builder: Builder) -> None:
        self.lookup_tables["low_ferritin_probability"] = self.build_lookup_table(
            builder=builder,
            data_source=builder.data.load(data_keys.FERRITIN.PROBABILITY_LOW_FERRITIN),
            value_columns=["value"],
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        if INTERVENTION_SCENARIOS[self.scenario].ferritin_screening_coverage == "full":
            ferritin_test_coverage = True
        else:  # 0% at baseline
            ferritin_test_coverage = False

        anemia_screening_data = pd.DataFrame(
            {
                COLUMNS.ANEMIA_SEVERITY: "not_anemic",
                COLUMNS.HEMOGLOBIN_SCREENING_COVERAGE: True,
                COLUMNS.FERRITIN_SCREENING_COVERAGE: ferritin_test_coverage,
                COLUMNS.TESTED_HEMOGLOBIN: "not_tested",
                COLUMNS.TESTED_FERRITIN: "not_tested",
            },
            index=pop_data.index,
        )

        self.population_view.update(anemia_screening_data)

    def on_time_step_cleanup(self, event: Event) -> None:
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.PREGNANCY:
            return
        pop = self.population_view.get(event.index)

        hemoglobin = self.ifa_deleted_hemoglobin(pop.index)
        pop[COLUMNS.ANEMIA_SEVERITY] = (
            pd.cut(
                hemoglobin,
                bins=[-np.inf] + ANEMIA_THRESHOLDS,
                labels=["severe", "moderate", "mild"],
                right=False,
            )
            .astype("object")
            .fillna("not_anemic")
        )

        self.population_view.update(pop)

        if INTERVENTION_SCENARIOS[self.scenario].hemoglobin_screening_coverage == "baseline":
            gets_screening = self.randomness.choice(
                index=event.index,
                choices=[True, False],
                p=[self.hemoglobin_screening_coverage],
                additional_key="hemoglobin_screening_coverage",
            )
            pop[COLUMNS.HEMOGLOBIN_SCREENING_COVERAGE] = gets_screening

        # hemoglobin screening
        screened_pop = pop.loc[pop[COLUMNS.HEMOGLOBIN_SCREENING_COVERAGE]]
        true_hemoglobin_is_low = hemoglobin[screened_pop.index] < LOW_HEMOGLOBIN_THRESHOLD

        test_results_for_truly_low = self.randomness.choice(
            index=screened_pop[true_hemoglobin_is_low].index,
            choices=["low", "adequate"],
            p=[HEMOGLOBIN_TEST_SENSITIVITY],
            additional_key="low_hemoglobin_test_result",
        )
        test_results_for_truly_adequate = self.randomness.choice(
            index=screened_pop[~true_hemoglobin_is_low].index,
            choices=["adequate", "low"],
            p=[HEMOGLOBIN_TEST_SPECIFICITY],
            additional_key="adequate_hemoglobin_test_result",
        )

        pop.loc[
            test_results_for_truly_low.index, COLUMNS.TESTED_HEMOGLOBIN
        ] = test_results_for_truly_low
        pop.loc[
            test_results_for_truly_adequate.index, COLUMNS.TESTED_HEMOGLOBIN
        ] = test_results_for_truly_adequate

        # ferritin testing
        if pop[COLUMNS.FERRITIN_SCREENING_COVERAGE].sum() != 0:
            has_anemia_idx = pop.index[pop[COLUMNS.ANEMIA_SEVERITY] != "not_anemic"]
            low_ferritin_probabilities = self.lookup_tables["low_ferritin_probability"](
                has_anemia_idx
            )
            propensities = self.randomness.get_draw(
                index=has_anemia_idx, additional_key="tested_ferritin"
            )
            pop.loc[has_anemia_idx, COLUMNS.TESTED_FERRITIN] = np.where(
                propensities < low_ferritin_probabilities, "low", "adequate"
            )

        self.population_view.update(pop)
