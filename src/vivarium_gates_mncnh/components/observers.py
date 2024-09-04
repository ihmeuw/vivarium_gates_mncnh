from typing import Any

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.results import Observer
from vivarium_public_health.results import COLUMNS, PublicHealthObserver
from vivarium_public_health.results import ResultsStratifier as ResultsStratifier_

from vivarium_gates_mncnh.constants.data_values import PREGNANCY_OUTCOMES


class ResultsStratifier(ResultsStratifier_):
    def setup(self, builder: Builder) -> None:
        self.age_bins = self.get_age_bins(builder)
        self.start_year = builder.configuration.time.start.year
        self.end_year = builder.configuration.time.end.year

        self.register_stratifications(builder)

    def register_stratifications(self, builder: Builder) -> None:
        super().register_stratifications(builder)

        builder.results.register_stratification(
            "pregnancy_outcome",
            list(
                PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME,
                PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME,
                PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME,
            ),
            requires_columns=["pregnancy_outcome"],
        )
