import pandas as pd
from vivarium_testing_utils.automated_validation.constants import DRAW_INDEX, SEED_INDEX
from vivarium_testing_utils.automated_validation.data_transformation import calculations
from vivarium_testing_utils.automated_validation.data_transformation.formatting import (
    TotalPopulationPersonTime,
)


class TotalLiveBirths(TotalPopulationPersonTime):
    """Formatter for simulation data that contains total live births."""

    def __init__(self, scenario_columns: list[str]) -> None:
        self.measure = "live_births"
        self.entity = "total"
        self.raw_dataset_name = "births"
        self.name = "total_live_births"
        self.filter_value = "total"
        self.filters = {"sub_entity": [self.filter_value]}
        self.scenario_columns = ["child_sex"] + scenario_columns
        self.unused_columns = [
            "measure",
            "entity_type",
            "entity",
            "sub_entity",
        ]

    def format_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # Filter to only live births
        dataset = calculations.filter_data(dataset, {"pregnancy_outcome": ["live_birth"]})
        return super().format_dataset(dataset)
