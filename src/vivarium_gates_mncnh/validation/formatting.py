import pandas as pd
from vivarium_testing_utils.automated_validation.constants import DRAW_INDEX, SEED_INDEX
from vivarium_testing_utils.automated_validation.data_transformation import calculations
from vivarium_testing_utils.automated_validation.data_transformation.formatting import (
    SimDataFormatter,
)


class TotalLiveBirths(SimDataFormatter):
    """Formatter for simulation data that contains total live births."""

    def __init__(self) -> None:
        self.measure = "live_births"
        self.entity = "total"
        # Hardcoding raw dataset name due to custom observers
        self.raw_dataset_name = "births"
        self.unused_columns = [
            "measure",
            "entity_type",
            "entity",
            "sub_entity",
        ]
        self.filter_value = "total"
        # NOTE: Sub entity column is currently null but will have useful values at some point
        self.filters = {"sub_entity": [self.filter_value]}
        self.name = f"{self.filter_value}_{self.measure}"

    def format_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Clean up unused columns, and filter for the state."""
        cols_to_marginalize = self.unused_columns + [DRAW_INDEX, SEED_INDEX]
        return calculations.marginalize(dataset, cols_to_marginalize)
