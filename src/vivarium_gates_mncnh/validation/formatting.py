import pandas as pd
from vivarium_testing_utils.automated_validation.constants import DRAW_INDEX, SEED_INDEX
from vivarium_testing_utils.automated_validation.data_transformation.formatting import (
    SimDataFormatter,
)

from vivarium_gates_mncnh.validation.utils import map_child_index_levels


class ChildDataFormatter(SimDataFormatter):
    """Base formatter for simulation data for children data outputs."""

    def format_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        data = super().format_dataset(data)
        # Remove any subscript of "child_" on columns in data
        return map_child_index_levels(data)


class LiveBirths(ChildDataFormatter):
    """Formatter for simulation data that contains total live births."""

    def __init__(self, scenario_columns: list[str]) -> None:
        super().__init__(measure="live_births", entity="births", filter_value="live_birth")
        self.raw_dataset_name = "births"
        self.name = "live_births"
        self.filters = {"pregnancy_outcome": [self.filter_value]}
        self.scenario_columns = scenario_columns
        self.unused_columns = [
            "measure",
            "entity_type",
            "entity",
            "sub_entity",
        ]


class CauseDeaths(ChildDataFormatter):
    """Formatter for simulation data that contains deaths. This specifically handles simulation
    outputs for deaths separated by file instead of a single deaths file."""

    def __init__(self, cause: str) -> None:
        super().__init__(measure=cause, entity="death_counts", filter_value="total")
        self.name = f"{cause}_deaths"
        self.unused_columns = [
            "measure",
            "entity_type",
            "entity",
            "sub_entity",
        ]
