import pandas as pd
from vivarium_testing_utils.automated_validation.data_transformation import utils
from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    SingleNumericColumn,
)
from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    RatioMeasure,
)
from vivarium_testing_utils.automated_validation.data_transformation.rate_aggregation import (
    RateAggregationWeights,
)

from vivarium_gates_mncnh.validation.formatting import CauseDeaths, LiveBirths


class NeonatalCauseSpecificMortalityRates(RatioMeasure):
    """Computes cause-specific mortality rate in the population."""

    @property
    def rate_aggregation_weights(self) -> RateAggregationWeights:
        """Returns rate aggregated weights."""
        raise NotImplementedError  # MIC-6675

    def __init__(self, cause: str) -> None:
        super().__init__(
            entity_type="cause",
            entity=cause,
            measure="cause_specific_mortality_rate",
            numerator=CauseDeaths(cause),
            # TODO: handle denominator, need to handle calculation across age group for deaths/births
            denominator=LiveBirths([]),
        )

    @utils.check_io(data=SingleNumericColumn, out=SingleNumericColumn)
    def get_measure_data_from_sim_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError  # MIC-6675
