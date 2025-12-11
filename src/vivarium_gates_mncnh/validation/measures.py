import pandas as pd
from vivarium_testing_utils.automated_validation.data_transformation import utils
from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    SimOutputData,
    SingleNumericColumn,
)
from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    RatioMeasure,
    _align_indexes,
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

    @utils.check_io(
        numerator_data=SimOutputData,
        denominator_data=SimOutputData,
    )
    def get_ratio_datasets_from_sim(
        self,
        numerator_data: pd.DataFrame,
        denominator_data: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """Process raw simulation data and return numerator and denominator DataFrames separately."""
        numerator_data = self.numerator.format_dataset(numerator_data)
        denominator_data = self.denominator.format_dataset(denominator_data)
        numerator_data, denominator_data = _align_indexes(numerator_data, denominator_data)
        # TODO: Separate ENN deaths and LNN to have proper denominator (births in ENN and LNN)
        denominator_data = self._adjust_births_by_age_group(numerator_data, denominator_data)
        return {"numerator_data": numerator_data, "denominator_data": denominator_data}

    def _adjust_births_by_age_group(
        self, deaths: pd.DataFrame, births: pd.DataFrame
    ) -> pd.DataFrame:
        """Adjust births due to deaths in early neonatal age group to have correct population."""
        pass
