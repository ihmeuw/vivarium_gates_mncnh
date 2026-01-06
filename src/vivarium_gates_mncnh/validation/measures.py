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
from vivarium_gates_mncnh.validation.utils import map_child_index_levels


class NeonatalCauseSpecificMortalityRisk(RatioMeasure):
    """Computes cause-specific mortality rate in the population."""

    @property
    def rate_aggregation_weights(self) -> RateAggregationWeights:
        """Returns rate aggregated weights."""
        return RateAggregationWeights(
            weight_keys={
                "adjusted_births": f"cause.{self.entity}.adjusted_birth_counts",
            },
            formula=lambda adjusted_births: map_child_index_levels(adjusted_births),
            description="Beginning of age group population, births adjusted for early neonatal deaths",
        )

    def __init__(self, cause: str) -> None:
        super().__init__(
            entity_type="cause",
            entity=cause,
            measure="mortality_risk",
            numerator=CauseDeaths(cause),
            denominator=LiveBirths([]),
        )

    @utils.check_io(data=SingleNumericColumn, out=SingleNumericColumn)
    def get_measure_data_from_sim_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        return map_child_index_levels(data)

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
        # Separate ENN deaths and LNN to have proper denominator (births in ENN and LNN)
        denominator_data = self._adjust_births_by_age_group(numerator_data, denominator_data)
        numerator_data, denominator_data = _align_indexes(numerator_data, denominator_data)

        return {"numerator_data": numerator_data, "denominator_data": denominator_data}

    def _adjust_births_by_age_group(
        self, deaths: pd.DataFrame, births: pd.DataFrame
    ) -> pd.DataFrame:
        """Adjust births due to deaths in early neonatal age group to have correct population.
        This function does the following two things:
        1. Adds child_age_group index level to births DataFrame to match deaths DataFrame.
        2. Updates the births dataframe so that the deaths from the early neonatal age group have
        been subtracted from the births of the late neonatal age group."""

        age_group_values = {"age_group": deaths.index.get_level_values("age_group").unique()}
        # Cast age groups onto births
        births = births.reindex(
            pd.MultiIndex.from_product(
                [
                    births.index.get_level_values(level).unique()
                    for level in births.index.names
                ]
                + list(age_group_values.values()),
                names=list(births.index.names) + list(age_group_values.keys()),
            )
        )

        # Subtract early neonatal deaths from late neonatal births
        enn_deaths = (
            deaths.loc[deaths.index.get_level_values("age_group") == "early_neonatal"]
            .droplevel("age_group")
            .values
        )
        lnn_mask = births.index.get_level_values("age_group") == "late_neonatal"
        births.loc[lnn_mask] -= enn_deaths
        return births
