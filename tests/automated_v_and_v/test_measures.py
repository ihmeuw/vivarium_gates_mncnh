import pandas as pd

from vivarium_gates_mncnh.validation.measures import NeonatalCauseSpecificMortalityRates


def test_neonatal_csmr(
    get_births_observer_data: pd.DataFrame, get_deaths_observer_data: pd.DataFrame
) -> None:
    cause = "neonatal_testing"
    measure = NeonatalCauseSpecificMortalityRates(cause)
    assert measure.title == "Neonatal Cause Specific Mortality Rate"
    assert measure.sim_input_datasets == {"data": measure.measure_key}

    ratio_datasets = measure.get_ratio_datasets_from_sim(
        numerator_data=get_deaths_observer_data,
        denominator_data=get_births_observer_data,
    )

    measure_data_from_ratio = measure.get_measure_data_from_ratio(**ratio_datasets)
    measure_data = measure.get_measure_data_from_sim(
        numerator_data=get_deaths_observer_data, denominator_data=get_births_observer_data
    )

    # TODO: compare expected dataframe to measure_data_from_ratio and measure_data


def test_neonatal_csmr__adjust_births_by_age_group(
    get_births_observer_data: pd.DataFrame,
    get_deaths_observer_data: pd.DataFrame,
) -> None:
    cause = "neonatal_testing"
    measure = NeonatalCauseSpecificMortalityRates(cause)

    # TODO: will likely need to align indexes first - the first part of get_measure_data_from_sim
    adjusted_births = measure._adjust_births_by_age_group(
        deaths=get_deaths_observer_data,
        births=get_births_observer_data,
    )
