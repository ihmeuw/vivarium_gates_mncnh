import pandas as pd

from vivarium_gates_mncnh.validation.measures import NeonatalCauseSpecificMortalityRates


def test_neonatal_csmr(
    get_births_observer_data: pd.DataFrame, get_deaths_observer_data: pd.DataFrame
) -> None:
    cause = "neonatal_testing"
    measure = NeonatalCauseSpecificMortalityRates(cause)
    assert measure.measure_key == f"cause.{cause}.cause_specific_mortality_rate"
    assert measure.title == "Neonatal Testing Cause Specific Mortality Rate"
    assert measure.sim_input_datasets == {"data": measure.measure_key}
    assert measure.sim_output_datasets == {
        "numerator_data": f"{cause}_death_counts",
        "denominator_data": "births",
    }

    ratio_datasets = measure.get_ratio_datasets_from_sim(
        numerator_data=get_deaths_observer_data,
        denominator_data=get_births_observer_data,
    )
    measure_data_from_ratio = measure.get_measure_data_from_ratio(**ratio_datasets)
    expected = ratio_datasets["numerator_data"] / ratio_datasets["denominator_data"]
    pd.testing.assert_frame_equal(measure_data_from_ratio, expected)


def test_neonatal_csmr__adjust_births_by_age_group(
    get_births_observer_data: pd.DataFrame,
    get_deaths_observer_data: pd.DataFrame,
) -> None:
    cause = "neonatal_testing"
    measure = NeonatalCauseSpecificMortalityRates(cause)
    deaths = measure.numerator.format_dataset(get_deaths_observer_data)
    births = measure.denominator.format_dataset(get_births_observer_data)

    adjusted_births = measure._adjust_births_by_age_group(
        deaths=deaths,
        births=births,
    )

    # Test 1: child_age_group should be added to adjusted_births
    assert "child_age_group" in adjusted_births.index.names
    assert "child_age_group" not in births.index.names

    # Early neonatal births equals original births
    enn_mask = adjusted_births.index.get_level_values("child_age_group") == "early_neonatal"
    enn_births = adjusted_births.loc[enn_mask].droplevel("child_age_group")
    pd.testing.assert_frame_equal(enn_births, births.loc[enn_births.index])

    # Late neonatal births equals original births minus early neonatal deaths
    lnn_mask = adjusted_births.index.get_level_values("child_age_group") == "late_neonatal"
    lnn_births = adjusted_births.loc[lnn_mask].droplevel("child_age_group")
    enn_deaths = deaths.loc[
        deaths.index.get_level_values("child_age_group") == "early_neonatal"
    ].droplevel("child_age_group")
    # Expected late neonatal births = original births - early neonatal deaths
    common_index = births.index.intersection(lnn_births.index).intersection(enn_deaths.index)
    expected_lnn = births.loc[common_index] - enn_deaths.loc[common_index]
    pd.testing.assert_frame_equal(lnn_births, expected_lnn)
