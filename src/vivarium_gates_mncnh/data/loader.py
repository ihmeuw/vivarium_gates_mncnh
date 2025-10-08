"""Loads, standardizes and validates input data for the simulation.

Abstract the extract and transform pieces of the artifact ETL.
The intent here is to provide a uniform interface around this portion
of artifact creation. The value of this interface shows up when more
complicated data needs are part of the project. See the BEP project
for an example.
`BEP <https://github.com/ihmeuw/vivarium_gates_bep/blob/master/src/vivarium_gates_bep/data/loader.py>`_

.. admonition::

   No logging is done here. Logging is done in vivarium inputs itself and forwarded.
"""

import pickle
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import vivarium_inputs.validation.sim as validation
from scipy.interpolate import RectBivariateSpline, griddata
from vivarium.framework.artifact import EntityKey
from vivarium_inputs import core as vi_core
from vivarium_inputs import globals as vi_globals
from vivarium_inputs import interface
from vivarium_inputs import utilities as vi_utils
from vivarium_inputs import utility_data

from vivarium_gates_mncnh.constants import data_keys, data_values, metadata, paths
from vivarium_gates_mncnh.data import extra_gbd, sampling, utilities
from vivarium_gates_mncnh.utilities import get_random_variable_draws


def get_data(
    lookup_key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame | float | str:
    """Retrieves data from an appropriate source.

    Parameters
    ----------
    lookup_key
        The key that will eventually get put in the artifact with
        the requested data.
    location
        The location to get data for.

    Returns
    -------
        The requested data.

    """
    mapping = {
        data_keys.POPULATION.LOCATION: load_population_location,
        data_keys.POPULATION.STRUCTURE: load_population_structure,
        data_keys.POPULATION.AGE_BINS: load_age_bins,
        data_keys.POPULATION.DEMOGRAPHY: load_demographic_dimensions,
        data_keys.POPULATION.TMRLE: load_theoretical_minimum_risk_life_expectancy,
        data_keys.POPULATION.SCALING_FACTOR: load_scaling_factor,
        data_keys.POPULATION.ACMR: load_standard_data,
        data_keys.POPULATION.ALL_CAUSES_MORTALITY_RISK: load_mortality_risk,
        data_keys.PREGNANCY.ASFR: load_asfr,
        data_keys.PREGNANCY.SBR: load_sbr,
        data_keys.PREGNANCY.RAW_INCIDENCE_RATE_MISCARRIAGE: load_raw_incidence_data,
        data_keys.PREGNANCY.RAW_INCIDENCE_RATE_ECTOPIC: load_raw_incidence_data,
        data_keys.LBWSG.DISTRIBUTION: load_metadata,
        data_keys.LBWSG.CATEGORIES: load_metadata,
        data_keys.LBWSG.SEX_SPECIFIC_ORDERED_CATEGORIES: load_sex_specific_ordered_lbwsg_categories,
        data_keys.LBWSG.BIRTH_EXPOSURE: load_lbwsg_birth_exposure,
        data_keys.LBWSG.EXPOSURE: load_lbwsg_exposure,
        data_keys.LBWSG.RELATIVE_RISK: load_lbwsg_rr,
        data_keys.LBWSG.RELATIVE_RISK_INTERPOLATOR: load_lbwsg_interpolated_rr,
        data_keys.LBWSG.PAF: load_paf_data,
        data_keys.ANC.ANCfirst: load_anc_proportion,
        data_keys.ANC.ANC1: load_anc_proportion,
        data_keys.ANC.ANC4: load_anc_proportion,
        data_keys.MATERNAL_SEPSIS.RAW_INCIDENCE_RATE: load_standard_data,
        data_keys.MATERNAL_SEPSIS.CSMR: load_standard_data,
        data_keys.MATERNAL_SEPSIS.YLD_RATE: load_maternal_disorder_yld_rate,
        data_keys.MATERNAL_HEMORRHAGE.RAW_INCIDENCE_RATE: load_standard_data,
        data_keys.MATERNAL_HEMORRHAGE.CSMR: load_standard_data,
        data_keys.MATERNAL_HEMORRHAGE.YLD_RATE: load_maternal_disorder_yld_rate,
        data_keys.OBSTRUCTED_LABOR.RAW_INCIDENCE_RATE: load_standard_data,
        data_keys.OBSTRUCTED_LABOR.CSMR: load_standard_data,
        data_keys.OBSTRUCTED_LABOR.YLD_RATE: load_maternal_disorder_yld_rate,
        data_keys.PRETERM_BIRTH.PAF: load_paf_data,
        data_keys.PRETERM_BIRTH.PREVALENCE: load_preterm_prevalence,
        data_keys.PRETERM_BIRTH.MORTALITY_RISK: load_mortality_risk,
        data_keys.NEONATAL_SEPSIS.MORTALITY_RISK: load_mortality_risk,
        data_keys.NEONATAL_ENCEPHALOPATHY.MORTALITY_RISK: load_mortality_risk,
        data_keys.FACILITY_CHOICE.IN_FACILITY_DELIVERY_PROPORTION: load_facility_proportion,
        data_keys.FACILITY_CHOICE.P_HOME: load_probability_home_delivery,
        data_keys.FACILITY_CHOICE.P_BEmONC: load_overall_probability_birth_facility_type,
        data_keys.FACILITY_CHOICE.P_CEmONC: load_overall_probability_birth_facility_type,
        # data_keys.FACILITY_CHOICE.P_HOME_PRETERM: load_probability_birth_facility_type,
        # data_keys.FACILITY_CHOICE.P_HOME_FULL_TERM: load_probability_birth_facility_type,
        data_keys.FACILITY_CHOICE.BEmONC_FACILITY_FRACTION: load_probability_birth_facility_type,
        data_keys.NO_CPAP_RISK.P_RDS: load_p_rds,
        data_keys.NO_CPAP_RISK.P_CPAP_HOME: load_cpap_facility_access_probability,
        data_keys.NO_CPAP_RISK.P_CPAP_BEMONC: load_cpap_facility_access_probability,
        data_keys.NO_CPAP_RISK.P_CPAP_CEMONC: load_cpap_facility_access_probability,
        data_keys.NO_CPAP_RISK.RELATIVE_RISK: load_no_cpap_relative_risk,
        data_keys.NO_CPAP_RISK.PAF: load_no_cpap_paf,
        data_keys.NO_ACS_RISK.RELATIVE_RISK: load_no_acs_relative_risk,
        data_keys.NO_ACS_RISK.PAF: load_no_acs_paf,
        data_keys.NO_ANTIBIOTICS_RISK.P_ANTIBIOTIC_HOME: load_antibiotic_coverage_probability,
        data_keys.NO_ANTIBIOTICS_RISK.P_ANTIBIOTIC_BEMONC: load_antibiotic_coverage_probability,
        data_keys.NO_ANTIBIOTICS_RISK.P_ANTIBIOTIC_CEMONC: load_antibiotic_coverage_probability,
        data_keys.NO_ANTIBIOTICS_RISK.RELATIVE_RISK: load_no_antibiotics_relative_risk,
        data_keys.NO_ANTIBIOTICS_RISK.PAF: load_no_antibiotics_paf,
        data_keys.NO_PROBIOTICS_RISK.P_PROBIOTIC_HOME: load_probiotics_facility_probability,
        data_keys.NO_PROBIOTICS_RISK.P_PROBIOTIC_BEMONC: load_probiotics_facility_probability,
        data_keys.NO_PROBIOTICS_RISK.P_PROBIOTIC_CEMONC: load_probiotics_facility_probability,
        data_keys.NO_PROBIOTICS_RISK.RELATIVE_RISK: load_no_probiotics_relative_risk,
        data_keys.NO_PROBIOTICS_RISK.PAF: load_no_probiotics_paf,
        data_keys.NO_AZITHROMYCIN_RISK.P_AZITHROMYCIN_HOME: load_azithromycin_facility_probability,
        data_keys.NO_AZITHROMYCIN_RISK.P_AZITHROMYCIN_BEMONC: load_azithromycin_facility_probability,
        data_keys.NO_AZITHROMYCIN_RISK.P_AZITHROMYCIN_CEMONC: load_azithromycin_facility_probability,
        data_keys.NO_AZITHROMYCIN_RISK.RELATIVE_RISK: load_no_azithromycin_relative_risk,
        data_keys.NO_AZITHROMYCIN_RISK.PAF: load_no_azithromycin_paf,
        data_keys.NO_MISOPROSTOL_RISK.P_MISOPROSTOL_HOME: load_misoprostol_coverage_probability,
        data_keys.NO_MISOPROSTOL_RISK.P_MISOPROSTOL_BEMONC: load_misoprostol_coverage_probability,
        data_keys.NO_MISOPROSTOL_RISK.P_MISOPROSTOL_CEMONC: load_misoprostol_coverage_probability,
        data_keys.NO_MISOPROSTOL_RISK.RELATIVE_RISK: load_no_misoprostol_relative_risk,
        data_keys.NO_MISOPROSTOL_RISK.PAF: load_no_misoprostol_paf,
        data_keys.IFA_SUPPLEMENTATION.COVERAGE: load_ifa_coverage,
        # data_keys.IFA_SUPPLEMENTATION.EFFECT_SIZE: load_oral_iron_effect_size,
        # data_keys.IFA_SUPPLEMENTATION.EXCESS_SHIFT: load_ifa_excess_shift,
        # data_keys.IFA_SUPPLEMENTATION.RISK_SPECIFIC_SHIFT: load_risk_specific_shift,
        # data_keys.MMN_SUPPLEMENTATION.EFFECT_SIZE: load_oral_iron_effect_size,
        data_keys.MMN_SUPPLEMENTATION.STILLBIRTH_RR: load_oral_iron_effect_size,
        # data_keys.MMN_SUPPLEMENTATION.EXCESS_SHIFT: load_mms_excess_shift,
        # data_keys.MMN_SUPPLEMENTATION.EXCESS_GA_SHIFT_SUBPOP_1: load_excess_gestational_age_shift,
        # data_keys.MMN_SUPPLEMENTATION.EXCESS_GA_SHIFT_SUBPOP_2: load_excess_gestational_age_shift,
        # data_keys.MMN_SUPPLEMENTATION.RISK_SPECIFIC_SHIFT: load_risk_specific_shift,
        # data_keys.POSTPARTUM_DEPRESSION.INCIDENCE_RISK: load_postpartum_depression_raw_incidence_risk,
        # data_keys.POSTPARTUM_DEPRESSION.CASE_FATALITY_RATE: load_postpartum_depression_case_fatality_rate,
        # data_keys.POSTPARTUM_DEPRESSION.CASE_DURATION: load_postpartum_depression_case_duration,
        # data_keys.POSTPARTUM_DEPRESSION.CASE_SEVERITY: load_postpartum_depression_case_severity,
        # data_keys.POSTPARTUM_DEPRESSION.DISABILITY_WEIGHT: load_postpartum_depression_disability_weight,
        data_keys.HEMOGLOBIN.EXPOSURE: load_hemoglobin_exposure_data,
        data_keys.HEMOGLOBIN.STANDARD_DEVIATION: load_hemoglobin_exposure_data,
        data_keys.HEMOGLOBIN.DISTRIBUTION_WEIGHTS: load_hemoglobin_distribution_weights,
        data_keys.HEMOGLOBIN.DISTRIBUTION: load_hemoglobin_distribution,
        data_keys.HEMOGLOBIN.RELATIVE_RISK: load_hemoglobin_relative_risk,
        # data_keys.HEMOGLOBIN.PAF: load_hemoglobin_paf,
        data_keys.HEMOGLOBIN.TMRED: load_hemoglobin_tmred,
        data_keys.HEMOGLOBIN.SCREENING_COVERAGE: load_hemoglobin_screening_coverage,
        data_keys.IV_IRON.HEMOGLOBIN_EFFECT_SIZE: load_iv_iron_hemoglobin_effect_size,
        # data_keys.PROPENSITY_CORRELATIONS.PROPENSITY_CORRELATIONS: load_propensity_correlations,
        # data_keys.FERRITIN.PROBABILITY_LOW_FERRITIN: load_probability_low_ferritin,
    }

    data = mapping[lookup_key](lookup_key, location, years)
    to_remap = utilities.determine_if_remap_group(lookup_key)
    if to_remap and isinstance(data, pd.DataFrame):
        data = utilities.rename_child_data_index_names(data)
    return data


def load_population_location(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> str:
    if key != data_keys.POPULATION.LOCATION:
        raise ValueError(f"Unrecognized key {key}")

    return location


def load_population_structure(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    return interface.get_population_structure(location, years)


def load_age_bins(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    return interface.get_age_bins()


def load_demographic_dimensions(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    return interface.get_demographic_dimensions(location, years)


def load_theoretical_minimum_risk_life_expectancy(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    return interface.get_theoretical_minimum_risk_life_expectancy()


def load_standard_data(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    key = EntityKey(key)
    entity = utilities.get_entity(key)
    return interface.get_measure(entity, key.measure, location, years).droplevel("location")


def load_metadata(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
):
    key = EntityKey(key)
    entity = utilities.get_entity(key)
    entity_metadata = entity[key.measure]
    if hasattr(entity_metadata, "to_dict"):
        entity_metadata = entity_metadata.to_dict()
    return entity_metadata


# TODO - add project-specific data functions here


def load_asfr(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    asfr = load_standard_data(key, location)
    asfr = asfr.reset_index()
    asfr_pivot = asfr.pivot(
        index=[col for col in metadata.ARTIFACT_INDEX_COLUMNS if col != "location"],
        columns="parameter",
        values="value",
    )
    seed = f"{key}_{location}"
    asfr_draws = sampling.generate_vectorized_lognormal_draws(asfr_pivot, seed)
    return asfr_draws


def load_sbr(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    year_start, year_end = metadata.ARTIFACT_YEAR_START, metadata.ARTIFACT_YEAR_END

    sbr = load_standard_data(key, location)
    lower_value = sbr.loc[(year_start, year_end, "lower_value"), "value"]
    mean_value = sbr.loc[(year_start, year_end, "mean_value"), "value"]
    upper_value = sbr.loc[(year_start, year_end, "upper_value"), "value"]
    sbr = sbr.reorder_levels(["parameter", "year_start", "year_end"]).loc["mean_value"]

    sbr_dist = sampling.get_truncnorm_from_quantiles(
        mean=mean_value, lower=lower_value, upper=upper_value
    )
    sbr_draws = get_random_variable_draws(metadata.ARTIFACT_COLUMNS, key, sbr_dist)
    sbr_draws = sbr_draws.values.flatten()
    # Ensure shape is (1, NUM_DRAWS)
    sbr_draws = sbr_draws.reshape(1, -1)

    draw_columns = [f"draw_{i:d}" for i in range(data_values.NUM_DRAWS)]
    sbr_draws_df = pd.DataFrame(sbr_draws, columns=draw_columns)
    sbr_draws_df["year_start"] = year_start
    sbr_draws_df["year_end"] = year_end
    return sbr_draws_df.set_index(["year_start", "year_end"])


# TODO: remove now that we are pulling from GBD 2023
def load_raw_incidence_data(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    """Temporary function to short circuit around validation issues in Vivarium Inputs"""
    key = EntityKey(key)
    entity = utilities.get_entity(key)
    data_type = vi_utils.DataType(key.measure, "draws")
    data = vi_core.get_data(entity, key.measure, location, years, data_type)
    data = vi_utils.scrub_gbd_conventions(data, location)
    validation.validate_for_simulation(
        data, entity, "incidence_rate", location, years, data_type.value_columns
    )
    data = vi_utils.split_interval(data, interval_column="age", split_column_prefix="age")
    data = vi_utils.split_interval(data, interval_column="year", split_column_prefix="year")
    return vi_utils.sort_hierarchical_data(data).droplevel("location")


def load_scaling_factor(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:

    incidence_c995 = get_data(
        data_keys.PREGNANCY.RAW_INCIDENCE_RATE_MISCARRIAGE, location, years
    ).reset_index()
    incidence_c374 = get_data(
        data_keys.PREGNANCY.RAW_INCIDENCE_RATE_ECTOPIC, location, years
    ).reset_index()
    asfr = get_data(data_keys.PREGNANCY.ASFR, location).reset_index()
    sbr = get_data(data_keys.PREGNANCY.SBR, location).reset_index()

    asfr = asfr.set_index(metadata.ARTIFACT_INDEX_COLUMNS)
    incidence_c995 = incidence_c995.set_index(metadata.ARTIFACT_INDEX_COLUMNS)
    incidence_c374 = incidence_c374.set_index(metadata.ARTIFACT_INDEX_COLUMNS)
    sbr = (
        sbr.set_index("year_start")
        .drop(columns=["year_end"])
        .reindex(asfr.index, level="year_start")
    )

    # Calculate pregnancy incidence
    preg_inc = asfr + asfr * sbr + incidence_c995 + incidence_c374

    return preg_inc


def load_anc_proportion(
    key: str, location: str, years: Optional[Union[int, str, list[int]]] = None
) -> pd.DataFrame:
    if key == data_keys.ANC.ANCfirst:
        data = pd.read_csv(paths.ANC_DATA_DIR / "anc_first.csv")
        location_id = utility_data.get_location_id(location)
        data = data.loc[data["location_id"] == location_id]
        data = data.loc[data["year_id"] == metadata.ARTIFACT_YEAR_START].rename(
            {"year_id": "year_start"}, axis=1
        )
        data["year_end"] = metadata.ARTIFACT_YEAR_END
        data = data.drop(
            ["age_group_id", "sex_id", "location_id", "mean", "lower", "upper"], axis=1
        )
        data = data.set_index(["year_start", "year_end"])
        return data
    elif key == data_keys.ANC.ANC1 or key == data_keys.ANC.ANC4:
        anc_proportion = load_standard_data(key, location, years)
        year_start, year_end = metadata.ARTIFACT_YEAR_START, metadata.ARTIFACT_YEAR_END
        lower_value = anc_proportion.loc[(year_start, year_end, "lower_value"), "value"]
        mean_value = anc_proportion.loc[(year_start, year_end, "mean_value"), "value"]
        upper_value = anc_proportion.loc[(year_start, year_end, "upper_value"), "value"]

        try:
            anc_proportion_dist = sampling.get_truncnorm_from_quantiles(
                mean=mean_value, lower=lower_value, upper=upper_value
            )
            anc_proportion_draws = get_random_variable_draws(
                metadata.ARTIFACT_COLUMNS, key, anc_proportion_dist
            )
            anc_proportion_draws = anc_proportion_draws.values.flatten()
            # Ensure shape is (1, NUM_DRAWS)
            anc_proportion_draws = anc_proportion_draws.reshape(1, -1)
        except FloatingPointError:
            print("FloatingPointError encountered, proceeding with caution.")
            anc_proportion_draws = np.full((1, data_values.NUM_DRAWS), mean_value)

        draw_columns = [f"draw_{i:d}" for i in range(data_values.NUM_DRAWS)]
        anc_proportion_draws_df = pd.DataFrame(anc_proportion_draws, columns=draw_columns)
        anc_proportion_draws_df["year_start"] = year_start
        anc_proportion_draws_df["year_end"] = year_end
        return anc_proportion_draws_df.set_index(["year_start", "year_end"])
    else:
        raise ValueError(f"Unrecognized key {key} when loading ANC proportion data.")


def load_maternal_disorder_yld_rate(
    key: str, location: str, years: Optional[Union[int, str, list[int]]] = None
) -> pd.DataFrame:

    groupby_cols = ["age_group_id", "sex_id", "year_id"]
    draw_cols = vi_globals.DRAW_COLUMNS
    yld_rate = extra_gbd.get_maternal_disorder_yld_rate(key, location)
    yld_rate = yld_rate[groupby_cols + draw_cols]
    yld_rate = reshape_to_vivarium_format(yld_rate, location)

    return yld_rate


def load_lbwsg_rr(
    key: str, location: str, years: Optional[Union[int, str, list[int]]] = None
) -> pd.DataFrame:
    if key != data_keys.LBWSG.RELATIVE_RISK:
        raise ValueError(f"Unrecognized key {key}")

    data = load_standard_data(key, location, years)
    data = data.query(f"year_start == {metadata.ARTIFACT_YEAR_START}").droplevel(
        ["affected_entity", "affected_measure"]
    )
    data = data[~data.index.duplicated()]
    caps = pd.read_csv(paths.LBWSG_RR_CAPS_DIR / f"{location.lower()}.csv")
    caps = caps.set_index(data.index.names)
    neonatal_data = data.query("age_start < 8 / 365")
    capped_neonatal_data = neonatal_data.where(neonatal_data <= caps, other=caps)
    data.loc[neonatal_data.index] = capped_neonatal_data
    return data


def load_lbwsg_interpolated_rr(
    key: str, location: str, years: Optional[Union[int, str, list[int]]]
) -> pd.DataFrame:
    if key != data_keys.LBWSG.RELATIVE_RISK_INTERPOLATOR:
        raise ValueError(f"Unrecognized key {key}")

    rr = get_data(data_keys.LBWSG.RELATIVE_RISK, location).reset_index()
    rr["parameter"] = pd.Categorical(
        rr["parameter"], [f"cat{i}" for i in range(metadata.DRAW_COUNT)]
    )
    rr = (
        rr.sort_values("parameter")
        .set_index(metadata.CHILDREN_INDEX_COLUMNS + ["parameter"])
        .stack()
        .unstack("parameter")
        .apply(np.log)
    )

    # get category midpoints
    def get_category_midpoints(lbwsg_type: str) -> pd.Series:
        categories = get_data(f"risk_factor.{data_keys.LBWSG.name}.categories", location)
        return utilities.get_intervals_from_categories(lbwsg_type, categories).apply(
            lambda x: x.mid
        )

    gestational_age_midpoints = get_category_midpoints("short_gestation")
    birth_weight_midpoints = get_category_midpoints("low_birth_weight")

    # build grid of gestational age and birth weight
    def get_grid(midpoints: pd.Series, endpoints: tuple[float, float]) -> np.array:
        grid = np.append(np.unique(midpoints), endpoints)
        grid.sort()
        return grid

    gestational_age_grid = get_grid(gestational_age_midpoints, (0.0, 42.0))
    birth_weight_grid = get_grid(birth_weight_midpoints, (0.0, 4500.0))

    def make_interpolator(log_rr_for_age_sex_draw: pd.Series) -> RectBivariateSpline:
        # Use scipy.interpolate.griddata to extrapolate to grid using nearest neighbor interpolation
        log_rr_grid_nearest = griddata(
            (gestational_age_midpoints, birth_weight_midpoints),
            log_rr_for_age_sex_draw,
            (gestational_age_grid[:, None], birth_weight_grid[None, :]),
            method="nearest",
            rescale=True,
        )
        # return a RectBivariateSpline object from the extrapolated values on grid
        return RectBivariateSpline(
            gestational_age_grid, birth_weight_grid, log_rr_grid_nearest, kx=1, ky=1
        )

    log_rr_interpolator = (
        rr.apply(make_interpolator, axis="columns")
        .apply(lambda x: pickle.dumps(x).hex())
        .unstack()
    )
    return log_rr_interpolator


def load_paf_data(
    key: str, location: str, years: Optional[Union[int, str, list[int]]]
) -> pd.DataFrame:
    if key == data_keys.LBWSG.PAF:
        filename = "calculated_lbwsg_paf_on_cause.all_causes.all_cause_mortality_risk.parquet"
    else:
        filename = "calculated_lbwsg_paf_on_cause.all_causes.all_cause_mortality_risk_preterm.parquet"

    output_dir = paths.PAF_DIR / location.lower()

    df = pd.read_parquet(output_dir / filename)
    if "input_draw" in df.columns:
        df = df.assign(input_draw="draw_" + df.input_draw.astype(str))
    else:
        df = df.assign(input_draw="draw_0")
    df = df.pivot_table(
        "value", [c for c in df if c not in ["input_draw", "value"]], "input_draw"
    ).reset_index()
    not_needed_columns = ["scenario", "random_seed"]
    df = df.drop(columns=[c for c in df.columns if c in not_needed_columns])

    age_start_dict = {
        "early_neonatal": data_values.EARLY_NEONATAL_AGE_START,
        "late_neonatal": data_values.LATE_NEONATAL_AGE_START,
    }
    age_end_dict = {
        "early_neonatal": data_values.LATE_NEONATAL_AGE_START,
        "late_neonatal": data_values.LATE_NEONATAL_AGE_END,
    }
    df["age_start"] = df["child_age_group"].replace(age_start_dict)
    df["age_end"] = df["child_age_group"].replace(age_end_dict)
    df["year_start"] = metadata.ARTIFACT_YEAR_START
    df["year_end"] = metadata.ARTIFACT_YEAR_END
    df = df.drop("child_age_group", axis=1)
    df = df.rename(columns={"child_sex": "sex"})
    index_columns = [
        "sex",
        "age_start",
        "age_end",
        "year_start",
        "year_end",
    ]
    df = df.set_index(index_columns)
    unaffected_age_groups = [(data_values.LATE_NEONATAL_AGE_END, 1.0), (1.0, 5.0)]
    for age_start, age_end in unaffected_age_groups:
        for sex in ["Male", "Female"]:
            df.loc[
                (
                    sex,
                    age_start,
                    age_end,
                    metadata.ARTIFACT_YEAR_START,
                    metadata.ARTIFACT_YEAR_END,
                ),
                :,
            ] = 0

    return df.sort_index()


def load_p_rds(
    lookup_key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> float:
    csmr = get_data(data_keys.PRETERM_BIRTH.MORTALITY_RISK, location, years)
    p_rds = csmr * data_values.PRETERM_DEATHS_DUE_TO_RDS_PROBABILITY
    return p_rds


def load_facility_proportion(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> float:
    df = load_standard_data(key, location)
    df = df.query("parameter=='mean_value'")
    if len(df) > 1:
        max_year = df.index.get_level_values("year_start").max()
        df = df.query("year_start==@max_year")
    return df.squeeze()


def load_probability_home_delivery(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> float:
    facility_proportion = get_data(
        data_keys.FACILITY_CHOICE.IN_FACILITY_DELIVERY_PROPORTION, location
    )
    return 1 - facility_proportion


def load_overall_probability_birth_facility_type(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> float:
    in_facility_delivery_proportion = get_data(
        data_keys.FACILITY_CHOICE.IN_FACILITY_DELIVERY_PROPORTION, location
    )
    bemonc_facility_fraction = get_data(
        data_keys.FACILITY_CHOICE.BEmONC_FACILITY_FRACTION, location
    )

    if key == data_keys.FACILITY_CHOICE.P_BEmONC:
        return in_facility_delivery_proportion * bemonc_facility_fraction
    elif key == data_keys.FACILITY_CHOICE.P_CEmONC:
        return in_facility_delivery_proportion * (1 - bemonc_facility_fraction)
    else:
        raise ValueError(f"Unrecognized key {key}")


def load_probability_birth_facility_type(
    lookup_key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> float:
    return data_values.DELIVERY_FACILITY_TYPE_PROBABILITIES[location][lookup_key]


def load_cpap_facility_access_probability(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> float:
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)
    facility_uniform_dist = data_values.CPAP_ACCESS_PROBABILITIES[location][key]
    draws = get_random_variable_draws(metadata.ARTIFACT_COLUMNS, key, facility_uniform_dist)
    data = pd.DataFrame([draws], columns=metadata.ARTIFACT_COLUMNS, index=demography.index)
    data.index = data.index.droplevel("location")

    return utilities.set_non_neonnatal_values(data, 0.0)


def load_no_cpap_relative_risk(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> float:
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)
    rr_distribution = data_values.CPAP_RELATIVE_RISK_DISTRIBUTION
    draws = get_random_variable_draws(metadata.ARTIFACT_COLUMNS, key, rr_distribution)
    data = pd.DataFrame([draws], columns=metadata.ARTIFACT_COLUMNS, index=demography.index)
    data.index = data.index.droplevel("location")

    # We want the relative risk of no cpap
    no_cpap_rr = (1 / data).fillna(0.0)
    return utilities.set_non_neonnatal_values(no_cpap_rr, 1.0)


def load_no_cpap_paf(
    lookup_key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> float:
    p_CPAP_BEmONC = get_data(data_keys.NO_CPAP_RISK.P_CPAP_BEMONC, location, years)
    p_CPAP_CEmONC = get_data(data_keys.NO_CPAP_RISK.P_CPAP_CEMONC, location, years)
    rr_no_CPAP = get_data(data_keys.NO_CPAP_RISK.RELATIVE_RISK, location, years)

    # get p_CPAP and p_no_CPAP
    p_BEmONC_given_facility = data_values.DELIVERY_FACILITY_TYPE_PROBABILITIES[location][
        data_values.FACILITY_CHOICE.BEmONC_FACILITY_FRACTION
    ]
    p_CEmONC_given_facility = 1 - p_BEmONC_given_facility

    p_BEmONC = get_data(data_keys.FACILITY_CHOICE.P_BEmONC, location)
    p_CEmONC = get_data(data_keys.FACILITY_CHOICE.P_CEmONC, location)

    # marginalize over facility (no CPAP at home)
    p_CPAP = (p_CPAP_BEmONC * p_BEmONC) + (p_CPAP_CEmONC * p_CEmONC)
    p_no_CPAP = 1 - p_CPAP

    population_average_rr = rr_no_CPAP * p_no_CPAP + 1 * p_CPAP
    paf_no_CPAP = 1 - 1 / population_average_rr

    return paf_no_CPAP


def load_no_acs_relative_risk(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> float:
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)
    rr_distribution = data_values.ACS_RELATIVE_RISK_DISTRIBUTION
    draws = get_random_variable_draws(metadata.ARTIFACT_COLUMNS, key, rr_distribution)
    data = pd.DataFrame([draws], columns=metadata.ARTIFACT_COLUMNS, index=demography.index)
    data.index = data.index.droplevel("location")

    # We want the relative risk of no cpap
    no_acs_rr = (1 / data).fillna(0.0)
    return utilities.set_non_neonnatal_values(no_acs_rr, 1.0)


def load_no_acs_paf(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> float:
    p_CPAP_BEmONC = get_data(data_keys.NO_CPAP_RISK.P_CPAP_BEMONC, location, years)
    p_CPAP_CEmONC = get_data(data_keys.NO_CPAP_RISK.P_CPAP_CEMONC, location, years)
    p_CPAP_home = get_data(data_keys.NO_CPAP_RISK.P_CPAP_HOME, location, years)
    rr_no_CPAP = get_data(data_keys.NO_CPAP_RISK.RELATIVE_RISK, location, years)
    rr_no_ACS = get_data(data_keys.NO_ACS_RISK.RELATIVE_RISK, location, years)
    p_home = get_data(data_keys.FACILITY_CHOICE.P_HOME, location, years)
    p_BEmONC = get_data(data_keys.FACILITY_CHOICE.P_BEmONC, location, years)
    p_CEmONC = get_data(data_keys.FACILITY_CHOICE.P_CEmONC, location, years)

    p_CPAP = (p_CPAP_BEmONC * p_BEmONC) + (p_CPAP_CEmONC * p_CEmONC) + (p_CPAP_home * p_home)

    p_intervention = p_CPAP
    p_no_intervention = 1 - p_intervention

    population_average_RR = p_no_intervention * rr_no_CPAP * rr_no_ACS + p_intervention * 1
    paf_no_CPAP_ACS = 1 - 1 / population_average_RR

    return paf_no_CPAP_ACS


def load_sex_specific_ordered_lbwsg_categories(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> dict[str, list[str]]:
    rrs = get_data(data_keys.LBWSG.RELATIVE_RISK, location).query("child_age_start==0.0")
    rrs = rrs.mean(axis=1)
    categories = get_data(data_keys.LBWSG.CATEGORIES, location)
    # Get preterm categories
    preterm_cats, full_term_cats = [], []
    for cat, description in categories.items():
        i = utilities.parse_short_gestation_description(description)
        if i.right <= metadata.PRETERM_AGE_CUTOFF:
            preterm_cats.append(cat)
        else:
            full_term_cats.append(cat)

    ordered_cats = {}
    for sex in ["Male", "Female"]:
        sex_specific_rrs = (
            pd.DataFrame(rrs).query("sex_of_child==@sex").rename({0: "value"}, axis=1)
        )
        # sort so earlier categories have higher RRs
        preterm_rrs = sex_specific_rrs.query("parameter == @preterm_cats")
        sorted_preterm_cats = preterm_rrs.sort_values(
            by="value", ascending=False
        ).reset_index()["parameter"]
        full_term_rrs = sex_specific_rrs.query("parameter == @full_term_cats")
        sorted_full_term_cats = full_term_rrs.sort_values(
            by="value", ascending=False
        ).reset_index()["parameter"]
        # all preterm before any full term
        sorted_cats = list(sorted_preterm_cats) + list(sorted_full_term_cats)
        ordered_cats[sex] = sorted_cats

    return ordered_cats


def load_lbwsg_birth_exposure(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:

    if key != data_keys.LBWSG.BIRTH_EXPOSURE:
        raise ValueError(f"Unrecognized key {key}")

    birth_exposure = extra_gbd.load_lbwsg_birth_exposure(location)
    # This category was a mistake in GBD 2019, so drop.
    exposure_key = "risk_factor.low_birth_weight_and_short_gestation.exposure"
    entity = utilities.get_entity(exposure_key)
    extra_residual_category = vi_globals.EXTRA_RESIDUAL_CATEGORY[entity.name]
    birth_exposure = birth_exposure.loc[
        birth_exposure["parameter"] != extra_residual_category
    ]
    idx_cols = ["location_id", "age_group_id", "year_id", "sex_id", "parameter"]
    birth_exposure = birth_exposure.set_index(idx_cols)[vi_globals.DRAW_COLUMNS]

    # Sometimes there are data values on the order of 10e-300 that cause
    # floating point headaches, so clip everything to reasonable values
    birth_exposure = birth_exposure.clip(lower=vi_globals.MINIMUM_EXPOSURE_VALUE)

    # normalize so all categories sum to 1
    total_exposure = birth_exposure.groupby(
        ["location_id", "age_group_id", "sex_id"]
    ).transform("sum")
    birth_exposure = (
        (birth_exposure / total_exposure).reset_index().drop(columns=["age_group_id"])
    )
    birth_exposure = reshape_to_vivarium_format(birth_exposure, location)

    return birth_exposure


def load_lbwsg_exposure(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:

    # Get exposure for all age groups except birth age group
    exposure = load_standard_data(key, location, years)
    # Sometimes there are data values on the order of 10e-300 that cause
    # floating point headaches, so clip everything to reasonable values
    exposure = exposure.clip(lower=vi_globals.MINIMUM_EXPOSURE_VALUE)

    # normalize so all categories sum to 1
    total_exposure = exposure.groupby(["age_start", "age_end", "sex"]).transform("sum")
    exposure = (
        (exposure / total_exposure)
        .reset_index()
        .set_index(metadata.ARTIFACT_INDEX_COLUMNS + ["parameter"])
        .sort_index()
    )
    return exposure


def load_preterm_prevalence(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    # get early neonatal prevalence from GBD
    exposure = get_data(data_keys.LBWSG.BIRTH_EXPOSURE, location, years).reset_index()
    categories = get_data(data_keys.LBWSG.CATEGORIES, location, years)
    # Get preterm categories
    preterm_cats = []
    for cat, description in categories.items():
        i = utilities.parse_short_gestation_description(description)
        if i.right <= metadata.PRETERM_AGE_CUTOFF:
            preterm_cats.append(cat)

    # Subset exposure to preterm categories
    preterm_exposure = exposure.loc[exposure["parameter"].isin(preterm_cats)]
    preterm_exposure = preterm_exposure.drop(columns=["parameter"])
    draw_cols = [col for col in preterm_exposure.columns if "draw" in col]
    sum_exposure = preterm_exposure.groupby(["sex_of_child", "year_start", "year_end"])[
        draw_cols
    ].sum()

    enn_data = sum_exposure.reset_index()
    enn_data["child_age_start"] = data_values.EARLY_NEONATAL_AGE_START
    enn_data["child_age_end"] = data_values.LATE_NEONATAL_AGE_START

    # get late neonatal prevalence from our PAF simulation
    filename = "calculated_late_neonatal_preterm_prevalence.parquet"
    filepath = paths.PRETERM_PREVALENCE_DIR / location.lower() / filename
    data = pd.read_parquet(filepath)
    data = data.drop(columns=[c for c in ["scenario", "random_seed"] if c in data.columns])
    if "input_draw" in data.columns:
        data = data.pivot(index="child_sex", columns="input_draw", values="value")
    else:
        # Treat value as draw 0, like with PAFs
        data = data.rename(columns={"value": 0}).set_index("child_sex")
    data.columns = [f"draw_{i}" for i in data.columns]

    lnn_data = data.reset_index().rename({"child_sex": "sex_of_child"}, axis=1)
    lnn_data["child_age_start"] = data_values.LATE_NEONATAL_AGE_START
    lnn_data["child_age_end"] = data_values.LATE_NEONATAL_AGE_END
    lnn_data["year_start"] = enn_data["year_start"]
    lnn_data["year_end"] = enn_data["year_end"]
    lnn_data = lnn_data[[c for c in lnn_data.columns if c in enn_data.columns]]

    df = pd.concat([enn_data, lnn_data], ignore_index=True)
    df = df.sort_values(metadata.CHILDREN_INDEX_COLUMNS).set_index(
        metadata.CHILDREN_INDEX_COLUMNS
    )

    return df


def load_antibiotic_coverage_probability(
    key: str,
    location: str,
    years: Optional[Union[int, str, List[int]]] = None,
) -> pd.DataFrame:
    # Model 8.3 sets coverage values at population level and not birth facility/location level
    coverage_dict = {
        "Ethiopia": 0.5,
        "Nigeria": 0.0,
        "Pakistan": 0.0,
    }
    return coverage_dict[location]


def load_no_antibiotics_relative_risk(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    rr_dist = data_values.ANTIBIOTIC_RELATIVE_RISK_DISTRIBUTION
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)
    draws = get_random_variable_draws(metadata.ARTIFACT_COLUMNS, key, rr_dist)
    data = pd.DataFrame([draws], columns=metadata.ARTIFACT_COLUMNS, index=demography.index)
    data.index = data.index.droplevel("location")
    # Update to distribution for model 8.3 requires inverting the rrs
    data = (1 / data).fillna(0.0)

    return utilities.set_non_neonnatal_values(data, 1.0)


def load_iv_iron_hemoglobin_effect_size(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    effect_size_dist = data_values.IV_IRON_HEMOGLOBIN_EFFECT_SIZE[location]
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)
    draws = get_random_variable_draws(metadata.ARTIFACT_COLUMNS, key, effect_size_dist)
    data = pd.DataFrame([draws], columns=metadata.ARTIFACT_COLUMNS)
    return data


def load_no_antibiotics_paf(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    relative_risk = get_data(data_keys.NO_ANTIBIOTICS_RISK.RELATIVE_RISK, location, years)
    # Only location specific coverage now
    p_antibiotics_coverage = get_data(
        data_keys.NO_ANTIBIOTICS_RISK.P_ANTIBIOTIC_HOME, location, years
    )
    # mean_rr = rr_no_antibiotics * (1 - p_antibiotics_coverage) + p_antibiotics_coverage
    mean_rr = relative_risk * (1 - p_antibiotics_coverage) + p_antibiotics_coverage
    # paf = (mean_rr - 1) / mean_rr
    paf = (mean_rr - 1) / mean_rr
    return paf


def load_probiotics_facility_probability(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> float:
    # Currently no coverage
    return data_values.PROBIOTICS_BASELINE_COVERAGE_PROABILITY


def load_no_probiotics_relative_risk(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    # Relative risk for no receieving intervention
    rr_dist = data_values.PROBIOTICS_RELATIVE_RISK_DISTRIBUTION
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)
    draws = get_random_variable_draws(metadata.ARTIFACT_COLUMNS, key, rr_dist)
    data = pd.DataFrame([draws], columns=metadata.ARTIFACT_COLUMNS, index=demography.index)
    data.index = data.index.droplevel("location")
    # model 8.2 requires inverting the relative risks since the distribution has been updated
    data = (1 / data).fillna(0.0)

    return utilities.set_non_neonnatal_values(data, 1.0)


def load_no_probiotics_paf(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:

    # Get all no_cpap data for calculations
    csmr = get_data(data_keys.NEONATAL_SEPSIS.MORTALITY_RISK, location, years)
    p_sepsis = csmr.copy()
    p_home = get_data(data_keys.FACILITY_CHOICE.P_HOME, location, years)
    p_BEmONC = get_data(data_keys.FACILITY_CHOICE.P_BEmONC, location, years)
    p_CEmONC = get_data(data_keys.FACILITY_CHOICE.P_CEmONC, location, years)
    p_probiotic_home = get_data(
        data_keys.NO_PROBIOTICS_RISK.P_PROBIOTIC_HOME, location, years
    )
    p_probiotic_BEmONC = get_data(
        data_keys.NO_PROBIOTICS_RISK.P_PROBIOTIC_BEMONC, location, years
    )
    p_probiotic_CEmONC = get_data(
        data_keys.NO_PROBIOTICS_RISK.P_PROBIOTIC_CEMONC, location, years
    )
    relative_risk = get_data(data_keys.NO_PROBIOTICS_RISK.RELATIVE_RISK, location, years)
    # This is derived in the CPAP PAF calculation
    p_sepsis_probiotic = p_sepsis / (
        (p_home * (1 - p_probiotic_home) * relative_risk)
        + (p_home * p_probiotic_home)
        + (p_BEmONC * (1 - p_probiotic_BEmONC) * relative_risk)
        + (p_CEmONC * (1 - p_probiotic_CEmONC) * relative_risk)
        + (p_BEmONC * p_probiotic_BEmONC)
        + (p_CEmONC * p_probiotic_CEmONC)
    )
    paf_no_probiotic = 1 - (p_sepsis_probiotic / p_sepsis)
    paf_no_probiotic = paf_no_probiotic.fillna(0.0)

    return utilities.set_non_neonnatal_values(paf_no_probiotic, 0.0)


def load_mortality_risk(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    if key == data_keys.POPULATION.ALL_CAUSES_MORTALITY_RISK:
        gbd_id = 294  # All causes neonatal mortality
    else:
        entity = utilities.get_entity(key)
        gbd_id = entity.gbd_id
    draw_columns = [f"draw_{i:d}" for i in range(data_values.NUM_DRAWS)]
    # Get birth counts
    births = extra_gbd.get_birth_counts(location)
    births = vi_utils.scrub_gbd_conventions(births, location)
    births = vi_utils.split_interval(
        births, interval_column="year", split_column_prefix="year"
    )
    births.index = births.index.droplevel("location")
    # Pull early and late neonatal death counts
    def get_deaths(age_group_id, gbd_id):
        deaths = extra_gbd.get_mortality_death_counts(
            location=location, age_group_id=age_group_id, gbd_id=gbd_id
        )
        deaths = deaths.set_index(["location_id", "sex_id", "age_group_id", "year_id"])[
            draw_columns
        ]
        deaths = reshape_to_vivarium_format(deaths, location)
        return deaths

    # Early neonatal deaths (all-cause and cause-specific)
    # and cause-specific late neonatal deaths
    enn_acmr_deaths = get_deaths(age_group_id=2, gbd_id=294)
    enn_deaths = get_deaths(age_group_id=2, gbd_id=gbd_id)
    lnn_deaths = get_deaths(age_group_id=3, gbd_id=gbd_id)

    # Build mortality risk dataframe
    enn = enn_deaths.merge(births, left_index=True, right_index=True)
    enn_mortality_risk = enn.filter(like="draw").div(enn.population, axis=0)
    # Get denominator for late neonatal mortality risk
    population_array = np.array(enn["population"]).reshape(-1, 1)
    denominator = population_array - enn_acmr_deaths[draw_columns]
    denominator = denominator.droplevel(["age_start", "age_end"])
    lnn_mortality_risk = lnn_deaths / denominator
    mortality_risk = pd.concat([enn_mortality_risk, lnn_mortality_risk]).reorder_levels(
        ["sex", "age_start", "age_end", "year_start", "year_end"]
    )
    return mortality_risk


def load_azithromycin_facility_probability(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> float:
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)
    facility_uniform_dist = data_values.AZITHROMYCIN_FACILITY_TYPE_DISTRIBUTION[location][key]
    draws = get_random_variable_draws(metadata.ARTIFACT_COLUMNS, key, facility_uniform_dist)
    data = pd.DataFrame([draws], columns=metadata.ARTIFACT_COLUMNS, index=demography.index)
    data.index = data.index.droplevel("location")

    return data


def load_no_azithromycin_relative_risk(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    # Relative risk for not receiving intervention
    rr_dist = data_values.AZITHROMYCIN_RELATIVE_RISK_DISTRIBUTION
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)
    draws = get_random_variable_draws(metadata.ARTIFACT_COLUMNS, key, rr_dist)
    data = pd.DataFrame([draws], columns=metadata.ARTIFACT_COLUMNS, index=demography.index)
    data.index = data.index.droplevel("location")

    return data


def load_no_azithromycin_paf(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    # Get all required data for calculations
    incidence_rate = get_data(data_keys.MATERNAL_SEPSIS.RAW_INCIDENCE_RATE, location, years)
    p_sepsis = incidence_rate.copy()
    p_home = get_data(data_keys.FACILITY_CHOICE.P_HOME, location, years)
    p_BEmONC = get_data(data_keys.FACILITY_CHOICE.P_BEmONC, location, years)
    p_CEmONC = get_data(data_keys.FACILITY_CHOICE.P_CEmONC, location, years)
    p_azith_home = get_data(
        data_keys.NO_AZITHROMYCIN_RISK.P_AZITHROMYCIN_HOME, location, years
    )
    p_azith_BEmONC = get_data(
        data_keys.NO_AZITHROMYCIN_RISK.P_AZITHROMYCIN_BEMONC, location, years
    )
    p_azith_CEmONC = get_data(
        data_keys.NO_AZITHROMYCIN_RISK.P_AZITHROMYCIN_CEMONC, location, years
    )
    # Relative risk of no azithromycin
    relative_risk = get_data(data_keys.NO_AZITHROMYCIN_RISK.RELATIVE_RISK, location, years)
    # This is derived in the CPAP PAF calculation
    p_sepsis_azith = p_sepsis / (
        (p_home * (1 - p_azith_home) * relative_risk)
        + (p_home * p_azith_home)
        + (p_BEmONC * (1 - p_azith_BEmONC) * relative_risk)
        + (p_CEmONC * (1 - p_azith_CEmONC) * relative_risk)
        + (p_BEmONC * p_azith_BEmONC)
        + (p_CEmONC * p_azith_CEmONC)
    )
    paf_no_azith = 1 - (p_sepsis_azith / p_sepsis)
    paf_no_azith = paf_no_azith.fillna(0.0)

    return paf_no_azith


def load_misoprostol_coverage_probability(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> float:
    # Mean coverate is 0. For the intervention component we have delivery facility specific coverage
    # values but this is at a location level and only at home births that received ANC are eligible
    coverage_dict = {
        "Ethiopia": 0.0,
        "Nigeria": 0.0,
        "Pakistan": 0.0,
    }
    return coverage_dict[location]


def load_no_misoprostol_relative_risk(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    # Relative risk for not receiving intervention
    rr_dist = data_values.MISOPROSTOL_RELATIVE_RISK_DISTRIBUTION
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)
    draws = get_random_variable_draws(metadata.ARTIFACT_COLUMNS, key, rr_dist)
    data = pd.DataFrame([draws], columns=metadata.ARTIFACT_COLUMNS, index=demography.index)
    data.index = data.index.droplevel("location")
    # Need to invert the relative risk to be a risk of no intervention
    data = (1 / data).fillna(0.0)

    return data


def load_no_misoprostol_paf(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    relative_risk = get_data(data_keys.NO_MISOPROSTOL_RISK.RELATIVE_RISK, location, years)
    # Only location specific coverage
    p_misoprostol_coverage = get_data(
        data_keys.NO_MISOPROSTOL_RISK.P_MISOPROSTOL_HOME, location, years
    )
    # mean_rr = rr_no_intervention * (1 - p_intervention_coverage) + p_intervention_coverage
    mean_rr = relative_risk * (1 - p_misoprostol_coverage) + p_misoprostol_coverage
    # paf = (mean_rr - 1) / mean_rr
    paf = (mean_rr - 1) / mean_rr
    return paf


def load_ifa_coverage(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    filepath = paths.ORAL_IRON_DATA_DIR / "anc_iron_prop_st.csv"
    return load_coverage_from_file(filepath, location)


def load_hemoglobin_screening_coverage(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    filepath = (
        paths.J_DIR / "anc_bloodsample_prop_st-gpr_results_aggregates_scaled2025-05-29.csv"
    )
    return load_coverage_from_file(filepath, location)


def load_coverage_from_file(filepath: Path, location: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df.query("year_id==2023")
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)
    child_demography = demography.query("age_end <= 5").droplevel("location")
    location_id = utility_data.get_location_id(location)
    df = df.query("location_id==@location_id")
    df = df[[f"draw_{i}" for i in range(vi_globals.NUM_DRAWS)]].reset_index(drop=True)
    # duplicate rows for each row in child demography
    cat2_rows = pd.concat([df] * len(child_demography), ignore_index=True)
    cat2_rows.index = child_demography.index
    # cat2 is exposed to intervention
    cat2_rows["parameter"] = "cat2"
    cat1_rows = 1 - cat2_rows.drop(columns=["parameter"])
    cat1_rows["parameter"] = "cat1"
    data = pd.concat([cat2_rows, cat1_rows])
    return data.set_index("parameter", append=True).sort_index()


def load_oral_iron_effect_size(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    effect_size_dists = data_values.ORAL_IRON_EFFECT_SIZES[key]
    effect_size_data = []

    for target, dist in effect_size_dists.items():
        draws = get_random_variable_draws(metadata.ARTIFACT_COLUMNS, key, dist)
        data = pd.DataFrame([draws], columns=metadata.ARTIFACT_COLUMNS)
        data["affected_target"] = target
        data = data.set_index("affected_target")
        effect_size_data.append(data)

    return pd.concat(effect_size_data)


def load_ifa_excess_shift(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    birth_weight_shift = load_mms_excess_shift(key, location)
    gestational_age_shift = load_excess_gestational_age_shift(key, location)
    all_ages_data = pd.concat([birth_weight_shift, gestational_age_shift])
    return all_ages_data.query("age_end <= 5.0")


def load_risk_specific_shift(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    try:
        key_group = {
            data_keys.IFA_SUPPLEMENTATION.RISK_SPECIFIC_SHIFT: data_keys.IFA_SUPPLEMENTATION,
            data_keys.MMN_SUPPLEMENTATION.RISK_SPECIFIC_SHIFT: data_keys.MMN_SUPPLEMENTATION,
        }[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")

    if key.name == "multiple_micronutrient_supplementation":
        excess_shift = get_data(key_group.EXCESS_SHIFT, location)
        single_cat_shift = excess_shift.query("parameter=='cat2'").droplevel("parameter")
        risk_specific_shift = pd.DataFrame(
            0.0, columns=single_cat_shift.columns, index=single_cat_shift.index
        )
    else:
        exposure = get_data(key_group.COVERAGE, location)
        excess_shift = get_data(key_group.EXCESS_SHIFT, location)

        risk_specific_shift = (
            (exposure * excess_shift)
            .groupby(
                metadata.ARTIFACT_INDEX_COLUMNS + ["affected_entity", "affected_measure"]
            )
            .sum()
        )

    return risk_specific_shift


def load_mms_excess_shift(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    try:
        distribution = {
            data_keys.IFA_SUPPLEMENTATION.EXCESS_SHIFT: data_values.ORAL_IRON_EFFECT_SIZES[
                data_keys.IFA_SUPPLEMENTATION.EFFECT_SIZE
            ]["birth_weight.birth_exposure"],
            data_keys.MMN_SUPPLEMENTATION.EXCESS_SHIFT: data_values.ORAL_IRON_EFFECT_SIZES[
                data_keys.MMN_SUPPLEMENTATION.EFFECT_SIZE
            ]["birth_weight.birth_exposure"],
        }[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")
    distribution_data = (key, distribution)
    data = load_dichotomous_excess_shift(location, distribution_data)
    return data.query("age_end <= 5").droplevel("location")


def load_dichotomous_excess_shift(
    location: str,
    distribution_data: tuple,
) -> pd.DataFrame:
    """Load excess birth weight exposure shifts using distribution data."""
    index = get_data(data_keys.POPULATION.DEMOGRAPHY, location).index
    shift = get_random_variable_draws(metadata.ARTIFACT_COLUMNS, *distribution_data)
    excess_shift = reshape_shift_data(
        shift, index, data_values.PIPELINES.BIRTH_WEIGHT_EXPOSURE
    )

    return excess_shift


def reshape_shift_data(shift: pd.Series, index: pd.Index, target: str) -> pd.DataFrame:
    """Read in draw-level shift values and return a DataFrame where the data are the shift values,
    and the index is the passed index appended with affected entity/measure and parameter data.
    """
    exposed = pd.DataFrame([shift], index=index)
    exposed["parameter"] = "cat2"
    unexposed = pd.DataFrame([pd.Series(0.0, index=metadata.ARTIFACT_COLUMNS)], index=index)
    unexposed["parameter"] = "cat1"

    excess_shift = pd.concat([exposed, unexposed])
    excess_shift["affected_entity"] = target.split(".")[0]
    excess_shift["affected_measure"] = target.split(".")[1]

    excess_shift = excess_shift.set_index(
        ["affected_entity", "affected_measure", "parameter"], append=True
    ).sort_index()
    return excess_shift


def load_excess_gestational_age_shift(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    """Load excess gestational age shift data from IFA and MMS from file.
    Returns the sum of the shift data in the directories defined in data_dirs."""
    try:
        data_dirs = {
            data_keys.IFA_SUPPLEMENTATION.EXCESS_SHIFT: [paths.IFA_GA_SHIFT_DATA_DIR],
            data_keys.MMN_SUPPLEMENTATION.EXCESS_GA_SHIFT_SUBPOP_1: [
                paths.MMS_GA_SHIFT_1_DATA_DIR
            ],
            data_keys.MMN_SUPPLEMENTATION.EXCESS_GA_SHIFT_SUBPOP_2: [
                paths.MMS_GA_SHIFT_1_DATA_DIR,
                paths.MMS_GA_SHIFT_2_DATA_DIR,
            ],
        }[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")

    index = get_data(data_keys.POPULATION.DEMOGRAPHY, location).index
    all_shift_data = [
        pd.read_csv(data_dir / f"{location.lower()}.csv") for data_dir in data_dirs
    ]
    shifts = [
        pd.Series(shift_data["value"].values, index=shift_data["draw"])
        for shift_data in all_shift_data
    ]
    if len(shifts) > 1:
        shifts[1] = shifts[1].loc[shifts[1].notnull()]
    summed_shifts = sum(shifts)  # only sum more than one Series for subpop 2

    excess_shift = reshape_shift_data(
        summed_shifts, index, data_values.PIPELINES.GESTATIONAL_AGE_EXPOSURE
    )
    excess_shift = excess_shift[metadata.ARTIFACT_COLUMNS]
    return excess_shift.query("age_end <= 5.0").droplevel("location")


def load_postpartum_depression_raw_incidence_risk(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    dist = data_values.POSTPARTUM_DEPRESSION_INCIDENCE_RISK
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)
    draws = get_random_variable_draws(metadata.ARTIFACT_COLUMNS, key, dist)
    data = pd.DataFrame([draws], columns=metadata.ARTIFACT_COLUMNS, index=demography.index)
    data.index = data.index.droplevel("location")

    return data


def load_postpartum_depression_case_fatality_rate(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    # YLD only disorder
    return 0


def load_postpartum_depression_case_duration(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    dist = data_values.POSTPARTUM_DEPRESSION_CASE_DURATION
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)
    draws = get_random_variable_draws(metadata.ARTIFACT_COLUMNS, key, dist)
    data = pd.DataFrame([draws], columns=metadata.ARTIFACT_COLUMNS, index=demography.index)
    data.index = data.index.droplevel("location")

    return data


def load_postpartum_depression_case_severity(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    return data_values.POSTPARTUM_DEPRESSION_CASE_SEVERITY_PROBABILITIES


def load_postpartum_depression_disability_weight(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    disability_weights_dict = (
        data_values.POSTPARTUM_DEPRESSION_CASE_SEVERITY_DISABILITY_WEIGHTS
    )
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)

    disability_weights = []
    for case_type, dist in disability_weights_dict.items():
        draws = get_random_variable_draws(
            metadata.ARTIFACT_COLUMNS, f"{key}_{case_type}", dist
        )
        data = pd.DataFrame(
            [draws], columns=metadata.ARTIFACT_COLUMNS, index=demography.index
        )
        data.index = data.index.droplevel("location")
        data[data_values.COLUMNS.POSTPARTUM_DEPRESSION_CASE_TYPE] = case_type
        data = data.set_index(
            data_values.COLUMNS.POSTPARTUM_DEPRESSION_CASE_TYPE, append=True
        )
        disability_weights.append(data)

    # Each item in the list is a dataframe with our demographic index + the case type so we do not
    # need to create these distributions on the fly during the simulation
    disability_weights = pd.concat(disability_weights)

    return disability_weights


def load_hemoglobin_exposure_data(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
):
    hemoglobin_data = extra_gbd.get_hemoglobin_exposure_data(key, location)
    hemoglobin_data = reshape_to_vivarium_format(hemoglobin_data, location)
    levels_to_drop = [
        "measure_id",
        "metric_id",
        "model_version_id",
        "modelable_entity_id",
        "rei_id",
    ]
    if key == data_keys.HEMOGLOBIN.EXPOSURE:
        levels_to_drop.append("parameter")
    hemoglobin_data.index = hemoglobin_data.index.droplevel(levels_to_drop)

    # Expand draw columns from 0-99 to 0-499 by repeating 5 times
    expanded_draws_df = utilities.expand_draw_columns(
        hemoglobin_data, num_draws=100, num_repeats=5
    )

    return expanded_draws_df


def load_hemoglobin_distribution_weights(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
):
    weight_values = data_values.HEMOGLOBIN_ENSEMBLE_DISTRIBUTION_WEIGHTS
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)
    data = pd.DataFrame(data=weight_values)
    weights = pd.concat([data] * len(demography), ignore_index=True)
    idx = demography.index.repeat(len(data))
    weights = weights.set_index(idx)
    weights = weights.set_index("parameter", append=True)
    weights.index = weights.index.droplevel("location")

    return weights


def load_hemoglobin_distribution(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> str:
    return data_values.HEMOGLOBIN_DISTRIBUTION


def load_hemoglobin_relative_risk(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
):
    hemoglobin_data = extra_gbd.get_hemoglobin_rr_data(key, location)
    hemoglobin_data = reshape_to_vivarium_format(hemoglobin_data, location)
    levels_to_drop = [
        "metric_id",
        "model_version_id",
        "modelable_entity_id",
        "rei_id",
    ]
    hemoglobin_data.index = hemoglobin_data.index.droplevel(levels_to_drop)

    # hemoglobin RR-specific processing
    hemoglobin_data = hemoglobin_data.reset_index()
    hemoglobin_data["parameter"] = hemoglobin_data["exposure"]
    hemoglobin_data["affected_measure"] = "incidence_risk"
    hemoglobin_data = hemoglobin_data.drop(["exposure", "morbidity", "mortality"], axis=1)
    hemoglobin_data = vi_utils.convert_affected_entity(hemoglobin_data, "cause_id")
    index_cols = metadata.ARTIFACT_INDEX_COLUMNS + [
        "affected_entity",
        "affected_measure",
        "parameter",
    ]
    hemoglobin_data = hemoglobin_data.set_index(index_cols)

    # TOOD: should we expand to 250 instead the same way?
    # Expand draw columns from 0-99 to 0-499 by repeating 5 times
    expanded_draws_df = utilities.expand_draw_columns(
        hemoglobin_data, num_draws=100, num_repeats=5
    )

    return expanded_draws_df


def load_hemoglobin_paf(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
):
    hemoglobin_data = extra_gbd.get_hemoglobin_paf_data(key, location)
    hemoglobin_data = reshape_to_vivarium_format(hemoglobin_data, location)
    levels_to_drop = ["metric_id", "measure_id", "rei_id", "version_id"]
    hemoglobin_data.index = hemoglobin_data.index.droplevel(levels_to_drop)

    # hemoglobin PAF-specific processing
    hemoglobin_data = hemoglobin_data.reset_index()
    # we are pulling PAF data for deaths to define incidence risk
    hemoglobin_data["affected_measure"] = "incidence_risk"
    hemoglobin_data = vi_utils.convert_affected_entity(hemoglobin_data, "cause_id")
    index_cols = metadata.ARTIFACT_INDEX_COLUMNS + ["affected_entity", "affected_measure"]
    hemoglobin_data = hemoglobin_data.set_index(index_cols)

    # Expand draw columns from 0-99 to 0-499 by repeating 5 times
    expanded_draws_df = utilities.expand_draw_columns(
        hemoglobin_data, num_draws=100, num_repeats=5
    )

    return expanded_draws_df


def load_hemoglobin_tmred(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> dict[str, str | bool | float]:
    return {"distribution": "uniform", "min": 120.0, "max": 120.0}


def load_propensity_correlations(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> None:
    return data_values.PROPENSITY_CORRELATIONS[location]


def load_probability_low_ferritin(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
):
    df = pd.read_csv(
        paths.FERRITIN_TESTING_COVERAGE_DATA_DIR / "iron_responsive_fraction.csv"
    )
    df = df.drop("Unnamed: 0", axis=1)
    location_id = utility_data.get_location_id(location)
    # TODO: fix this once data has been updated
    location_id = 169 if location_id == 179 else location_id
    df = df.query("location_id==@location_id").drop("location_id", axis=1)
    df = df.rename(
        {"anemia_severity": data_values.COLUMNS.ANEMIA_STATUS_DURING_PREGNANCY}, axis=1
    )
    return reshape_to_vivarium_format(df, location)


def reshape_to_vivarium_format(df, location):
    df = vi_utils.reshape(df, value_cols=vi_globals.DRAW_COLUMNS)
    df = vi_utils.scrub_gbd_conventions(df, location)
    df = vi_utils.split_interval(df, interval_column="age", split_column_prefix="age")
    df = vi_utils.split_interval(df, interval_column="year", split_column_prefix="year")
    df = vi_utils.sort_hierarchical_data(df)
    df.index = df.index.droplevel("location")
    return df
