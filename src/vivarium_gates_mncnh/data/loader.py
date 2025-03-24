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
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import vivarium_inputs.validation.sim as validation
from gbd_mapping import causes, covariates, risk_factors
from scipy.interpolate import RectBivariateSpline, griddata
from vivarium.framework.artifact import EntityKey
from vivarium_gbd_access import gbd
from vivarium_inputs import core as vi_core
from vivarium_inputs import globals as vi_globals
from vivarium_inputs import interface
from vivarium_inputs import utilities as vi_utils
from vivarium_inputs import utility_data
from vivarium_inputs.mapping_extension import alternative_risk_factors

from vivarium_gates_mncnh.constants import data_keys, data_values, metadata, paths
from vivarium_gates_mncnh.data import extra_gbd, sampling, utilities


def get_data(
    lookup_key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
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
        # TODO - add appropriate mappings
        data_keys.PREGNANCY.ASFR: load_asfr,
        data_keys.PREGNANCY.SBR: load_sbr,
        data_keys.PREGNANCY.RAW_INCIDENCE_RATE_MISCARRIAGE: load_raw_incidence_data,
        data_keys.PREGNANCY.RAW_INCIDENCE_RATE_ECTOPIC: load_raw_incidence_data,
        data_keys.LBWSG.DISTRIBUTION: load_metadata,
        data_keys.LBWSG.CATEGORIES: load_metadata,
        data_keys.LBWSG.BIRTH_EXPOSURE: load_lbwsg_exposure,
        data_keys.LBWSG.RELATIVE_RISK: load_lbwsg_rr,
        data_keys.LBWSG.RELATIVE_RISK_INTERPOLATOR: load_lbwsg_interpolated_rr,
        data_keys.LBWSG.PAF: load_paf_data,
        data_keys.ANC.ESTIMATE: load_anc_proportion,
        data_keys.MATERNAL_SEPSIS.RAW_INCIDENCE_RATE: load_standard_data,
        data_keys.MATERNAL_SEPSIS.CSMR: load_standard_data,
        data_keys.MATERNAL_SEPSIS.YLD_RATE: load_maternal_disorder_yld_rate,
        data_keys.MATERNAL_HEMORRHAGE.RAW_INCIDENCE_RATE: load_standard_data,
        data_keys.MATERNAL_HEMORRHAGE.CSMR: load_standard_data,
        data_keys.MATERNAL_HEMORRHAGE.YLD_RATE: load_maternal_disorder_yld_rate,
        data_keys.OBSTRUCTED_LABOR.RAW_INCIDENCE_RATE: load_standard_data,
        data_keys.OBSTRUCTED_LABOR.CSMR: load_standard_data,
        data_keys.OBSTRUCTED_LABOR.YLD_RATE: load_maternal_disorder_yld_rate,
        data_keys.PRETERM_BIRTH.CSMR: load_standard_data,
        data_keys.PRETERM_BIRTH.PAF: load_paf_data,
        data_keys.PRETERM_BIRTH.PREVALENCE: load_preterm_prevalence,
        data_keys.NEONATAL_SEPSIS.CSMR: load_standard_data,
        data_keys.NEONATAL_ENCEPHALOPATHY.CSMR: load_standard_data,
        data_keys.FACILITY_CHOICE.P_HOME: load_probability_birth_facility_type,
        data_keys.FACILITY_CHOICE.P_BEmONC: load_probability_birth_facility_type,
        data_keys.FACILITY_CHOICE.P_CEmONC: load_probability_birth_facility_type,
        data_keys.NO_CPAP_RISK.P_RDS: load_p_rds,
        data_keys.NO_CPAP_RISK.P_CPAP_HOME: load_cpap_facility_access_probability,
        data_keys.NO_CPAP_RISK.P_CPAP_BEmONC: load_cpap_facility_access_probability,
        data_keys.NO_CPAP_RISK.P_CPAP_CEmONC: load_cpap_facility_access_probability,
        data_keys.NO_CPAP_RISK.RELATIVE_RISK: load_no_cpap_relative_risk,
        data_keys.NO_CPAP_RISK.PAF: load_no_cpap_paf,
        data_keys.NO_ANTIBIOTICS_RISK.P_ANTIBIOTIC_HOME: load_antibiotic_facility_probability,
        data_keys.NO_ANTIBIOTICS_RISK.P_ANTIBIOTIC_BEmONC: load_antibiotic_facility_probability,
        data_keys.NO_ANTIBIOTICS_RISK.P_ANTIBIOTIC_CEmONC: load_antibiotic_facility_probability,
        data_keys.NO_ANTIBIOTICS_RISK.RELATIVE_RISK: load_no_antibiotics_relative_risk,
        data_keys.NO_ANTIBIOTICS_RISK.PAF: load_no_antibiotics_paf,
    }
    return mapping[lookup_key](lookup_key, location, years)


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


def load_categorical_paf(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    try:
        risk = {
            # todo add keys as needed
            data_keys.KEYGROUP.PAF: data_keys.KEYGROUP,
        }[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")

    distribution_type = get_data(risk.DISTRIBUTION, location)

    if distribution_type != "dichotomous" and "polytomous" not in distribution_type:
        raise NotImplementedError(
            f"Unrecognized distribution {distribution_type} for {risk.name}. Only dichotomous and "
            f"polytomous are recognized categorical distributions."
        )

    exp = get_data(risk.EXPOSURE, location)
    rr = get_data(risk.RELATIVE_RISK, location)

    # paf = (sum_categories(exp * rr) - 1) / sum_categories(exp * rr)
    sum_exp_x_rr = (
        (exp * rr)
        .groupby(list(set(rr.index.names) - {"parameter"}))
        .sum()
        .reset_index()
        .set_index(rr.index.names[:-1])
    )
    paf = (sum_exp_x_rr - 1) / sum_exp_x_rr
    return paf


def _load_em_from_meid(location, meid, measure):
    location_id = utility_data.get_location_id(location)
    data = gbd.get_modelable_entity_draws(meid, location_id)
    data = data[data.measure_id == vi_globals.MEASURES[measure]]
    data = vi_utils.normalize(data, fill_value=0)
    data = data.filter(vi_globals.DEMOGRAPHIC_COLUMNS + vi_globals.DRAW_COLUMNS)
    data = vi_utils.reshape(data)
    data = vi_utils.scrub_gbd_conventions(data, location)
    data = vi_utils.split_interval(data, interval_column="age", split_column_prefix="age")
    data = vi_utils.split_interval(data, interval_column="year", split_column_prefix="year")
    return vi_utils.sort_hierarchical_data(data).droplevel("location")


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
    sbr = load_standard_data(key, location)
    sbr = sbr.reorder_levels(["parameter", "year_start", "year_end"]).loc["mean_value"]
    return sbr


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
    preg_inc = asfr + asfr.multiply(sbr["value"], axis=0) + incidence_c995 + incidence_c374

    return preg_inc


def load_anc_proportion(
    key: str, location: str, years: Optional[Union[int, str, list[int]]] = None
) -> pd.DataFrame:
    anc_proportion = load_standard_data(key, location, years)
    year_start, year_end = 2021, 2022
    lower_value = anc_proportion.loc[(year_start, year_end, "lower_value"), "value"]
    mean_value = anc_proportion.loc[(year_start, year_end, "mean_value"), "value"]
    upper_value = anc_proportion.loc[(year_start, year_end, "upper_value"), "value"]

    try:
        anc_proportion_dist = sampling.get_truncnorm_from_quantiles(
            mean=mean_value, lower=lower_value, upper=upper_value
        )
        anc_proportion_draws = anc_proportion_dist.rvs(data_values.NUM_DRAWS).reshape(
            1, data_values.NUM_DRAWS
        )
    except FloatingPointError:
        print("FloatingPointError encountered, proceeding with caution.")
        anc_proportion_draws = np.full((1, data_values.NUM_DRAWS), mean_value)

    draw_columns = [f"draw_{i:d}" for i in range(data_values.NUM_DRAWS)]
    anc_proportion_draws_df = pd.DataFrame(anc_proportion_draws, columns=draw_columns)
    anc_proportion_draws_df["year_start"] = year_start
    anc_proportion_draws_df["year_end"] = year_end
    return anc_proportion_draws_df.set_index(["year_start", "year_end"])


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
    data = data.query("year_start == 2021").droplevel(["affected_entity", "affected_measure"])
    data = data[~data.index.duplicated()]
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
        .set_index(metadata.ARTIFACT_INDEX_COLUMNS + ["parameter"])
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
        filename = (
            "calculated_lbwsg_paf_on_cause.all_causes.cause_specific_mortality_rate.parquet"
        )
    else:
        filename = "calculated_lbwsg_paf_on_cause.all_causes.cause_specific_mortality_rate_preterm.parquet"

    location_mapper = {
        "Ethiopia": "ethiopia",
        "Nigeria": "nigeria",
        "Pakistan": "pakistan",
    }

    output_dir = paths.PAF_DIR / location_mapper[location]

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

    age_start_dict = {"early_neonatal": 0.0, "late_neonatal": 0.01917808}
    age_end_dict = {"early_neonatal": 0.01917808, "late_neonatal": 0.07671233}
    df["age_start"] = df["age_group"].replace(age_start_dict)
    df["age_end"] = df["age_group"].replace(age_end_dict)
    df["year_start"] = 2021
    df["year_end"] = 2022
    df = df.drop("age_group", axis=1)
    index_columns = ["sex", "age_start", "age_end", "year_start", "year_end"]
    df = df.set_index(index_columns)
    unaffected_age_groups = [(0.07671233, 1.0), (1.0, 5.0)]
    for age_start, age_end in unaffected_age_groups:
        for sex in ["Male", "Female"]:
            df.loc[(sex, age_start, age_end, 2021, 2022), :] = 0

    return df.sort_index()


def load_p_rds(
    lookup_key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> float:
    csmr = get_data(data_keys.PRETERM_BIRTH.CSMR, location, years)
    p_rds = csmr * data_values.PRETERM_DEATHS_DUE_TO_RDS_PROBABILITY
    return p_rds


def load_probability_birth_facility_type(
    lookup_key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> float:
    return data_values.DELIVERY_FACILITY_TYPE_PROBABILITIES[location][lookup_key]


def load_cpap_facility_access_probability(
    lookup_key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> float:
    return data_values.CPAP_ACCESS_PROBABILITIES[location][lookup_key]


def load_no_cpap_relative_risk(
    lookup_key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> float:
    return 1 / 0.53


def load_no_cpap_paf(
    lookup_key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> float:

    # Get all no_cpap data for calculations
    p_rds = get_data(data_keys.NO_CPAP_RISK.P_RDS, location, years)
    p_home = get_data(data_keys.FACILITY_CHOICE.P_HOME, location, years)
    p_BEmONC = get_data(data_keys.FACILITY_CHOICE.P_BEmONC, location, years)
    p_CEmONC = get_data(data_keys.FACILITY_CHOICE.P_CEmONC, location, years)
    p_CPAP_home = get_data(data_keys.NO_CPAP_RISK.P_CPAP_HOME, location, years)
    p_CPAP_BEmONC = get_data(data_keys.NO_CPAP_RISK.P_CPAP_BEmONC, location, years)
    p_CPAP_CEmONC = get_data(data_keys.NO_CPAP_RISK.P_CPAP_CEmONC, location, years)
    relative_risk = get_data(data_keys.NO_CPAP_RISK.RELATIVE_RISK, location, years)
    # rr_cpap = 1 / relative_risk)
    # p_rds_cpap = (1 / relative_risk) * p_rds_no_cpap
    # p_rds_no_cpap = p_rds_cpap * relative_risk

    # Get death probability of each path
    # p_home_no_cpap = p_home * p_rds_no_cpap
    # p_BEmONC_no_cpap = p_BEmONC * 1 - p_CPAP_BEmONC * p_rds_no_cpap
    # p_CEmONC_no_cpap = p_CEmONC * 1 - p_CPAP_CEmONC * p_rds_no_cpap
    # p_BEmONC_cpap = p_BEmONC * p_CPAP_BEmONC * p_rds_cpap
    # p_CEmONC_cpap = p_CEmONC * p_CPAP_CEmONC * p_rds_cpap

    # p_rds = (
    #     p_home * p_rds_cpap * relative_risk
    #     + p_BEmONC * (1 - p_CPAP_CEmONC) * p_rds_cpap * relative_risk
    #     + p_CEmONC * (1 - p_CPAP_CEmONC) * p_rds_cpap * relative_risk
    #     + p_BEmONC * p_CPAP_BEmONC * p_rds_cpap
    #     + p_CEmONC * p_CPAP_CEmONC * p_rds_cpap
    # )
    # p_rds = (
    #     0.5 * 1.0 * p_rds_cpap * (1 / 0.53)
    #     + 0.1 * (1 - 0.075) * p_rds_cpap * (1 / 0.53)
    #     + 0.4 * (1 - 0.393) * p_rds_cpap * (1 / 0.53)
    #     + 0.1 * 0.075 * p_rds_cpap
    #     + 0.4 * 0.393 * p_rds_cpap
    # )
    # p_rds_cpap(
    #     (0.5 * 1 * (1 / 0.53))
    #     + (0.1 * (1 - 0.075) * (1 / 0.53))
    #     + (0.4 * (1 / 0.393) * (1 / 0.53))
    #     + (0.1 * 0.075)
    #     + (0.4 * 0.393)
    # ) = p_rds

    p_rds_cpap = p_rds / (
        (p_home * relative_risk)
        + (p_BEmONC * (1 - p_CPAP_BEmONC) * relative_risk)
        + (p_CEmONC * (1 - p_CPAP_CEmONC) * relative_risk)
        + (p_BEmONC * p_CPAP_BEmONC)
        + (p_CEmONC * p_CPAP_CEmONC)
    )
    paf_no_cpap = 1 - (p_rds_cpap / p_rds)
    paf_no_cpap = paf_no_cpap.fillna(0.0)
    return paf_no_cpap


def load_lbwsg_exposure(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:

    if key != data_keys.LBWSG.BIRTH_EXPOSURE:
        raise ValueError(f"Unrecognized key {key}")

    # THis is using the old key due to VPH and VI update
    exposure_key = "risk_factor.low_birth_weight_and_short_gestation.exposure"
    # Get exposure for all age groups except birth age group
    all_age_exposure = load_standard_data(exposure_key, location, years)

    entity = utilities.get_entity(exposure_key)
    birth_exposure = extra_gbd.load_lbwsg_exposure(location, exposure_key)
    # This category was a mistake in GBD 2019, so drop.
    extra_residual_category = vi_globals.EXTRA_RESIDUAL_CATEGORY[entity.name]
    birth_exposure = birth_exposure.loc[
        birth_exposure["parameter"] != extra_residual_category
    ]
    birth_exposure = birth_exposure.set_index(
        ["location_id", "year_id", "sex_id", "parameter"]
    )[vi_globals.DRAW_COLUMNS]
    # Sometimes there are data values on the order of 10e-300 that cause
    # floating point headaches, so clip everything to reasonable values
    birth_exposure = birth_exposure.clip(lower=vi_globals.MINIMUM_EXPOSURE_VALUE)
    birth_exposure = reshape_to_vivarium_format(birth_exposure, location).reset_index()
    birth_exposure["age_start"] = (0 - 7) / 365.0
    birth_exposure["age_end"] = 0.0
    idx_cols = ["sex", "age_start", "age_end", "year_start", "year_end", "parameter"]
    exposure = pd.concat([all_age_exposure.reset_index(), birth_exposure])
    exposure = exposure.set_index(idx_cols)[vi_globals.DRAW_COLUMNS]

    # normalize so all categories sum to 1
    total_exposure = exposure.groupby(["age_start", "age_end", "sex"]).transform("sum")
    exposure = (exposure / total_exposure).reset_index().set_index(idx_cols).sort_index()
    return exposure


def load_preterm_prevalence(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    # TODO: implement
    exposure = get_data(data_keys.LBWSG.BIRTH_EXPOSURE, location, years).reset_index()
    # Remove birth age group
    exposure = exposure.loc[exposure["age_end"] > 0.0]
    categories = get_data(data_keys.LBWSG.CATEGORIES, location, years)
    # Get preterm categories
    preterm_cats = []
    for cat, description in categories.items():
        i = utilities.parse_short_gestation_description(description)
        if i.right < 37:
            preterm_cats.append(cat)

    # Subset exposure to preterm categories
    preterm_exposure = exposure.loc[exposure["parameter"].isin(preterm_cats)]
    preterm_exposure = preterm_exposure.drop(columns=["parameter"])
    draw_cols = [col for col in preterm_exposure.columns if "draw" in col]
    sum_exposure = preterm_exposure.groupby(metadata.ARTIFACT_INDEX_COLUMNS)[draw_cols].sum()

    return sum_exposure


def load_antibiotic_facility_probability(
    key: str,
    location: str,
    years: Optional[Union[int, str, List[int]]] = None,
) -> pd.DataFrame:
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)
    facility_uniform_dist = data_values.ANTIBIOTIC_FACILITY_TYPE_DISTRIBUTION[location][key]
    draws = utilities.get_random_variable_draws(
        metadata.ARTIFACT_INDEX_COLUMNS, key, facility_uniform_dist
    )
    data = pd.DataFrame([draws], columns=metadata.ARTIFACT_COLUMNS, index=demography.index)

    return data


def load_no_antibiotics_relative_risk(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    rr_dist = data_values.ANTIBIOTIC_RELATIVE_RISK_DISTRIBUTION
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)
    draws = utilities.get_random_variable_draws(metadata.ARTIFACT_INDEX_COLUMNS, key, rr_dist)
    data = pd.DataFrame([draws], columns=metadata.ARTIFACT_COLUMNS, index=demography.index)

    return data


def load_no_antibiotics_paf(
    key: str, location: str, years: Optional[Union[int, str, List[int]]] = None
) -> pd.DataFrame:
    # Get all no_cpap data for calculations
    csmr = get_data(data_keys.NEONATAL_SEPSIS.CSMR, location, years)
    p_sepsis = csmr.copy()
    p_home = get_data(data_keys.FACILITY_CHOICE.P_HOME, location, years)
    p_BEmONC = get_data(data_keys.FACILITY_CHOICE.P_BEmONC, location, years)
    p_CEmONC = get_data(data_keys.FACILITY_CHOICE.P_CEmONC, location, years)
    p_antibiotic_home = get_data(
        data_keys.NO_ANTIBIOTICS_RISK.P_ANTIBIOTIC_HOME, location, years
    )
    p_antibiotic_BEmONC = get_data(
        data_keys.NO_ANTIBIOTICS_RISK.P_ANTIBIOTIC_BEmONC, location, years
    )
    p_antibiotic_CEmONC = get_data(
        data_keys.NO_ANTIBIOTICS_RISK.P_ANTIBIOTIC_CEmONC, location, years
    )
    relative_risk = get_data(data_keys.NO_ANTIBIOTICS_RISK.RELATIVE_RISK, location, years)
    # This is derived in the CPAP PAF calculation
    p_sepsis_antibiotic = p_sepsis / (
        (p_home * (1 - p_antibiotic_home) * relative_risk)
        + (p_home * p_antibiotic_home)
        + (p_BEmONC * (1 - p_antibiotic_BEmONC) * relative_risk)
        + (p_CEmONC * (1 - p_antibiotic_CEmONC) * relative_risk)
        + (p_BEmONC * p_antibiotic_BEmONC)
        + (p_CEmONC * p_antibiotic_CEmONC)
    )
    paf_no_antibiotic = 1 - (p_sepsis_antibiotic / p_sepsis)
    paf_no_antibiotic = paf_no_antibiotic.fillna(0.0)

    return paf_no_antibiotic


def reshape_to_vivarium_format(df, location):
    df = vi_utils.reshape(df, value_cols=vi_globals.DRAW_COLUMNS)
    df = vi_utils.scrub_gbd_conventions(df, location)
    df = vi_utils.split_interval(df, interval_column="age", split_column_prefix="age")
    df = vi_utils.split_interval(df, interval_column="year", split_column_prefix="year")
    df = vi_utils.sort_hierarchical_data(df)
    df.index = df.index.droplevel("location")
    return df
