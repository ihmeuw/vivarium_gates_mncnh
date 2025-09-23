import os

import numpy as np
import pandas as pd
import scipy
from vivarium import Artifact, InteractiveContext

from vivarium_gates_mncnh.constants import metadata

"""
This code is intended to read in the effects of hemoglobin on stillbirth, birth weight
and gestational age outcomes as estimated from the IHME Hemoglobin team and 
prepare them for use in our simulation.

Specifically, this code:
- Standardizes the RR exposure levels to be 1,000 equal increments between 40 and 150 g/L
- Reorders the draws for GA and BW effects in order of magnitude of risk at the 40 g/L hemoglobin 
  level (which is the lowest hemoglobin exposure included in the risk effect estimation)
    - This is in order to make the individual RR estimates follow logical pairings
    between the effect of hemoglobin on LBWSG and the effect of hemoglobin on 
    neonatal sepsis, which is mediated through the effect of hemoglobin on LBWSG
- Converts the effect of hemoglobin on dichotomous preterm birth and low birth weight
outcomes to continuous shifts in gestational age and birth weight (and saves these values)
- Scales the effect of hemoglobin on stillbirth, gestational age, and birth weight to the effect of
IV iron and saves these values for ultimate use in our simulation

Note that this code is organized to perform a calculation for all modeled locations
for a single draw and therefore can be parallelized by draw.

A note on data inputs for this code: data in the {outcome}_bop_rrs.csv files
were copied from files uploaded to mnt/team/anemia/pub/bop/sim_studies
by Ditha Nanditha in the summer of 2025. There is an associated email correspondence between
Ditha, Ali, Abie, Syl, and Nathaniel for reference.
"""

artifact_directory = (
    "/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/artifacts/model13.1/"
)
# This code relies on data specific to:
# 1. The LBWSG birth exposure in GBD (using GBD 2021 data in artifact 13.1)
# 2. The hemoglobin risk exposure levels (using GBD 2023 data in artifact 13.1)
# Therefore, it will need to be re-run if either of these are updated


def load_bop_rrs(outcome):
    """Load burden of proof hemoglobin estimates for specified outcome"""
    rrs = pd.read_csv(f"{outcome}_bop_rrs.csv")
    rrs = rrs.set_index("risk")
    rrs = np.exp(
        rrs
    ).reset_index()  # convert beta coefficients to relative risks by exponentiating
    rrs["outcome"] = outcome
    return rrs


def get_gbd_exposure_levels():
    """Loads the 1,000 exposure estimates used in GBD for the maternal disorders outcomes."""
    art = Artifact(artifact_directory + "ethiopia.hdf")
    rrs = art.load("risk_factor.hemoglobin.relative_risk").reset_index()
    exposure_levels = rrs.parameter.unique()
    assert (
        len(exposure_levels) == 1000
    ), f"Expected 1000 unique exposure levels across age groups and causes, but got {len(rrs.exposure.unique())}"
    return exposure_levels


def convert_rrs_to_gbd_exposure(rrs, exposure_levels):
    """Using linear interpolation, make a new dataframe with relative risks specific to the exposure levels used in GBD
    rather than the exposure levels output from the outcome-specific burden of proof models."""
    from scipy.interpolate import interp1d

    x_new = exposure_levels
    rrs_interp = pd.DataFrame({"risk": x_new})
    x_old = rrs["risk"].values
    for col in rrs.columns:
        if col.startswith("draw_"):
            y = rrs[col].values
            f = interp1d(
                x_old, y, kind="linear", bounds_error=False, fill_value=(y[0], y[-1])
            )
            rrs_interp[col] = f(x_new)
    return rrs_interp


def transform_and_reorder_rrs(rrs, exposure_levels):
    """Make relative risks specific to TMREL value of 120 and reorder draws by magnitude of risk at the lowest exposure level."""
    tmrel = exposure_levels[len([x for x in exposure_levels if x < 120])]
    assert (
        tmrel.round(0) == 120
    ), f"Expected preterm birth tmrel to be 120 g/L, but got {tmrel}"
    tmrel_data = rrs.loc[rrs.risk == tmrel]
    # rescale to be relative to tmrel of 120 g/L
    rrs = (
        rrs.set_index(["outcome", "risk"])
        / tmrel_data.set_index("outcome").drop(columns="risk")
    ).reset_index()
    # reorder draws by magnitude of risk at the lowest exposure level
    rrs = (
        rrs.set_index(["outcome", "risk"])
        .rename_axis("draw", axis="columns")
        .stack()
        .rename("value")
        .reset_index()
    )
    assert (
        exposure_levels[0] == 40
    ), "Expected lowest exposure level equal to 40 g/L, but got something else"
    order = rrs.loc[rrs.risk == exposure_levels[0]]
    order = order.sort_values(by="value")
    order["order"] = np.arange(len(order))
    order["ranked_draw"] = "draw_" + order.order.astype(str)
    rrs = (
        rrs.merge(order[["draw", "ranked_draw"]], on="draw")
        .drop(columns="draw")
        .rename(columns={"ranked_draw": "draw"})
    )
    rrs = rrs.set_index(["outcome", "risk", "draw"]).unstack("draw")  # .reset_index()
    rrs.columns = rrs.columns.droplevel()
    rrs = rrs.reset_index()
    return rrs


def load_prepped_rrs(outcome):
    """Load relative risks of hemoglobin on specific outcome that are:
    - rescaled to a tmrel of 120 g/L,
    - interpolated to the exposure levels used in GBD,
    - reordered by magnitude of risk at the lowest exposure level,
    - and scale up to 500 draws by duplicating existing draws."""
    rrs = load_bop_rrs(outcome)
    exposure_levels = get_gbd_exposure_levels()
    rrs = convert_rrs_to_gbd_exposure(rrs, exposure_levels)
    rrs["outcome"] = outcome
    rrs = transform_and_reorder_rrs(rrs, exposure_levels)
    rrs_copy = rrs.copy().set_index([x for x in rrs.columns if "draw" not in x])
    assert len(rrs_copy.columns) == 250, f"Expected 250 draws but got {len(rrs_copy.columns)}"
    rrs_copy.columns = [int(col.replace("draw_", "")) for col in rrs_copy.columns]
    rrs_copy.columns = [f"draw_{i + 250}" for i in rrs_copy.columns]
    rrs = rrs.merge(rrs_copy.reset_index(), on=[x for x in rrs.columns if "draw" not in x])
    return rrs


def get_lbwsg_metadata():
    """Loads metadata (birth weight and gestational age start/end values) for low birth weight and short gestation exposure categories.
    Note that this function does not return any actual exposure data.

    todo: this code could be improved by using the 'risk_factor.low_birth_weight_and_short_gestation.categories'
    key only (and not the 'risk_factor.low_birth_weight_and_short_gestation.exposure' key) to achieve the purpose
    of this method.
    """
    # there are hard-coded location/sex/draw values here, but these are not used in actual data generation
    # note: the LBWSG exposure categories are the same across locations/sex/draw/ages in GBD, so the
    # values used here are arbitrary
    art = Artifact(artifact_directory + "ethiopia.hdf")
    art_exposure = art.load(f"risk_factor.low_birth_weight_and_short_gestation.exposure")[
        "draw_0"
    ].reset_index()
    art_exposure = art_exposure.loc[
        (art_exposure.sex_of_child == "Male") & (art_exposure.child_age_start == 0)
    ][["parameter"]]
    art_cats = art.load(f"risk_factor.low_birth_weight_and_short_gestation.categories")
    art_exposure["bw_start"] = np.nan
    art_exposure["bw_end"] = np.nan
    art_exposure["ga_start"] = np.nan
    art_exposure["ga_end"] = np.nan
    for i in art_exposure.index:
        art_exposure["bw_start"][i] = (
            art_cats[art_exposure.parameter[i]].split("wks, [")[1].split(",")[0]
        )
        art_exposure["bw_end"][i] = (
            art_cats[art_exposure.parameter[i]]
            .split("wks, [")[1]
            .split(", ")[1]
            .split(")")[0]
        )
        art_exposure["ga_start"][i] = (
            art_cats[art_exposure.parameter[i]].split("- [")[1].split(", ")[0]
        )
        art_exposure["ga_end"][i] = (
            art_cats[art_exposure.parameter[i]]
            .split("- [")[1]
            .split(") wks")[0]
            .split(", ")[1]
        )
    art_exposure["ga_end"] = np.where(
        art_exposure.ga_end == "42+] wks", "42", art_exposure.ga_end
    )
    art_exposure["bw_end"] = np.where(
        art_exposure.bw_end == "9999] g", 1000, art_exposure.bw_end
    )
    for col in ["ga_end", "ga_start", "bw_end", "bw_start"]:
        art_exposure[f"{col}"] = art_exposure[f"{col}"].astype(int)
    return art_exposure


def get_lbwsg_birth_exposure(location):
    # this will return birth exposure data data for both sexes and all draws for the specified location
    # it will also include metadata for the exposure categories (GA/BW start/end values)
    art = Artifact(artifact_directory + location + ".hdf")
    exp = art.load(
        "risk_factor.low_birth_weight_and_short_gestation.birth_exposure"
    ).reset_index()
    art_exposure = get_lbwsg_metadata()
    exp = exp.merge(art_exposure, on="parameter")
    exp = (
        exp.set_index([x for x in exp.columns if "draw" not in x])
        .rename_axis("draw", axis="columns")
        .stack()
        .rename("exposure")
        .reset_index()
        .rename(columns={"sex_of_child": "sex"})
    )
    exp["location"] = location
    return exp


def calculate_shift_from_rr(exp_data, rr_data, location, outcome, sex, draw, exposure_level):
    """Calculates the shift in gestational age or birth weight exposure needed to achieve a target relative risk."""

    def shift_optimization(shift):
        exp_exp = (exp_sub + shift).reset_index()
        exp_exp["frac_exposed"] = np.where(
            exp_exp[f"{column_filter}_end"] <= threshold,
            1,
            np.where(
                exp_exp[f"{column_filter}_start"] >= threshold,
                0,
                (
                    (threshold - exp_exp[f"{column_filter}_start"])
                    / (exp_exp[f"{column_filter}_end"] - exp_exp[f"{column_filter}_start"])
                ),
            ),
        )
        exp_exposed_prevalence = (exp_exp.exposure * exp_exp.frac_exposed).sum()
        rr = exp_exposed_prevalence / tmrel_exposure_prevalence
        return np.abs(rr - rr_target)

    # define outcome-specific parameters
    column_filter = "ga" if outcome == "ptb" else "bw"
    threshold = 37 if outcome == "ptb" else 2500
    bounds = (-5, 3) if outcome == "ptb" else (-3000, 2500)

    exp_sub = exp_data.loc[(exp_data.sex == sex) & (exp_data.draw == draw)].set_index(
        [x for x in exp_data.columns if column_filter not in x]
    )
    exp_tmrel = exp_sub.reset_index()
    rr_target = rr_data.loc[(rr_data.exposure == exposure_level)][draw].iloc[0]
    exp_tmrel["frac_exposed"] = np.where(
        exp_tmrel[f"{column_filter}_end"] <= threshold,
        1,
        np.where(
            exp_tmrel[f"{column_filter}_start"] >= threshold,
            0,
            (
                (threshold - exp_tmrel[f"{column_filter}_start"])
                / (exp_tmrel[f"{column_filter}_end"] - exp_tmrel[f"{column_filter}_start"])
            ),
        ),
    )
    tmrel_exposure_prevalence = (exp_tmrel.exposure * exp_tmrel.frac_exposed).sum()

    return scipy.optimize.minimize_scalar(
        shift_optimization, bounds=bounds, method="bounded"
    )["x"]


def get_lbwsg_shifts(draw):
    """For a single draw across all locations and sexes, returns GA and BW shifts for all hemoglobin exposure levels"""
    ptb_rrs = load_prepped_rrs("ptb").rename(columns={"risk": "exposure"})[["exposure", draw]]
    lbw_rrs = load_prepped_rrs("lbw").rename(columns={"risk": "exposure"})[["exposure", draw]]
    assert np.all(
        ptb_rrs.exposure.unique() == lbw_rrs.exposure.unique()
    ), "Hemoglobin risk exposure values are different between PTB and LBW outcomes"
    exposure_levels = ptb_rrs.exposure.unique()

    results = pd.DataFrame()
    for location in [location.lower() for location in metadata.LOCATIONS]:
        exp = get_lbwsg_birth_exposure(location)
        exp = exp.loc[exp.draw == draw]
        for sex in ["Male", "Female"]:
            ga_shifts = []
            bw_shifts = []
            for exposure_level in exposure_levels:
                ga_shifts.append(
                    calculate_shift_from_rr(
                        exp_data=exp,
                        rr_data=ptb_rrs,
                        location=location,
                        outcome="ptb",
                        sex=sex,
                        draw=draw,
                        exposure_level=exposure_level,
                    )
                )
                bw_shifts.append(
                    calculate_shift_from_rr(
                        exp_data=exp,
                        rr_data=lbw_rrs,
                        location=location,
                        outcome="lbw",
                        sex=sex,
                        draw=draw,
                        exposure_level=exposure_level,
                    )
                )
            temp_ga = pd.DataFrame(
                {
                    "location": location,
                    "sex": sex,
                    "draw": draw,
                    "exposure": exposure_levels,
                    "value": ga_shifts,
                    "outcome": "gestational_age",
                }
            )
            temp_bw = pd.DataFrame(
                {
                    "location": location,
                    "sex": sex,
                    "draw": draw,
                    "exposure": exposure_levels,
                    "value": bw_shifts,
                    "outcome": "birth_weight",
                }
            )
            results = pd.concat([results, temp_ga, temp_bw], ignore_index=True)
    return results


def load_iv_iron_mean_difference(draw):
    # TODO: replace this with an artifact version so that the provenance is more clear
    """Load mean difference in hemoglobin due to IV iron intervention for all locations"""
    from vivarium_gates_mncnh.data.loader import load_iv_iron_hemoglobin_effect_size

    df = pd.DataFrame(
        {
            "location": metadata.LOCATIONS,
            "iv_iron_md": [
                load_iv_iron_hemoglobin_effect_size(
                    key="iv_iron_hemoglobin_effect_size", location=location
                )[draw][0]
                for location in metadata.LOCATIONS
            ],
        }
    )
    return df


def run_exposure_level_tests(data):
    exposure_levels = sorted(data.exposure.unique().tolist())
    assert np.all(
        exposure_levels[i] < exposure_levels[i + 1] for i in range(len(exposure_levels) - 1)
    ), "exposure levels not sorted in ascending order"
    if data.outcome.values[0] == "stillbirth":
        for draw in data.draw.unique():
            for location in data.location.unique():
                assert np.all(
                    data.loc[
                        (data.draw == draw) & (data.location == location)
                    ].exposure.values
                    == exposure_levels
                ), "Inconsistencies in exposure level values and/or orders"
    else:
        for location in data.location.unique():
            for sex in data.sex.unique():
                for outcome in data.outcome.unique():
                    assert np.all(
                        data.loc[
                            (data.location == location)
                            & (data.sex == sex)
                            & (data.outcome == outcome)
                        ].exposure.values
                        == exposure_levels
                    ), "Inconsistencies in exposure level values and/or orders"
    assert np.allclose(
        exposure_levels,
        np.linspace(
            exposure_levels[0],
            exposure_levels[len(exposure_levels) - 1],
            num=len(exposure_levels),
        ),
    ), "Exposure levels are not evenly spaced"


def scale_effects_to_iv_iron(data):
    """Scales hemoglobin effects on stillbirth or GA and BW (relative to the hemoglobin TMREL) to the effect size of IV iron.
    
    Note that the data that gets passed to this function either contains:
    - dataframe with stillbirth relative risks by hemoglobin exposure level for all locations and a single draw (not sex-specific), or
    - dataframe with GA and BW shifts by hemoglobin exposure level for all locations and sexes for a single draw

    """
    assert len(data.draw.unique()) == 1, "Data should only contain a single draw"
    draw = data.draw.unique()[0]
    iv_iron_md = load_iv_iron_mean_difference(draw)
    exposure_levels = sorted(data.exposure.unique().tolist())
    data = data.sort_values(
        by=[x for x in data.columns if x not in ["exposure", "value"]] + ["exposure"]
    )
    run_exposure_level_tests(data)  # run tests before adding additional exposure levels

    exposure_increment = exposure_levels[1] - exposure_levels[0]
    iv_iron_exposure_increment = (
        round(iv_iron_md.set_index("location") / exposure_increment, 0)
    ).astype(int)

    # add rows to dataframe so that there are enough exposure levels to shift by the max iv_iron_exposure_increment
    # we assume that the effect at these new exposure levels (of hemoglobin on BW, GA, or stillbirth outcome)
    # is the same as the effect at the highest existing exposure level
    start_val = exposure_levels[len(exposure_levels) - 1] + exposure_increment
    stop_val = start_val + iv_iron_exposure_increment.iv_iron_md.max() * exposure_increment
    extra_exposure_levels = np.arange(start=start_val, stop=stop_val, step=exposure_increment)
    data_extra = pd.DataFrame()
    for location in data.location.unique():
        if data.outcome.values[0] == "stillbirth":
            temp = pd.DataFrame()
            temp["exposure"] = extra_exposure_levels
            temp["location"] = location
            temp["outcome"] = "stillbirth"
            temp["value"] = data.loc[
                (data.location == location)
                & (data.exposure == exposure_levels[len(exposure_levels) - 1])
            ].value.values[0]
            data_extra = pd.concat([data_extra, temp], ignore_index=True)
        else:
            for sex in data.sex.unique():
                for outcome in data.outcome.unique():
                    temp = pd.DataFrame()
                    temp["exposure"] = extra_exposure_levels
                    temp["location"] = location
                    temp["sex"] = sex
                    temp["outcome"] = outcome
                    temp["value"] = data.loc[
                        (data.location == location)
                        & (data.sex == sex)
                        & (data.outcome == outcome)
                        & (data.exposure == exposure_levels[len(exposure_levels) - 1])
                    ].value.values[0]
                    data_extra = pd.concat([data_extra, temp], ignore_index=True)
    data_extra["draw"] = draw
    data = pd.concat([data, data_extra], ignore_index=True)
    data = data.sort_values(
        by=[x for x in data.columns if x not in ["exposure", "value"]] + ["exposure"]
    )
    run_exposure_level_tests(data)  # run tests after adding additional exposure levels

    result = pd.DataFrame()
    for location in data.location.unique():
        location_data = data[data.location == location]
        location_shift = iv_iron_exposure_increment.loc[location.title(), "iv_iron_md"]
        if data.outcome.values[0] == "stillbirth":
            location_data["value"] = (
                location_data["value"].shift(-location_shift) / location_data["value"]
            )
        else:
            location_data["value"] = (
                location_data["value"].shift(-location_shift) - location_data["value"]
            )
        result = pd.concat([result, location_data], ignore_index=True)
    return result.loc[
        result.exposure <= exposure_levels[len(exposure_levels) - 1]
    ]  # return only original exposure levels


def calculate_and_save_lbwsg_shifts(results_directory, draw):
    assert os.path.exists(
        results_directory + "lbwsg_shifts/"
    ), f"Results directory {results_directory + 'lbwsg_shifts/'} does not exist"
    lbwsg_shifts = get_lbwsg_shifts(draw)
    lbwsg_shifts.to_csv(results_directory + "lbwsg_shifts/" + draw + ".csv", index=False)


def calculate_and_save_iv_iron_lbwsg_shifts(results_directory, draw):
    # note that this function takes about 10-15 minutes to run for a single draw
    assert os.path.exists(
        results_directory + "lbwsg_shifts/"
    ), f"Results directory {results_directory + 'lbwsg_shifts/'} does not exist"
    assert os.path.exists(
        results_directory + "iv_iron_lbwsg_shifts/"
    ), f"Results directory {results_directory + 'iv_iron_lbwsg_shifts/'} does not exist"
    if f"{draw}.csv" not in os.listdir(results_directory + "lbwsg_shifts/"):
        calculate_and_save_lbwsg_shifts(results_directory, draw)
    lbwsg_shifts = pd.read_csv(results_directory + "lbwsg_shifts/" + draw + ".csv")
    iv_iron_shifts = scale_effects_to_iv_iron(lbwsg_shifts)
    iv_iron_shifts.to_csv(
        results_directory + "iv_iron_lbwsg_shifts/" + draw + ".csv", index=False
    )


def calculate_iv_iron_stillbirth_effects():
    """Calculates relative risk of IV iron on stillbirth specific to
    hemoglobin exposure level. Depends on effect size and threshold of
    IV iron intervention and hemoglobin team's estimates of hemoglobin's
    effect on stillbirth. Calculates all draws at once and is not
    sex or age specific."""
    # takes about 7 minutes to run
    bop_rrs = load_bop_rrs("stillbirth").sort_values(by="risk")
    exposure_levels = sorted(get_gbd_exposure_levels())
    rrs = convert_rrs_to_gbd_exposure(bop_rrs, exposure_levels)
    rrs_prepped = rrs.rename(columns={"risk": "exposure"}).set_index("exposure")
    rrs_prepped = (
        rrs_prepped.stack().reset_index().rename(columns={"level_1": "draw", 0: "value"})
    )
    rrs_prepped["outcome"] = "stillbirth"
    rrs_prepped = pd.concat(
        [rrs_prepped.assign(location=location) for location in metadata.LOCATIONS],
        ignore_index=True,
    )
    rrs_prepped["location"] = rrs_prepped.location.str.lower()

    effects = pd.concat(
        [
            scale_effects_to_iv_iron(rrs_prepped.loc[rrs_prepped.draw == draw])
            for draw in rrs_prepped.draw.unique()
        ],
        ignore_index=True,
    )
    effects = effects.pivot_table(
        index=["location", "exposure"], values="value", columns="draw"
    ).reset_index()
    effects.to_csv("iv_iron_stillbirth_rrs.csv")
    return effects
