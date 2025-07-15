import os

import numpy as np
import pandas as pd
import scipy
from vivarium import Artifact, InteractiveContext


def load_rrs(outcome):
    """Loads hemoglobin outcome-specific relative risks as provided by hemoglobin team and transforms
    them by exponentiating the beta coefficients to get relative risks, rescaling to a tmrel of 120 g/L,
    and reordering the draws by magnitude of risk at the lowest exposure level."""
    # TODO: move these files into model repo?
    rrs = pd.read_csv(f"/mnt/team/anemia/pub/bop/sim_studies/{outcome}/inner_draws.csv")
    rrs = rrs.set_index("risk")
    rrs = np.exp(
        rrs
    ).reset_index()  # convert beta coefficients to relative risks by exponentiating
    rrs["outcome"] = outcome
    exposure_levels = rrs.risk.unique()
    if outcome == "ptb":
        index_val = 691
    elif outcome == "lbw":
        index_val = 692
    tmrel = exposure_levels[index_val]  # this corresponds to an exposure level of 120 g/L
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
        .stack()
        .reset_index()
        .rename(columns={"level_2": "draw", 0: "value"})
    )
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


def get_lbwsg_metadata():
    """Loads metadata (birth weight and gestational age start/end values) for low birth weight and short gestation exposure categories.
    Note that this function does not return any actual exposure data."""
    # there are hard-coded location/sex/draw values here, but these are not used in actual data generation

    # TODO: is there a better way to load this within the repo?
    artifact_dir = (
        "/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/artifacts/model10.0/"
    )
    art = Artifact(artifact_dir + "ethiopia.hdf")
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

    # TODO: update this to be relative to most recent artifact?
    # Or pull directly using repo code?
    artifact_dir = (
        "/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/artifacts/model10.0/"
    )
    art = Artifact(artifact_dir + location + ".hdf")
    exp = art.load(
        "risk_factor.low_birth_weight_and_short_gestation.birth_exposure"
    ).reset_index()
    art_exposure = get_lbwsg_metadata()
    exp = exp.merge(art_exposure, on="parameter")
    exp = (
        exp.set_index([x for x in exp.columns if "draw" not in x])
        .stack()
        .reset_index()
        .rename(columns={0: "exposure", "level_8": "draw", "sex_of_child": "sex"})
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
        ga_shift_optimization, bounds=bounds, method="bounded"
    )["x"]


def get_lbwsg_shifts(draw):
    """For a single draw across all locations and sexes, returns GA and BW shifts for all hemoglobin exposure levels"""
    ptb_rrs = load_rrs("ptb").rename(columns={"risk": "exposure"})[["exposure", draw]]
    lbw_rrs = load_rrs("lbw").rename(columns={"risk": "exposure"})[["exposure", draw]]
    ptb_exposure_levels = ptb_rrs.exposure.unique()
    lbw_exposure_levels = lbw_rrs.exposure.unique()

    results = pd.DataFrame()
    # TODO: specify locations using model spec file or something?
    for location in ["ethiopia", "nigeria", "pakistan"]:
        exp = get_lbwsg_birth_exposure(location)
        exp = exp.loc[exp.draw == draw]
        for sex in ["Male", "Female"]:
            ga_shifts = []
            bw_shifts = []
            for exposure_level in ptb_exposure_levels:
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
            for exposure_level in lbw_exposure_levels:
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
                    "exposure": ptb_exposure_levels,
                    "value": ga_shifts,
                    "outcome": "gestational_age",
                }
            )
            temp_bw = pd.DataFrame(
                {
                    "location": location,
                    "sex": sex,
                    "draw": draw,
                    "exposure": lbw_exposure_levels,
                    "value": bw_shifts,
                    "outcome": "birth_weight",
                }
            )
            results = pd.concat([results, temp_ga, temp_bw], ignore_index=True)
    return results


def scale_lbwsg_to_iv_iron(lbwsg_shifts, draw):
    """Scales hemoglobin effects on GA and BW (relative to the hemoglobin TMREL) to the effect size of IV iron"""

    # TODO: read in IV iron effect size/threshold from the repo/artifact here instead of hard-coding it
    iv_iron_md = 23  # change in hemoglobin exposure associated with IV iron
    iv_iron_threshold = 100  # maximum hemoglobin exposure level (g/L) eligible for IV iron

    lbwsg_shifts = lbwsg_shifts.reset_index().sort_values(
        by=[x for x in lbwsg_shifts.columns if x not in ["value", "exposure"]] + ["exposure"]
    )
    ga_shifts = lbwsg_shifts.loc[lbwsg_shifts.outcome == "gestational_age"]
    bw_shifts = lbwsg_shifts.loc[lbwsg_shifts.outcome == "birth_weight"]
    ptb_exposure_levels = sorted(ga_shifts.exposure.unique())
    lbw_exposure_levels = sorted(bw_shifts.exposure.unique())

    assert np.all(
        ga_shifts.loc[
            (ga_shifts.sex == ga_shifts.sex.unique()[0])
            & (ga_shifts.location == ga_shifts.location.unique()[0])
        ].exposure
        == ptb_exposure_levels
    ), "PTB exposures not in the right order"
    assert np.all(
        bw_shifts.loc[
            (bw_shifts.sex == bw_shifts.sex.unique()[0])
            & (bw_shifts.location == bw_shifts.location.unique()[0])
        ].exposure
        == lbw_exposure_levels
    ), "LBW exposures not in the right order"
    assert np.allclose(
        ptb_exposure_levels,
        np.linspace(
            ptb_exposure_levels[0],
            ptb_exposure_levels[len(ptb_exposure_levels) - 1],
            num=len(ptb_exposure_levels),
        ),
    ), "PTB exposure levels are not evenly spaced"
    assert np.allclose(
        lbw_exposure_levels,
        np.linspace(
            lbw_exposure_levels[0],
            lbw_exposure_levels[len(lbw_exposure_levels) - 1],
            num=len(lbw_exposure_levels),
        ),
    ), "LBW exposure levels are not evenly spaced"

    ptb_exposure_increment = (
        ptb_exposure_levels[1] - ptb_exposure_levels[0]
    )  # change in hemoglobin exposure (g/L) per exposure level used in GBD risk curve
    ptb_iv_iron_exposure_increment = (
        (iv_iron_md / ptb_exposure_increment).round(0).astype(int)
    )
    lbw_exposure_increment = (
        lbw_exposure_levels[1] - lbw_exposure_levels[0]
    )  # change in hemoglobin exposure (g/L) per exposure level used in GBD risk curve
    lbw_iv_iron_exposure_increment = (
        (iv_iron_md / lbw_exposure_increment).round(0).astype(int)
    )

    # ensure that we are not running into issues with shifting beyond the valid exposure range
    # if these tests fail, it means that for some hemoglobin exposure levels, the IV iron effect moves the expousre level beyond levels that exist in this data frame
    # and therefore will end up reading data specific to another sex/location demographic group
    # to fix thise, we would need to add exposure levels to these data frames and make some assumotion about how to fill in the data for them
    # (probably just assume that they ave the same data as the highest existing exposure level)
    assert (
        ptb_exposure_levels[len(ptb_exposure_levels) - 1 - ptb_iv_iron_exposure_increment]
        > iv_iron_threshold
    ), "Invalid exposure range for IV iron shifts"
    assert (
        lbw_exposure_levels[len(lbw_exposure_levels) - 1 - lbw_iv_iron_exposure_increment]
        > iv_iron_threshold
    ), "Invalid exposure range for IV iron shifts"

    # note: this .shift() approach relies highly on the ordering of exposure levels in the ga_shift and bw_shift dataframes
    # an improved approach would utilize exposure values directly, but for now we run tests to ensure that the ordering is as required before running this function
    ga_shifts["value"] = (
        ga_shifts["value"].shift(-ptb_iv_iron_exposure_increment) - ga_shifts["value"]
    )
    bw_shifts["value"] = (
        bw_shifts["value"].shift(-lbw_iv_iron_exposure_increment) - bw_shifts["value"]
    )
    ga_shifts["value"] = np.where(
        ga_shifts.exposure > iv_iron_threshold, np.nan, ga_shifts["value"]
    )  # missing values for exposure levels not eligible for IV iron
    bw_shifts["value"] = np.where(
        bw_shifts.exposure > iv_iron_threshold, np.nan, bw_shifts["value"]
    )  # missing values for exposure levels not eligible for IV iron

    iv_iron_lbwsg_shifts = pd.concat([ga_shifts, bw_shifts], ignore_index=True).dropna()
    return iv_iron_lbwsg_shifts


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
    iv_iron_shifts = scale_lbwsg_to_iv_iron(lbwsg_shifts, draw)
    iv_iron_shifts.to_csv(
        results_directory + "iv_iron_lbwsg_shifts/" + draw + ".csv", index=False
    )
