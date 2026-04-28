from pathlib import Path

_CODE_DIR = Path(__file__).parent

import numpy as np
import pandas as pd

from vivarium_gates_mncnh.constants import metadata
from vivarium_gates_mncnh.data.sim_utils import initialize_simulation

# This code relies on data specific to:
# 1. The low hemoglobin risk exposure levels (using GBD 2023 data in artifact 18.0)
# 2. The low hemoglobin relative risk values (using GBD 2021 data in artifact 18.0)
# 3. The direct effect of neonatal sepsis relative risk values, which are themselves dependent on:
# a. The LBWSG risk factor data (using GBD 2021 data in artifact 18.0)
# b. Hemoglobin risk factor data, specifically the effect on neonatal sepsis (using GBD 2023 data in artifact 18.0)
# Therefore, it will need to be re-run if any of these are updated


def load_hemoglobin_rrs_on_maternal_disorders():
    """Load hemoglobin relative risks on maternal disorders, scaled to the TMRED.

    Load from a cached CSV if available; otherwise compute from GBD data,
    scale to the TMRED, and cache to disk. Calling from loader.py instead of
    the artifact because the latest artifact version available as of 9.30.25
    (model 18.0) has version that cuts at 100 draws and we want to update to
    250 draws instead.
    """
    if (_CODE_DIR / "hemoglobin_rrs_scaled_to_tmred.csv").exists():
        rrs_scaled = pd.read_csv(_CODE_DIR / "hemoglobin_rrs_scaled_to_tmred.csv")
    else:
        print(
            "NOTE: data for scaled hemoglobin RRs on maternal disorders must be generated using the artifact environment prior to running the interactive context using the simulation environment. see code for more details."
        )
        location = "Ethiopia"  # RRs do not vary by location, so using this as an arbitrary hard coded value
        from vivarium_gates_mncnh.data.loader import (
            load_hemoglobin_relative_risk,
            load_hemoglobin_tmred,
        )

        rrs = load_hemoglobin_relative_risk(
            key="risk_factor.hemoglobin.relative_risk", location=location
        ).reset_index()
        assert (
            len([x for x in rrs.columns if "draw" in x]) == 500
        ), "Hemoglobin RR data does not have 500 draws"
        for i in list(range(0, 250)):
            assert np.all(
                rrs[f"draw_{i}"] == rrs[f"draw_{i + 250}"]
            ), "Hemoglobin RR data does not have 250 unique draws"
        assert np.all(
            rrs.set_index(["parameter", "age_start", "affected_entity"])[
                [x for x in rrs.columns if "draw" in x]
            ]
            .round(5)
            .groupby(["parameter", "affected_entity"])
            .nunique()
            == 1
        ), "Hemoglobin RR values differ by maternal age"
        rrs = (
            rrs.loc[rrs.age_start == rrs.age_start.min()]
            .rename(columns={"parameter": "exposure"})
            .reset_index()
        )
        tmred = load_hemoglobin_tmred(key="risk_factor.hemoglobin.tmred", location=location)
        assert tmred["min"] == tmred["max"], "TMRED min and max are not equal"
        tmred = tmred["min"]
        if tmred != metadata.HEMOGLOBIN_TMRED:
            raise ValueError(
                f"TMRED is not equal to expected value {metadata.HEMOGLOBIN_TMRED} g/L. "
                "Please confirm that the RR for the direct effect of neonatal "
                "sepsis has been scaled to the correct TMRED value before proceeding"
            )
        # find RR value for the TMRED using linear interpolation
        interpolated_rows = []
        for entity in rrs["affected_entity"].unique():
            entity_df = rrs[rrs["affected_entity"] == entity].sort_values("exposure")
            interp_row = {"affected_entity": entity}
            for col in [col for col in rrs.columns if col.startswith("draw_")]:
                interp_row[col] = np.interp(tmred, entity_df["exposure"], entity_df[col])
            interpolated_rows.append(interp_row)
        rrs_tmred = pd.DataFrame(interpolated_rows).set_index("affected_entity")
        rrs_scaled = (
            rrs.set_index(["affected_entity", "exposure"])[
                [col for col in rrs.columns if col.startswith("draw_")]
            ]
            / rrs_tmred
        )
        rrs_scaled = rrs_scaled.reset_index()
        rrs_scaled.to_csv(_CODE_DIR / "hemoglobin_rrs_scaled_to_tmred.csv", index=False)
    return rrs_scaled


# initialize_simulation is imported from vivarium_gates_mncnh.data.utilities


def load_direct_nn_sepsis_rrs(location, draw):
    # Note: neonatal sepsis direct effects have already been scaled to the
    # TMRED (see metadata.HEMOGLOBIN_TMRED) during data generation.
    path = _CODE_DIR / "../../hemoglobin_effects/direct_sepsis_effects"
    sepsis_rrs = pd.read_csv(path / f"draw_{draw}.csv")
    sepsis_rrs = sepsis_rrs.loc[sepsis_rrs.location == location]
    sepsis_rrs["child_age_group"] = np.where(
        sepsis_rrs.age_group_id == 2, "early_neonatal", "late_neonatal"
    )
    sepsis_rrs["outcome"] = (
        "neonatal_sepsis_"
        + sepsis_rrs.child_age_group
        + "_"
        + sepsis_rrs.sex_of_child.str.lower()
    )
    sepsis_rrs = sepsis_rrs.set_index(["draw", "location", "exposure", "outcome"])[
        ["value"]
    ].unstack()
    sepsis_rrs.columns = sepsis_rrs.columns.droplevel(0)
    sepsis_rrs = sepsis_rrs.reset_index()
    return sepsis_rrs


def load_rrs(location, draw, population_size, md_rr_data):
    """Load relative risks from the simulation and assign to the population, then stratify
    by GBD age group."""
    sim = initialize_simulation(location, draw, population_size)
    pop = sim.get_population(["age", "hemoglobin.exposure", "pregnancy_outcome"])

    sepsis_rrs = load_direct_nn_sepsis_rrs(location, draw)
    for col in [x for x in sepsis_rrs.columns if "neonatal_sepsis" in x]:
        pop[col] = np.interp(
            pop["hemoglobin.exposure"], sepsis_rrs["exposure"], sepsis_rrs[col]
        )

    # read in and assign maternal disorders RRs
    md_rrs = md_rr_data[["affected_entity", "exposure", f"draw_{draw}"]]
    for cause in md_rrs["affected_entity"].unique():
        if cause != "maternal_hypertensive_disorders":
            pop[f"{cause}_rr"] = np.interp(
                pop["hemoglobin.exposure"],
                md_rrs.loc[md_rrs["affected_entity"] == cause, "exposure"],
                md_rrs.loc[md_rrs["affected_entity"] == cause, f"draw_{draw}"],
            )

    # Use shared maternal age bins from metadata (canonical source for both
    # PAF generation and loader).
    labels = list(metadata.MATERNAL_AGE_MAP.keys())
    bins = sorted({v for bounds in metadata.MATERNAL_AGE_MAP.values() for v in bounds})
    pop["age_group"] = pd.cut(
        pop["age"], bins=bins, labels=labels, right=False, include_lowest=True
    )
    pop["location"] = location
    return pop


def calculate_pafs(location, draw, population_size, md_rr_data):
    """Calculate PAFs for maternal disorders and neonatal sepsis.

    For each GBD age group, calculate the mean RR for maternal hemorrhage and sepsis
    and then calculate the PAFs using the formula PAF = (mean_RR - 1) / mean_RR.

    The PAF for neonatal sepsis is restricted to non partial term pregnancies only.
    We do this because the ratio of partial term pregnancies as well as hemoglobin
    exposure is age-specific, so this will influence the overall hemoglobin exposure
    distribution of all pregnancies that advance to live birth.

    Note that we keep stillbirths here even though they are not at risk of
    neonatal sepsis, however, there should be no difference in hemoglobin exposure
    distribution between stillbirths and live births so we keep them in for
    statistical power.

    Note that we are not (but could) advance the simulation to late neonatal age group
    and then restrict to surviving neonates for the calculation of the neonatal sepsis
    PAF among the late neonatal age group. However, given that hemoglobin has a smaller
    overall effect on neonatal mortality than LBWSG, we hypothesize that this
    limitation in our PAF calculation will not have a large impact on our PAF value
    (we assume that the distribution of hemoglobin of parents of all live births ~= that
    of neonates that survive ENN age group).

    We can revisit this assumption if our simulated LNN mortality does not calibrate
    after the inclusion of hemoglobin effects on NN sepsis. Also note that in that case
    we'd probably want to re-run WITHOUT also calculating the effects on maternal
    disorders again because we could probably get away with a smaller population size.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple of (maternal_paf, neonatal_paf) DataFrames.
    """
    data = load_rrs(location, draw, population_size, md_rr_data)
    md_mean_rr = (
        data.drop(columns=["age", "hemoglobin.exposure", "pregnancy_outcome"])
        .groupby(["location", "age_group"])[
            [x for x in data.columns if "neonatal_sepsis" not in x and "rr" in x]
        ]
        .mean()
    )
    maternal_paf = ((md_mean_rr - 1) / md_mean_rr).reset_index()
    # Rename columns from *_rr to *_paf since these are now PAF values
    maternal_paf = maternal_paf.rename(
        columns={
            c: c.replace("_rr", "_paf") for c in maternal_paf.columns if c.endswith("_rr")
        }
    )

    nn_mean_rr = (
        data.loc[data.pregnancy_outcome == "full_term"]
        .drop(columns=["age", "hemoglobin.exposure", "pregnancy_outcome"])
        .groupby(["location"])[[x for x in data.columns if "neonatal_sepsis" in x]]
        .mean()
    )
    neonatal_paf = ((nn_mean_rr - 1) / nn_mean_rr).reset_index()

    return maternal_paf, neonatal_paf
