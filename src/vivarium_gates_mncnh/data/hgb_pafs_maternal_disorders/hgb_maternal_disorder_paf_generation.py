import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from vivarium import Artifact, InteractiveContext
from vivarium.framework.configuration import build_model_specification

# This code relies on data specific to:
# 1. The low hemoglobin risk exposure levels (using GBD 2023 data in artifact 18.0)
# 2. The low hemoglobin relative risk values (using GBD 2021 data in artifact 18.0)
# 3. The direct effect of neonatal sepsis relative risk values, which are themselves dependent on:
# a. The LBWSG risk factor data (using GBD 2021 data in artifact 18.0)
# b. Hemoglobin risk factor data, specifically the effect on neonatal sepsis (using GBD 2023 data in artifact 18.0)
# Therefore, it will need to be re-run if any of these are updated


def load_hemoglobin_rrs_on_maternal_disorders():
    # calling this from loader.py instead of the artifact because the latest artifact version available
    # as of 9.30.25 (model 18.0) has version that cuts at 100 draws
    # and we want to upate to 250 draws instead
    # note that eventually we could update this code to use RR values assigned within the simulation
    # or directly from some specified artifact version
    # this would allow us to use the same environment to load the RRs as we use to run the simulation

    if "hemoglobin_rrs_scaled_to_tmred.csv" in os.listdir():
        rrs_scaled = pd.read_csv("hemoglobin_rrs_scaled_to_tmred.csv")
    else:
        print(
            "NOTE: data for scaled hemoglobin RRs on maternal disorders must be generated using the artifact enviornment prior to running the interactive context using the simulation environment. see code for more details."
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
        if tmred != 120.0:
            raise ValueError(
                "TMRED is not equal to expected value 120 g/L. Please confirm that the RR for the direct effect of neonatal sepsis has been scaled to the correct TMRED value before proceeding"
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
        rrs_scaled.to_csv("hemoglobin_rrs_scaled_to_tmred.csv", index=False)
    return rrs_scaled


def initialize_simulation(location, draw, population_size):
    """This function uses the interactive context to initialze a simulation with the specified
    location and draw and artifact directory."""
    path = Path(os.getcwd() + "/../../model_specifications/model_spec.yaml")
    custom_model_specification = build_model_specification(path)
    del custom_model_specification.configuration.observers
    artifact_directory = custom_model_specification.configuration.input_data.artifact_path
    artifact_base = artifact_directory.rsplit("/", 1)[0] + "/"
    custom_model_specification.configuration.input_data.artifact_path = (
        artifact_base + location + ".hdf"
    )
    custom_model_specification.configuration.input_data.input_draw_number = draw
    custom_model_specification.configuration.population.population_size = population_size
    sim = InteractiveContext(custom_model_specification)
    return sim


def load_direct_nn_sepsis_rrs(location, draw):
    # note that the neonatal sepsis direct effects have already been scaled to the TMRED of 120 g/L during the data generation process
    path = os.getcwd() + "/../hemoglobin_effects/direct_sepsis_effects/"
    sepsis_rrs = pd.read_csv(path + "draw_" + str(draw) + ".csv")
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
    """Load up maternal disorder PAFs and relative risks from the simulation, then stratify
    by GBD age group."""
    sim = initialize_simulation(location, draw, population_size)
    pop = sim.get_population()[["age", "hemoglobin_exposure", "pregnancy_outcome"]]

    sepsis_rrs = load_direct_nn_sepsis_rrs(location, draw)
    for col in [x for x in sepsis_rrs.columns if "neonatal_sepsis" in x]:
        pop[col] = np.interp(
            pop["hemoglobin_exposure"], sepsis_rrs["exposure"], sepsis_rrs[col]
        )

    # read in and assign maternal disorders RRs
    md_rrs = md_rr_data[["affected_entity", "exposure", f"draw_{draw}"]]
    for cause in md_rrs["affected_entity"].unique():
        if cause != "maternal_hypertensive_disorders":
            pop[f"{cause}_rr"] = np.interp(
                pop["hemoglobin_exposure"],
                md_rrs.loc[md_rrs["affected_entity"] == cause, "exposure"],
                md_rrs.loc[md_rrs["affected_entity"] == cause, f"draw_{draw}"],
            )

    bins = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    labels = [
        "10_to_14",
        "15_to_19",
        "20_to_24",
        "25_to_29",
        "30_to_34",
        "35_to_39",
        "40_to_44",
        "45_to_49",
        "50_to_54",
    ]

    pop["age_group"] = pd.cut(
        pop["age"], bins=bins, labels=labels, right=False, include_lowest=True
    )
    pop["location"] = location
    return pop


def calculate_pafs(location, draw, population_size, md_rr_data):
    """For each GBD age group, calculate the mean RR for maternal hemorrhage and sepsis
    and then calculate and save the PAFs using the formula PAF = (mean_RR-1)/mean_RR"""
    data = load_rrs(location, draw, population_size, md_rr_data)
    md_mean_rr = (
        data.drop(columns=["age", "hemoglobin_exposure", "pregnancy_outcome"])
        .groupby(["location", "age_group"])[
            [x for x in data.columns if "neonatal_sepsis" not in x and "rr" in x]
        ]
        .mean()
    )
    # restrict PAF for neonatal sepsis to non partial term pregnancies only
    # we do this because the ratio of partial term pregnancies as well as hemoglobin exposure is age-specific,
    # so this will influence the overall hemoglobin exposure distribution of all pregnancies that advance to live birth
    # note that I am keeping in stillbirths here even though they are not at risk of neonatal sepsis,
    # however, there should be no difference in hemoglobin exposure distribution between stillbirths and live births
    # so I am keeping them in for statistical power
    # note that we are not (but could) advance the simulation to late neonatal age group and then restrict to surviving neonates
    # for the calculation of the neonatal sepsis PAF among the late neonatal age group
    # however, given that hemoglobin has a smaller overall effect on neonatal mortality than LBWSG,
    # we will hypothesize that this limitation in our PAF calculation will not have a large impact on our PAF value
    # (we assume that the distribution of hemoglobin of parents of all live births ~= that of neonates that survive ENN age group)
    # we can revisit this assumption if our simulated LNN mortality does not calibrate after the inclusion of hemoglobin effects on NN sepsis
    # also note that in that case we'd probably want to re-run WITHOUT also calculating the effects on maternal disorders again because we could probably get away with a smaller population size
    nn_mean_rr = (
        data.loc[data.pregnancy_outcome == "live_birth"]
        .drop(columns=["age", "hemoglobin_exposure", "pregnancy_outcome"])
        .groupby(["location"])[[x for x in data.columns if "neonatal_sepsis" in x]]
        .mean()
    )
    nn_mean_rr["age_group"] = "N/A"
    mean_rr = pd.concat(
        [md_mean_rr, nn_mean_rr.reset_index().set_index(["location", "age_group"])], axis=0
    )
    paf = (mean_rr - 1) / mean_rr
    return paf.reset_index()
