import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from hgb_birth_effect_generation import *
from vivarium import Artifact, InteractiveContext
from vivarium.framework.configuration import build_model_specification
from vivarium_public_health.risks.implementations.low_birth_weight_and_short_gestation import (
    LBWSGRisk as LBWSGRisk_,
)
from vivarium_public_health.risks.implementations.low_birth_weight_and_short_gestation import (
    LBWSGRiskEffect as LBWSGRiskEffect_,
)

artifact_directory = (
    "/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/artifacts/model13.1/"
)
# This code relies on data specific to:
# 1. The LBWSG birth exposure in GBD (using GBD 2021 data in artifact 13.1)
# 2. The hemoglobin risk exposure levels (using GBD 2023 data in artifact 13.1)
# 3. The LBWSG relative risk values (using GBD 2021 data in artifact 13.1)
# Therefore, it will need to be re-run if any of these are updated


def get_simulated_population(location, draw):
    """This function uses the interactive context to initialze a simulation with the specified
    location and draw and artifact directory. It then advances the simulation past the early
    neonatal mortality timestep."""
    path = Path(os.getcwd() + "/../../model_specifications/model_spec.yaml")
    custom_model_specification = build_model_specification(path)
    del custom_model_specification.configuration.observers
    custom_model_specification.configuration.input_data.artifact_path = (
        artifact_directory + location + ".hdf"
    )
    custom_model_specification.configuration.input_data.input_draw_number = draw
    # NOTE: setting population size to what we are using in the simulation for a single draw
    custom_model_specification.configuration.population.population_size = 20_000 * 10
    sim = InteractiveContext(custom_model_specification)
    # WARNING/TO-DO: this needs to be manually maintained
    # as of model 13.3, it takes 12 steps to advance past early neonatal mortality
    # if/when additional timesteps are added to the simulation, this number will need to be changed accordingly
    sim.step()
    sim.step()
    sim.step()
    sim.step()
    sim.step()
    sim.step()
    sim.step()
    sim.step()
    sim.step()
    sim.step()
    sim.step()
    sim.step()

    pop = sim.get_population()
    cols = [
        "sex_of_child",
        "pregnancy_outcome",
        "child_alive",
        "birth_weight_exposure",
        "gestational_age_exposure",
        "effect_of_low_birth_weight_and_short_gestation_on_early_neonatal_neonatal_sepsis_and_other_neonatal_infections_relative_risk",
        "effect_of_low_birth_weight_and_short_gestation_on_late_neonatal_neonatal_sepsis_and_other_neonatal_infections_relative_risk",
    ]
    return pop[cols]


def load_interpolators(location, draw):
    """Load up LBWSG risk interpolators"""
    art = Artifact(
        artifact_directory + location + ".hdf", filter_terms={"child_age_end < 0.5"}
    )
    interpolators = art.load(
        "risk_factor.low_birth_weight_and_short_gestation.relative_risk_interpolator"
    )[f"draw_{draw}"].reset_index()
    interpolators["age_group_id"] = np.where(interpolators["child_age_start"] == 0, 2, 3)
    interpolators = (
        interpolators.drop(
            columns=["child_age_start", "child_age_end", "year_start", "year_end"]
        ).set_index(["sex_of_child", "age_group_id"])
    ).reset_index()
    interpolators = interpolators.set_index(["sex_of_child", "age_group_id"])[f"draw_{draw}"]
    interpolators = interpolators.apply(lambda x: pickle.loads(bytes.fromhex(x)))
    return interpolators


def load_lbwsg_shifts(location, draw):
    """Read in effect of hemoglobin on gestational age and birthweight, as calculated and saved separately"""
    data = pd.read_csv(os.getcwd() + f"/lbwsg_shifts/draw_{draw}.csv")
    data = data.loc[data.location == location]
    return data


def calculate_indirect_effect(location, draw):
    """For each hemoglobin exposure level, calculate the effect of hemoglobin on neonatal sepsis
    mortality as mediated through LBWSG at the age and sex specific level.
    This is done by comparing the individual LBWSG
    RR values among the simulated population without modification to the LBWSG RR values that have
    been shifted according to the birth weight and gestational age shift values for a specific
    hemoglobin exposure level (calculated separately). This is done under the assumption that the
    GBD population-level LBWSG exposure distribution is the same as the LBWSG exposure distribution
    among the population with a hemoglobin exposure level equal to the hemoglobin TMREL exposure."""

    data = get_simulated_population(location, draw)
    data["enn_pop"] = data.pregnancy_outcome == "live_birth"
    data["lnn_pop"] = (data.pregnancy_outcome == "live_birth") & (data.child_alive == "alive")

    interpolators = load_interpolators(location, draw)
    shifts = load_lbwsg_shifts(location, draw).rename(columns={"sex": "sex_of_child"})
    shifts["outcome"] = shifts.outcome + "_shift"
    shifts = shifts.pivot_table(
        index=["sex_of_child", "exposure"], values="value", columns="outcome"
    ).reset_index()

    is_male = data.sex_of_child == "Male"
    exposure_levels = shifts.exposure.unique()

    result = pd.DataFrame()

    for exposure_level in exposure_levels:
        sub_shifts = shifts.loc[shifts.exposure == exposure_level]
        sub_data = data.merge(sub_shifts, on="sex_of_child")
        gestational_age = sub_data.gestational_age_exposure + sub_data.gestational_age_shift
        birth_weight = sub_data.birth_weight_exposure + sub_data.birth_weight_shift

        enn_log_relative_risk = pd.Series(0.0, index=data.index, name="interpolated_rr")
        lnn_log_relative_risk = pd.Series(0.0, index=data.index, name="interpolated_rr")
        enn_log_relative_risk[is_male] = interpolators["Male", 2](
            gestational_age[is_male], birth_weight[is_male], grid=False
        )
        enn_log_relative_risk[~is_male] = interpolators["Female", 2](
            gestational_age[~is_male], birth_weight[~is_male], grid=False
        )
        lnn_log_relative_risk[is_male] = interpolators["Male", 3](
            gestational_age[is_male], birth_weight[is_male], grid=False
        )
        lnn_log_relative_risk[~is_male] = interpolators["Female", 3](
            gestational_age[~is_male], birth_weight[~is_male], grid=False
        )
        enn_relative_risk = np.exp(enn_log_relative_risk)
        lnn_relative_risk = np.exp(lnn_log_relative_risk)
        sub_data["enn_indirect_effect"] = (
            enn_relative_risk
            / sub_data.effect_of_low_birth_weight_and_short_gestation_on_early_neonatal_neonatal_sepsis_and_other_neonatal_infections_relative_risk
        )
        sub_data["lnn_indirect_effect"] = (
            lnn_relative_risk
            / sub_data.effect_of_low_birth_weight_and_short_gestation_on_late_neonatal_neonatal_sepsis_and_other_neonatal_infections_relative_risk
        )
        enn_indirect_effect = (
            sub_data.loc[sub_data.enn_pop]
            .groupby("sex_of_child")
            .enn_indirect_effect.mean()
            .reset_index()
            .rename(columns={"enn_indirect_effect": "value"})
        )
        enn_indirect_effect["age_group_id"] = 2
        lnn_indirect_effect = (
            sub_data.loc[sub_data.lnn_pop]
            .groupby("sex_of_child")
            .lnn_indirect_effect.mean()
            .reset_index()
            .rename(columns={"lnn_indirect_effect": "value"})
        )
        lnn_indirect_effect["age_group_id"] = 3
        temp = pd.concat([enn_indirect_effect, lnn_indirect_effect], ignore_index=True)
        temp["exposure"] = exposure_level
        result = pd.concat([result, temp], ignore_index=True)

    result["location"] = location
    result["draw"] = f"draw_{draw}"
    return result


def calculate_direct_effect(results_directory, location, draw):
    """Calculate and save the direct (unmediated) effect of hemoglobin on neonatal sepsis mortality.
    This is done by dividing the total effect by the indirect effect."""
    indirect_rrs = calculate_indirect_effect(location, draw)
    total_rrs = load_prepped_rrs("neonatal_sepsis")
    total_effects_prepped = (
        total_rrs.set_index(["outcome", "risk"])
        .stack()
        .reset_index()
        .rename(columns={"level_2": "draw", 0: "value", "risk": "exposure"})
    )
    total_effects_prepped = total_effects_prepped.loc[
        total_effects_prepped.draw == f"draw_{draw}"
    ]
    direct_rrs = (
        total_effects_prepped.set_index(["outcome", "draw", "exposure"])
        / indirect_rrs.set_index([x for x in indirect_rrs.columns if x != "value"])
    ).reset_index()
    direct_rrs.to_csv(
        f"{results_directory}/direct_sepsis_effects/draw_{draw}.csv", index=False
    )
    indirect_rrs.to_csv(
        f"{results_directory}/indirect_sepsis_effects/draw_{draw}.csv", index=False
    )
    return direct_rrs, indirect_rrs
