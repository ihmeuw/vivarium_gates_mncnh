import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from vivarium import Artifact, InteractiveContext
from vivarium.framework.configuration import build_model_specification

# note that if the artifact is out of date with the simulation status in the environment
# used to run this code then you may get an error when initializing the interactive context
# and you will need to update this artifact directory to a more recent version
artifact_directory = (
    "/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/artifacts/model16.0/"
)
# This code relies on data specific to:
# 1. The low hemoglobin risk exposure levels (using GBD 2023 data in artifact 16.0)
# 2. The low hemoglobin relative risk values (using GBD 2021 data in artifact 16.0)
# Therefore, it will need to be re-run if any of these are updated


def get_simulated_population(location, draw, population_size):
    """This function uses the interactive context to initialze a simulation with the specified
    location and draw and artifact directory."""
    path = Path(os.getcwd() + "/../../model_specifications/model_spec.yaml")
    custom_model_specification = build_model_specification(path)
    del custom_model_specification.configuration.observers
    custom_model_specification.configuration.input_data.artifact_path = (
        artifact_directory + location + ".hdf" 
    )
    draw_num = custom_model_specification.configuration.input_data.input_draw_number
    draw = "draw_" + str(draw_num)
    custom_model_specification.configuration.population.population_size = population_size
    sim = InteractiveContext(custom_model_specification)

    pop = sim.get_population()
    cols = ["alive", "sex", "age", "hemoglobin_exposure"]
    # Return both population and simulation object
    return pop[cols], sim


def load_maternal_disorders(location, draw, population_size):
    """Load up maternal disorder PAFs and relative risks from the simulation, then stratify
    by GBD age group."""
    pop, sim = get_simulated_population(location, draw, population_size)
    df = pd.concat(
        [
            pop[["alive", "sex", "age", "hemoglobin_exposure"]],
            sim.get_value("hemoglobin.exposure")(pop.index),
            sim.get_value("hemoglobin_on_maternal_hemorrhage.relative_risk")(pop.index),
            sim.get_value(
                "hemoglobin_on_maternal_sepsis_and_other_maternal_infections.relative_risk"
            )(pop.index),
            sim.get_value("maternal_hemorrhage.incidence_risk")(pop.index),
            sim.get_value("maternal_hemorrhage.incidence_risk.paf")(pop.index).rename(
                "hemorrhage_paf"
            ),
            sim.get_value("maternal_sepsis_and_other_maternal_infections.incidence_risk")(
                pop.index
            ),
            sim.get_value("maternal_sepsis_and_other_maternal_infections.incidence_risk.paf")(
                pop.index
            ).rename("sepsis_paf"),
        ],
        axis=1,
    )
    bins = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55] 
    labels = ['10_to_14', '15_to_19', '20_to_24', '25_to_29', '30_to_34', 
              '35_to_39', '40_to_44', '45_to_49', '50_to_54']
    
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False, include_lowest=True)
    df["location"] = location
    return df

def calculate_pafs(location, draw, population_size):
    """For each GBD age group, calculate the mean RR for maternal hemorrhage and sepsis
    and then calculate and save the PAFs using the formula PAF = (mean_RR-1)/mean_RR"""
    data = load_maternal_disorders(location, draw, population_size)
    
    # Calculate hemorrhage PAFs
    hemorrhage_mean_rr = data.groupby("age_group")[
        "hemoglobin_on_maternal_hemorrhage.relative_risk"
    ].mean()
    hemorrhage_paf = (hemorrhage_mean_rr - 1) / hemorrhage_mean_rr
    hemorrhage_df = hemorrhage_paf.reset_index()
    hemorrhage_df.columns = ['age_group', 'paf']
    hemorrhage_df['location'] = location
    hemorrhage_df.to_csv(f"hemorrhage_{draw}.csv", index=False)
    
    # Calculate sepsis PAFs
    sepsis_mean_rr = data.groupby("age_group")[
        "hemoglobin_on_maternal_sepsis_and_other_maternal_infections.relative_risk"
    ].mean()
    sepsis_paf = (sepsis_mean_rr - 1) / sepsis_mean_rr
    sepsis_df = sepsis_paf.reset_index()
    sepsis_df.columns = ['age_group', 'paf']
    sepsis_df['location'] = location
    sepsis_df.to_csv(f"sepsis_{draw}.csv", index=False)

    # TODO: Add calculation of postpartum depression PAFs here when ready
    
    return hemorrhage_df, sepsis_df
