import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from hgb_birth_effect_generation import *
from vivarium import Artifact, InteractiveContext
from vivarium.framework.configuration import build_model_specification

# note that if the artifact is out of date with the simulation status in the environment
# used to run this code then you may get an error when initializing the interactive context
# and you will need to update this artifact directory to a more recent version
artifact_directory = (
    "/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/artifacts/model15.0/"
)
# This code relies on data specific to:
# 1. The low hemoglobin risk exposure levels (using GBD 2023 data in artifact 15.0)
# 2. The low hemoglobin relative risk values (using GBD 2021 data in artifact 15.0)
# Therefore, it will need to be re-run if any of these are updated


def get_simulated_population(location, draw):
    """This function uses the interactive context to initialze a simulation with the specified
    location and draw and artifact directory."""
    path = Path(os.getcwd() + "/../../model_specifications/model_spec.yaml")
    custom_model_specification = build_model_specification(path)
    del custom_model_specification.configuration.observers
    custom_model_specification.configuration.input_data.artifact_path = (
        artifact_directory + location + ".hdf"
    )
    draw_num = custom_model_specification.configuration.input_data.input_draw_number
    draw = 'draw_' + str(draw_num)
    # NOTE: setting population size to what we are using in the simulation for a single draw
    gbd_draw = 'draw_' + str(draw_num % 100)
    # NOTE: We use only the first 100 draws from GBD, repeating them for later draws.
    custom_model_specification.configuration.population.population_size = 20_000 * 10
    sim = InteractiveContext(custom_model_specification)


    pop = sim.get_population()
    cols = [
        'alive','sex','age','hemoglobin_exposure'
    ]
    return pop[cols]


def load_maternal_disorders(location, draw):
    """Load up maternal disorder PAFs and relative risks from the simulation, then stratigy
    by GBD age group."""
    df = pd.concat([pop[['alive','sex','age','hemoglobin_exposure']],
                sim.get_value('hemoglobin.exposure')(pop.index),
                sim.get_value('hemoglobin_on_maternal_hemorrhage.relative_risk')(pop.index),
                sim.get_value('hemoglobin_on_maternal_sepsis_and_other_maternal_infections.relative_risk')(pop.index),
                sim.get_value('maternal_hemorrhage.incidence_risk')(pop.index),
                sim.get_value('maternal_hemorrhage.incidence_risk.paf')(pop.index).rename('hemorrhage_paf'),
                sim.get_value('maternal_sepsis_and_other_maternal_infections.incidence_risk')(pop.index),
                sim.get_value('maternal_sepsis_and_other_maternal_infections.incidence_risk.paf')(pop.index).rename('sepsis_paf'),

                ], axis=1)
    
    def assign_gbd_age_group(age):
    if 10 <= age < 15:
        return '10_to_14'
    elif 15 <= age < 20:
        return '15_to_19'
    elif 20 <= age < 25:
        return '20_to_24'
    elif 25 <= age < 30:
        return '25_to_29'
    elif 30 <= age < 35:
        return '30_to_34'
    elif 35 <= age < 40:
        return '35_to_39'
    elif 40 <= age < 45:
        return '40_to_44'
    elif 45 <= age < 50:
        return '45_to_49'
    else:
        return 'other'
    
    df['age_group'] = df['age'].apply(assign_gbd_age_group)
    return df

def calculate_paf_hemorrhage(location, draw):
    """For each GBD age group, calculate the mean RR for maternal hemorrhage 
    and then calculate and save the PAF using the formula PAF = (mean_RR-1)/mean_RR"""
    data = load_maternal_disorders(location, draw)
    hemorrhage_mean_rr = data.groupby('age_group')['hemoglobin_on_maternal_hemorrhage.relative_risk'].mean()
    hemorrhage_paf = (hemorrhage_mean_rr - 1) / hemorrhage_mean_rr 
    hemorrhage_paf.to_csv(
        f"{results_directory}/hgb_hemorrhage_paf/draw_{draw}.csv", index=False
    )
    return hemorrhage_paf

def calculate_paf_sepsis(location, draw):
    """For each GBD age group, calculate the mean RR for maternal sepsis
    and then calculate and save the PAF using the formula PAF = (mean_RR-1)/mean_RR"""
    data = load_maternal_disorders(location, draw)
    sepsis_mean_rr = data.groupby('age_group')['hemoglobin_on_maternal_sepsis_and_other_maternal_infections.relative_risk'].mean()
    sepsis_paf = (sepsis_mean_rr  - 1) / sepsis_mean_rr 
    sepsis_paf.to_csv(
        f"{results_directory}/hgb_sepsis_paf/draw_{draw}.csv", index=False
    )
    return sepsis_paf

  