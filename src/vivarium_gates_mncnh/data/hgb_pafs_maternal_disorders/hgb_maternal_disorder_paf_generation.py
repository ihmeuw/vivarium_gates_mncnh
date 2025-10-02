import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from vivarium import Artifact, InteractiveContext
from vivarium.framework.configuration import build_model_specification


# This code relies on data specific to:
# 1. The low hemoglobin risk exposure levels (using GBD 2023 data in artifact 16.0)
# 2. The low hemoglobin relative risk values (using GBD 2021 data in artifact 16.0)
# Therefore, it will need to be re-run if any of these are updated

def load_hemoglobin_rrs_on_maternal_disorders():
    # calling this from loader.py instead of the artifact because the latest artifact version available
    # as of 9.30.25 (model 18.0) has version that cuts at 100 draws
    # and we want to upate to 250 draws instead
    # note that eventually we could update this code to use RR values assigned within the simulation
    # or directly from some specified artifact version
        # this would allow us to use the same environment to load the RRs as we use to run the simulation

    if 'hemoglobin_rrs_scaled_to_tmred.csv' in os.listdir():
        rrs_scaled = pd.read_csv('hemoglobin_rrs_scaled_to_tmred.csv')
    else:
        print("NOTE: data for scaled hemoglobin RRs on maternal disorders must be generated using the artifact enviornment prior to running the interactive context using the simulation environment. see code for more details.")
        location = 'Ethiopia'  # RRs do not vary by location, so using this as an arbitrary hard coded value
        from vivarium_gates_mncnh.data.loader import load_hemoglobin_relative_risk, load_hemoglobin_tmred
        rrs = load_hemoglobin_relative_risk(key='risk_factor.hemoglobin.relative_risk', location=location).reset_index()
        assert len([x for x in rrs.columns if 'draw' in x]) == 500, "Hemoglobin RR data does not have 500 draws"
        for i in list(range(0,250)):
            assert np.all(rrs[f'draw_{i}'] == rrs[f'draw_{i + 250}']), "Hemoglobin RR data does not have 250 unique draws"
        assert np.all(rrs.set_index(['parameter','age_start','affected_entity'])[[x for x in rrs.columns if 'draw' in x]].round(5).groupby(['parameter','affected_entity']).nunique() == 1), "Hemoglobin RR values differ by maternal age"
        rrs = rrs.loc[rrs.age_start == rrs.age_start.min()].rename(columns={'parameter':'exposure'}).reset_index()
        tmred = load_hemoglobin_tmred(key='risk_factor.hemoglobin.tmred', location=location)
        assert tmred['min'] == tmred['max'], "TMRED min and max are not equal"
        tmred = tmred['min']
        if tmred != 120.0:
            raise ValueError("TMRED is not equal to expected value 120 g/L. Please confirm that the RR for the direct effect of neonatal sepsis has been scaled to the correct TMRED value before proceeding")
        # find RR value for the TMRED using linear interpolation
        interpolated_rows = []
        for entity in rrs['affected_entity'].unique():
            entity_df = rrs[rrs['affected_entity'] == entity].sort_values('exposure')
            interp_row = {'affected_entity': entity}
            for col in [col for col in rrs.columns if col.startswith('draw_')]:
                interp_row[col] = np.interp(
                    120.0,
                    entity_df['exposure'],
                    entity_df[col]
                )
            interpolated_rows.append(interp_row)
        rrs_tmred = pd.DataFrame(interpolated_rows).set_index('affected_entity')
        rrs_scaled = rrs.set_index(['affected_entity','exposure'])[[col for col in rrs.columns if col.startswith('draw_')]] / rrs_tmred
        rrs_scaled = rrs_scaled.reset_index()
        rrs_scaled.to_csv('hemoglobin_rrs_scaled_to_tmred.csv', index=False)
    return rrs_scaled


def initialize_simulation(location, draw, population_size):
    """This function uses the interactive context to initialze a simulation with the specified
    location and draw and artifact directory."""
    path = Path(os.getcwd() + "/../../model_specifications/model_spec.yaml")
    custom_model_specification = build_model_specification(path)
    del custom_model_specification.configuration.observers
    artifact_directory = custom_model_specification.configuration.input_data.artifact_path
    global artifact_base
    artifact_base = artifact_directory.rsplit("/", 1)[0] + "/"
    custom_model_specification.configuration.input_data.artifact_path = (
        artifact_base + location + ".hdf"
    )
    draw_num = custom_model_specification.configuration.input_data.input_draw_number
    draw = "draw_" + str(draw_num)
    custom_model_specification.configuration.population.population_size = population_size
    sim = InteractiveContext(custom_model_specification)
    return sim

def load_direct_nn_sepsis_rrs(location, draw):
    # note that the neonatal sepsis direct effects have already been scaled to the TMRED of 120 g/L during the data generation process
    path = os.getcwd() + '/../hemoglobin_effects/direct_sepsis_effects/'
    sepsis_rrs = pd.read_csv(path + draw + '.csv')
    sepsis_rrs = sepsis_rrs.loc[sepsis_rrs.location==location]
    sepsis_rrs['child_age_group'] = np.where(sepsis_rrs.age_group_id==2,'early_neonatal', 'late_neonatal')
    sepsis_rrs['outcome'] = 'neonatal_sepsis_' + sepsis_rrs.child_age_group + '_' + sepsis_rrs.sex_of_child.str.lower()
    sepsis_rrs = sepsis_rrs.set_index(['draw','location','exposure','outcome'])[['value']].unstack()
    sepsis_rrs.columns = sepsis_rrs.columns.droplevel(0)
    sepsis_rrs = sepsis_rrs.reset_index()
    return sepsis_rrs

def load_rrs(location, draw, population_size):
    """Load up maternal disorder PAFs and relative risks from the simulation, then stratify
    by GBD age group."""
    sim = initialize_simulation(location, draw, population_size)
    pop = sim.get_population()
    df = pd.concat(
        [
            pop[["age", "hemoglobin_exposure", "pregnancy_outcome"]],
            # read assigned RRs for maternal hemorrhage and maternal sepsis from simulation pipeline values
            sim.get_value("hemoglobin_on_maternal_hemorrhage.relative_risk")(pop.index).rename('maternal_hemorrhage_rr'),
            sim.get_value(
                "hemoglobin_on_maternal_sepsis_and_other_maternal_infections.relative_risk"
            )(pop.index).rename('maternal_sepsis_rr'),
        ],
        axis=1,
    )

    # read in and assign neonatal sepsis rrs
        # NOTE: this strategy uses the hemoglobin exposure distribution among all pregnancies to calculate neonatal sepsis PAFs
        # this makes the assumption that there is no difference in hemoglobin exposure distribution for all pregnancies versus:
            # 1. live births (this assumption should hold in the baseline scenario since baseline IFA does not affect pregnancy outcome)
            # 2. The population of living neonates upon entry to the late neonatal age group
                # This assumption is known to be invalid given that hemoglobin is associated with neonatal mortality
                # through the modeled effect on neonatal sepsis mortality and as mediated through baseline IFA
                # we are hypothesizing that this assumption will have a small effect on the final PAFs and will
                # revisit the assumption if our model does not calibrate
    sepsis_rrs = load_direct_nn_sepsis_rrs(location, draw)
    for col in [x for x in sepsis_rrs.columns if 'neonatal_sepsis' in x]:
        df[col] = np.interp(
            df['hemoglobin_exposure'],
            sepsis_rrs['exposure'],
            sepsis_rrs[col]
        )

    # read in and assign postpartum depression RRs
    ppd_rrs = load_scaled_ppd_rrs(location, draw)
    df['ppd_rr'] = np.interp(
        df['hemoglobin_exposure'],
        ppd_rrs['exposure'],
        ppd_rrs['ppd_rr']
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
    data = load_rrs(location, draw, population_size)
    md_mean_rr = data.drop(columns=['age','hemoglobin_exposure','pregnancy_outcome']).groupby(['location','age_group'])[['maternal_hemorrhage_rr',
                                                                                                     'maternal_sepsis_rr',
                                                                                                     'ppd_rr']].mean()
    # restrict PAF for neonatal sepsis to non partial term pregnancies only
        # we do this because the ratio of partial term pregnancies as well as hemoglobin exposure is age-specific,
        # so this will influence the overall hemoglobin exposure distribution of all pregnancies that advance to live birth
        # note that I am keeping in stillbirths here even though they are not at risk of neonatal sepsis,
        # however, there should be no difference in hemoglobin exposure distribution between stillbirths and live births
        # so I am keeping them in for statistical power
    nn_mean_rr = (data.loc[data.pregnancy_outcome=='live_birth']
                  .drop(columns=['age','hemoglobin_exposure','pregnancy_outcome']).groupby(['location'])[[x for x in data.columns if 'neonatal_sepsis' in x]].mean())
    nn_mean_rr['age_group'] = 'N/A'
    mean_rr = pd.concat([md_mean_rr, nn_mean_rr.reset_index().set_index(['location','age_group'])], axis=0)
    paf = (mean_rr - 1) / mean_rr
    return paf.reset_index()


def save_pafs(draws, locations, population_size, out_dir):
    """Calculate and save PAFs for all specified draws and locations."""

    # list of draws obtained from: https://vivarium-research.readthedocs.io/en/latest/models/concept_models/vivarium_mncnh_portfolio/concept_model.html#id24
    draws = [115, 60, 368, 197, 79, 244, 272, 167, 146, 71, 278, 406, 94, 420, 109, 26, 35, 114, 428, 218]
    for draw in draws:
        data = pd.DataFrame(draw)