import pandas as pd, numpy as np, os
from vivarium import Artifact
import scipy.stats as stats
import scipy
from pathlib import Path

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from vivarium import InteractiveContext
from vivarium.framework.configuration import build_model_specification

from IPython.display import display

LOCATION_DATA = {'ethiopia':179,
            'nigeria':214,
            'pakistan':165}
ART_DIR = "/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/artifacts/model22.0.2/"

def load_artifact(location):
    return Artifact(Path(ART_DIR) / (location + ".hdf"))


# this data comes from Annie Haakkenstad's Health Systems estimates and represents oral iron in pregnancy among those who attend ANC 
# docs https://vivarium-research.readthedocs.io/en/latest/models/intervention_models/mncnh_pregnancy/oral_iron_antenatal/oral_iron_antenatal.html#baseline-coverage-data
def load_ifa(location_ids):
    ifa = pd.read_csv('/snfs1/Project/simulation_science/mnch_grant/MNCNH portfolio/anc_iron_prop_st-gpr_results_aggregates_scaled2025-05-30.csv')
    ifa = ifa.loc[(ifa.location_id.isin(location_ids)) 
                & (ifa.year_id==2023)]
    ifa = ifa.set_index('location_id')[[x for x in ifa.columns if 'draw' in x]]
    ifa = ifa.stack().reset_index()
    ifa.columns = ['location_id', 'draw', 'value']
    return ifa

# use artifact data for ANC1 (key 'covariate.antenatal_care_1_visit_coverage_proportion.estimate')
def load_anc_art(art, location):
    anc = art.load('covariate.antenatal_care_1_visit_coverage_proportion.estimate').rename_axis("draw", axis=1).stack().rename("value").reset_index()
    anc["location_id"] = LOCATION_DATA[location]
    anc = anc[["draw", "location_id", "value"]]
    anc = anc.set_index(list(anc.columns[:-1]))
    return anc

def load_baseline_ifa(anc, ifa):
    baseline = (anc * ifa.set_index(["draw","location_id"]))
    return baseline[baseline["value"].notna()].reset_index()  # anc only has 250 draws

# generate lognormal distribution of IFA OR
def lognorm_from_median_lower_upper(median, lower, upper, quantile_ranks=(0.025,0.975)):
  """Returns a frozen lognormal distribution with the specified median, such that
  the values (lower, upper) are approximately equal to the quantiles with ranks
  (quantile_ranks[0], quantile_ranks[1]). More precisely, if q0 and q1 are
  the quantiles of the returned distribution with ranks quantile_ranks[0]
  and quantile_ranks[1], respectively, then q1/q0 = upper/lower. If the
  quantile ranks are symmetric about 0.5, lower and upper will coincide with
  q0 and q1 precisely when median^2 = lower*upper.
  """
  # Let Y ~ Norm(mu, sigma^2) and X = exp(Y), where mu = log(median)
  # so X ~ Lognorm(s=sigma, scale=exp(mu)) in scipy's notation.
  # We will determine sigma from the two specified quantiles lower and upper.

  # mean (and median) of the normal random variable Y = log(X)
  mu = np.log(median)
  # quantiles of the standard normal distribution corresponding to quantile_ranks
  stdnorm_quantiles = stats.norm.ppf(quantile_ranks)
  # quantiles of Y = log(X) corresponding to the quantiles (lower, upper) for X
  norm_quantiles = np.log([lower, upper])
  # standard deviation of Y = log(X) computed from the above quantiles for Y
  # and the corresponding standard normal quantiles
  sigma = (norm_quantiles[1] - norm_quantiles[0]) / (stdnorm_quantiles[1] - stdnorm_quantiles[0])
  # Frozen lognormal distribution for X = exp(Y)
  # (s=sigma is the shape parameter; the scale parameter is exp(mu), which equals the median)
  return stats.lognorm(s=sigma, scale=median)

# This doesn't exist in the artifact and so can be kept in the nb
def load_ifa_rr_draws(dist):
    ifa_rr_draws = pd.DataFrame()
    n_draws = 250
    ifa_rr_draws['draw'] = [f'draw_{x}' for x in list(range(0,n_draws))]
    ifa_rr_draws['rr'] = dist.rvs(size=n_draws, random_state=456)
    return ifa_rr_draws

# Load Nathaniel's CSV that map LBWSG categories to BW and GA continuous values

def load_art_exposure():
    art_exposure = pd.read_csv('lbwsg_category_data_gbd_2021.csv').rename(columns={'lbwsg_category':'parameter'})
    return art_exposure[['parameter','ga_start','ga_end','bw_start','bw_end']] # Only keep the columns that we need right now 

def initialize_simulation(location, draw):    
    no_oral_iron_spec = build_model_specification(Path("../../model_specifications/model_spec.yaml"))
    # Filter out components with 'OralIron' 'iron_folic_acid' or 'multiple_micronutrient_supplementation' in the name
    no_oral_iron_spec.components.vivarium_gates_mncnh.components = [
        c for c in no_oral_iron_spec.components.vivarium_gates_mncnh.components
        if 'OralIron' not in c and 'iron_folic_acid' not in c and 'multiple_micronutrient_supplementation' not in c
    ]
    # specify draw number
    no_oral_iron_spec.configuration.input_data.input_draw_number = draw
    # specify artifact path/location
    no_oral_iron_spec.configuration.input_data.artifact_path = (Path(ART_DIR) / (location + ".hdf"))    
    sim = InteractiveContext(no_oral_iron_spec)
    sim.take_steps(4, None, True)
    return sim

def lookup_continuous_bw_ga(row, art_exposure):
    for (param, ga_start, ga_end, bw_start, bw_end) in art_exposure.itertuples(index=False, name=None):
        bw = row["birth_weight_exposure"]
        ga = row["gestational_age_exposure"]
        if bw>=bw_start and bw<bw_end and ga>=ga_start and ga<ga_end:
            return param
        
def return_sim_data(location, draw, art_exposure):
    sim = initialize_simulation(location, draw)
    pop = sim.get_population()
    df = pop[pop["pregnancy_outcome"] == "live_birth"].copy()
    df["parameter"] = df.apply(lookup_continuous_bw_ga, axis=1, args=(art_exposure,))
    df["anc1"] = df["anc_attendance"] != "none"
    df = df.groupby('anc1').parameter.value_counts(normalize=True).rename('exposure').reset_index()
    # check expsure values sum to 1
    assert np.all(df.groupby('anc1').exposure.sum() == 1), 'LBWSG exposure values across categories do not sum to 1'
    # now check to see that we have all expected categories represented in our data
    if len([x for x in art_exposure.parameter.unique() if x not in df.parameter.unique()]) > 0:
        print(f"WARNING: not all LBWSG exposure categories represented in interactive simulation population for draw {draw} in {location}. Consider increasing population size and rerunning")
    df['draw'] = 'draw_' + str(draw)
    df['location'] = location
    return df

def calculate_rrs_from_shifts(shifts):
    shift_true = shifts[0]
    shift_false = shifts[1]
    weighted_avg_shift = (shift_true * anc_coverage + shift_false * (1 - anc_coverage))
    exp_uncovered_no_anc = (no_anc - weighted_avg_shift * coverage)
    exp_uncovered_no_anc["frac_ptb"] = np.where(exp_uncovered_no_anc.ga_start >= 37, 0,
                                                np.where(exp_uncovered_no_anc.ga_end < 37, 1,
                                                    (37 - exp_uncovered_no_anc.ga_start)/(exp_uncovered_no_anc.ga_end - exp_uncovered_no_anc.ga_start)
                                                    ))
    uncovered_no_anc_ptb = (exp_uncovered_no_anc.reset_index().exposure * exp_uncovered_no_anc.reset_index().frac_ptb).sum()

    exp_covered_no_anc = (exp_uncovered_no_anc + shift_false)
    exp_covered_no_anc["frac_ptb"] = np.where(exp_covered_no_anc.ga_start >= 37, 0,
                                                np.where(exp_covered_no_anc.ga_end < 37, 1,
                                                    (37 - exp_covered_no_anc.ga_start)/(exp_covered_no_anc.ga_end - exp_covered_no_anc.ga_start)
                                                    ))
    covered_no_anc_ptb = (exp_covered_no_anc.reset_index().exposure * exp_covered_no_anc.reset_index().frac_ptb).sum()

    exp_uncovered_anc = (anc - weighted_avg_shift * coverage)
    exp_uncovered_anc["frac_ptb"] = np.where(exp_uncovered_anc.ga_start >= 37, 0,
                                                np.where(exp_uncovered_anc.ga_end < 37, 1,
                                                    (37 - exp_uncovered_anc.ga_start)/(exp_uncovered_anc.ga_end - exp_uncovered_anc.ga_start)
                                                    ))
    uncovered_anc_ptb = (exp_uncovered_anc.reset_index().exposure * exp_uncovered_anc.reset_index().frac_ptb).sum()
    
    exp_covered_anc = (exp_uncovered_anc + shift_true)
    exp_covered_anc["frac_ptb"] = np.where(exp_covered_anc.ga_start >= 37, 0,
                                            np.where(exp_covered_anc.ga_end < 37, 1,
                                                (37 - exp_covered_anc.ga_start)/(exp_covered_anc.ga_end - exp_covered_anc.ga_start)
                                                ))
    covered_anc_ptb = (exp_covered_anc.reset_index().exposure * exp_covered_anc.reset_index().frac_ptb).sum()
    
    rr_anc = covered_anc_ptb / uncovered_anc_ptb
    rr_no_anc = covered_no_anc_ptb / uncovered_no_anc_ptb
    return rr_anc, rr_no_anc

def generate_location_draw_global_data(location, draw, anc_art, baseline_ifa, art_exposure, ifa_rr_draws):
    df = return_sim_data(location, draw, art_exposure)
    data = df.merge(ifa_rr_draws, on='draw')
    location_id = LOCATION_DATA[location]
    data['location_id'] = location_id
    data = data.merge(baseline_ifa.rename(columns={'value':'ifa_coverage'}), on=['location_id', 'draw'])
    data = data.merge(art_exposure, on='parameter')
    exp = data.loc[(data.draw==f'draw_{draw}')&(data.location_id==location_id)]
    rr = data.rr.values[0]
    coverage = exp['ifa_coverage'].values[0]
    anc = exp[exp["anc1"] == True].set_index([c for c in data.columns if 'ga_' not in c]) # anc from interactive sim - gestational age columns are the operative ones (shift gets added to these)
    no_anc = exp[exp["anc1"] == False].set_index([c for c in data.columns if 'ga_' not in c])
    anc_coverage = anc_art.loc[f'draw_{draw}',location_id].value
    return coverage, anc, no_anc, anc_coverage, rr

def perform_single_optimization():
    def shift_optimization(shifts):
        rr_anc, rr_no_anc = calculate_rrs_from_shifts(shifts)
        return (np.abs(rr_anc - rr) + np.abs(rr_no_anc - rr))* 500        
    return scipy.optimize.minimize(shift_optimization, [0, 0], tol=10e-4, method='Nelder-Mead')
    
def check_single_optimization_result(location, draw):
    optimization_result = perform_single_optimization()
    rrs = calculate_rrs_from_shifts(optimization_result.x)
    assert optimization_result.success, f"Optimization failed for {location} draw {draw}"
    return optimization_result.x, rrs

def store_optimization_results(location, draw, anc_art, baseline_ifa, art_exposure, ifa_rr_draws):
    global coverage, anc, no_anc, anc_coverage, rr
    coverage, anc, no_anc, anc_coverage, rr = generate_location_draw_global_data(location, draw, anc_art, baseline_ifa, art_exposure, ifa_rr_draws)
    shifts, rrs = check_single_optimization_result(location, draw)
    output = pd.DataFrame()
    output['draw'] = [draw]
    output['location_id'] = LOCATION_DATA[location]
    output['shift_anc'] = shifts[0]
    output['shift_no_anc'] = shifts[1]
    output['rr_anc'] = rrs[0]
    output['rr_no_anc'] = rrs[1]
    output['rr_target'] = rr
    return output, anc.copy()

def run_location_draws(location, draws, args_all_locations):
    (ifa, art_exposure, ifa_rr_draws) = args_all_locations
    anc_art = load_anc_art(load_artifact(location), location)
    baseline_ifa = load_baseline_ifa(anc_art, ifa[ifa["location_id"]==LOCATION_DATA[location]])
    results = pd.DataFrame()
    anc_exp = pd.DataFrame()
    
    for draw in draws:
        output, exp_res = store_optimization_results(location, draw, anc_art, baseline_ifa, art_exposure, ifa_rr_draws)
        results = pd.concat([results, output], ignore_index=True)
        anc_exp = pd.concat([anc_exp, exp_res])

    return results, anc_exp


def run_all_locations(draws_to_run):
    ifa = load_ifa(list(LOCATION_DATA.values()))
    art_exposure = load_art_exposure()
    # IRA OR: 0.9 (0.86, 0.95) relative to no IFA
    # MMS RR: 0.91 (0.84, 0.99) relative to IFA... UPDATED FROM: 0.95 (0.90, 1.01) relative to IFA
    # MMS VERY PRETERM: RR = 0.81 (0.71, 0.93) relative to IFA
    dist = lognorm_from_median_lower_upper(0.9, 0.86, 0.96, quantile_ranks=(0.025,0.975))
    ifa_rr_draws = load_ifa_rr_draws(dist)
    args_all_locations = (ifa, art_exposure, ifa_rr_draws)

    results = pd.DataFrame()
    anc_exp = pd.DataFrame()

    for location in list(LOCATION_DATA.keys()):
        res, exp_res = run_location_draws(location, draws_to_run, args_all_locations)
        results = pd.concat([results, res], ignore_index=True)
        anc_exp = pd.concat([anc_exp, exp_res])

    return results, anc_exp

def load_mms_data(exp_anc, ifa_shifts):
    baseline_ifa = pd.concat([load_anc_art(load_artifact(location), location) for location in list(LOCATION_DATA.keys())]).reset_index()
    anc_ifa_shifts = ifa_shifts.iloc[:, :3].rename(columns={'shift_anc': 'value'}) # get shift_anc
    anc_ifa_shifts['draw'] = 'draw_' + anc_ifa_shifts['draw'].astype(str)
    
    exp_prepped = exp_anc.set_index([c for c in exp_anc.columns if 'draw' not in c]).stack().reset_index()
    exp_prepped = exp_prepped.drop(["level_12"], axis=1).rename(columns={0:'draw'})
    '''
    uncovered_shift = (-anc_ifa_shifts.set_index(['location_id','draw']) * baseline_ifa.set_index(['location_id','draw'])).dropna()
    uncovered_shift = uncovered_shift.reset_index().rename(columns={'value':'uncovered_shift'})
    uncovered_exp = exp_prepped.merge(uncovered_shift, on=['location_id','draw'])
    uncovered_exp['ga_start'] = uncovered_exp.ga_start + uncovered_exp.uncovered_shift
    uncovered_exp['ga_end'] = uncovered_exp.ga_end + uncovered_exp.uncovered_shift
    uncovered_exp['frac_ptb'] = np.where(uncovered_exp.ga_start >= 37, 0,
                                        np.where(uncovered_exp.ga_end < 37, 1, 
                                                (37 - uncovered_exp.ga_start) / (uncovered_exp.ga_end - uncovered_exp.ga_start)))
    uncovered_exp['exposure_ptb'] = uncovered_exp.frac_ptb * uncovered_exp.exposure
    uncovered_ptb = uncovered_exp.groupby(['location_id','draw']).sum()[['exposure_ptb']]
    '''

    covered_shift = (-anc_ifa_shifts.set_index(['location_id','draw']) * baseline_ifa.set_index(['location_id','draw'])
                    + anc_ifa_shifts.set_index(['location_id','draw']))
    covered_shift = covered_shift.reset_index().rename(columns={'value':'covered_shift'}).dropna()
    covered_exp = exp_prepped.merge(covered_shift, on=['location_id','draw'])
    covered_exp['ga_start'] = covered_exp.ga_start + covered_exp.covered_shift
    covered_exp['ga_end'] = covered_exp.ga_end + covered_exp.covered_shift
    covered_exp['frac_ptb'] = np.where(covered_exp.ga_start > 37, 0,
                                        np.where(covered_exp.ga_end < 37, 1, 
                                                (37 - covered_exp.ga_start) / (covered_exp.ga_end - covered_exp.ga_start)))
    covered_exp['exposure_ptb'] = covered_exp.frac_ptb * covered_exp.exposure

    # Add exposure_vptb calculation for MMS validation
    covered_exp['frac_vptb'] = np.where(covered_exp.ga_start >= 32, 0,
                                        np.where(covered_exp.ga_end < 32, 1, 
                                                (32 - covered_exp.ga_start) / (covered_exp.ga_end - covered_exp.ga_start)))
    covered_exp['exposure_vptb'] = covered_exp.frac_vptb * covered_exp.exposure

    covered_ptb = covered_exp.groupby(['location_id','draw']).sum()[['exposure_ptb', 'exposure_vptb']]
    covered_ptb.groupby('location_id').describe(percentiles=[0.025,0.975])[['exposure_ptb']]

    mms_pt_dist = lognorm_from_median_lower_upper(0.91, 0.84, 0.99, quantile_ranks=(0.025,0.975))
    mms_pt_rr_draws = pd.DataFrame()
    mms_pt_rr_draws['draw'] = [f'draw_{x}' for x in list(range(0,1000))]
    mms_pt_rr_draws['rr'] = mms_pt_dist.rvs(size=1000, random_state=789)
    # mms_pt_rr_draws.describe(percentiles=[0.025,0.975])

    mms_vpt_dist = lognorm_from_median_lower_upper(0.81, 0.71, 0.93, quantile_ranks=(0.025,0.975))
    mms_vpt_rr_draws = pd.DataFrame()
    mms_vpt_rr_draws['draw'] = [f'draw_{x}' for x in list(range(0,1000))]
    mms_vpt_rr_draws['rr'] = mms_vpt_dist.rvs(size=1000, random_state=101112)
    # mms_vpt_rr_draws.describe(percentiles=[0.025,0.975])

    #display(covered_exp)
    ifa_exp = covered_exp.copy().drop(columns='covered_shift')
    ifa_exp = ifa_exp.groupby([c for c in ifa_exp if c!='sex_id' and 'exposure' not in c]).mean().reset_index()
    ifa_exp['frac_vptb'] = np.where(ifa_exp.ga_start >= 32, 0,
                                            np.where(ifa_exp.ga_end < 32, 1,
                                                    (32 - ifa_exp.ga_start)/(ifa_exp.ga_end - ifa_exp.ga_start)
                                                    ))
    ifa_exp['exposure_vptb'] = ifa_exp.frac_vptb * ifa_exp.exposure
    data_mms_pt = ifa_exp.drop("rr", axis=1).merge(mms_pt_rr_draws, on='draw').drop(columns=['frac_vptb'])
    data_mms_vpt = ifa_exp.drop("rr", axis=1).merge(mms_vpt_rr_draws, on='draw').drop(columns=['frac_ptb'])
    return data_mms_pt, data_mms_vpt
    
