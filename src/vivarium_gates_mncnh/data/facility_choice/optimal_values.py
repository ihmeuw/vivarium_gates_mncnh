"""
Optimal Values Calculation for Birth Facility Model

This script calculates optimal correlation values between different birth-related 
variables (ANC visits, LBWSG categories, and facility delivery) to match target 
probabilities derived from various data sources.
"""

from pathlib import Path
from pprint import pprint

import birth_facility as bf
import numpy as np
import pandas as pd
from vivarium_helpers.prob_distributions.fit import log_loss


class BirthModelParameters:
    """Class to handle loading and preprocessing model parameters"""

    def __init__(
        self, location="pakistan", draw=1, seed=51552426269265631082560804688613666824
    ):
        self.location = location
        self.draw = draw
        self.draw_col = f"draw_{draw}"
        self.seed = seed
        self.cat_df = pd.read_csv("lbwsg_category_data_gbd_2021.csv")
        self.preterm_categories = self.cat_df.loc[
            self.cat_df["ga_end"] <= 37, "lbwsg_category"
        ]

        # Load LBWSG data from artifact
        self._load_lbwsg_data()

        # Set demographic parameters
        self._set_demographic_parameters()

        # Calculate derived parameters
        self._calculate_derived_parameters()

        # Set target probabilities
        self._set_target_probabilities()

    def _load_lbwsg_data(self):
        """Load LBWSG data from artifact file"""

        artifacts_path = Path(
            "/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/artifacts/"
        )
        model_subdirectory = "lbwsg"
        location = "Pakistan"
        artifact_path = artifacts_path / model_subdirectory / f"{location.lower()}.hdf"

        # artifact_path = f'../{self.location.lower()}.hdf'
        exposure_key = "/risk_factor/low_birth_weight_and_short_gestation/exposure"
        rr_key = "/risk_factor/low_birth_weight_and_short_gestation/relative_risk"

        # Filter to valid age groups for LBWSG
        self.lbwsg_exposure = pd.read_hdf(artifact_path, exposure_key).query(
            "age_end < 30/365"
        )
        self.lbwsg_rr = pd.read_hdf(artifact_path, rr_key).query("age_end < 30/365")

        # For each sex, order the LBWSG categories by mean RR in the ENN age group
        self.sex_to_lbwsg_cat_dtype_dict = bf.get_sex_to_lbwsg_category_dtype_dict(
            self.lbwsg_rr
        )
        self.sorted_lbwsg_birth_exposure_cdf_by_sex = (
            bf.get_sorted_lbwsg_birth_exposure_cdf_by_sex(
                self.lbwsg_exposure, self.sex_to_lbwsg_cat_dtype_dict, self.draw
            )
        )

    def _set_demographic_parameters(self):
        """Set demographic parameters from GBD and other sources"""
        # Birth sex distribution
        self.prob_male_birth = 0.5  # TODO: Look up actual value for Pakistan
        self.male_female_birth_probabilities = pd.Series(
            {bf.MALE: self.prob_male_birth, bf.FEMALE: 1 - self.prob_male_birth}
        )

        # ANC visit proportion (covariate ID 7 - Antenatal Care 1 visit Coverage)
        self.anc1_proportion = 0.907563
        self.anc_probabilities = pd.Series(
            {bf.ANC1: self.anc1_proportion, bf.ANC0: 1 - self.anc1_proportion}
        )

        # Ultrasound rates from DHS for India, used as proxy for Pakistan
        PROB_STANDARD_ULTRASOUND_GIVEN_ANC1 = 0.667
        self.prob_ultrasound_given_anc = pd.DataFrame.from_dict(
            {
                bf.ANC0: [1.0, 0.0],
                bf.ANC1: [
                    1 - PROB_STANDARD_ULTRASOUND_GIVEN_ANC1,
                    PROB_STANDARD_ULTRASOUND_GIVEN_ANC1,
                ],
            },
            orient="index",
            columns=pd.Categorical([bf.NO_ULTRASOUND, bf.STANDARD_ULTRASOUND], ordered=True),
        ).rename_axis(index=bf.ANC, columns=bf.ULTRASOUND)

        # In-facility delivery proportion (covariate ID 51)
        self.ifd_proportion_gbd = 0.771766
        self.in_facility_probabilities = pd.Series(
            {bf.IN_FACILITY: self.ifd_proportion_gbd, bf.AT_HOME: 1 - self.ifd_proportion_gbd}
        )

        # Values from Chris's microdata -- how much do we trust these values?
        PROB_BELIEVED_TERM_GIVEN_TERM_STATUS = pd.Series(
            {
                bf.TERM: 0.9076,  # P(believed term | term)
                bf.PRETERM: 0.2612,  # P(believed term | preterm)
            }
        )

        # Values from Chris's microdata, assuming ultrasound GA is 100% accurate:
        PROB_BELIEVED_TERM_GIVEN_TERM_NO_ULTRASOUND = (
            0.9076  # P(believed term | term, no ultrasound)
        )
        PROB_BELIEVED_TERM_GIVEN_PRETERM_NO_ULTRASOUND = (
            0.2612  # P(believed term | preterm, no ultrasound)
        )
        # Values I'm making up:
        PROB_BELIEVED_TERM_GIVEN_TERM_ULTRASOUND = (
            0.995  # P(believed term | term, ultrasound)
        )
        PROB_BELIEVED_TERM_GIVEN_PRETERM_ULTRASOUND = (
            0.015  # P(believed term | preterm, ultrasound)
        )
        prob_believed_term_status_dict = {
            (bf.TERM, bf.NO_ULTRASOUND): [
                1 - PROB_BELIEVED_TERM_GIVEN_TERM_NO_ULTRASOUND,
                PROB_BELIEVED_TERM_GIVEN_TERM_NO_ULTRASOUND,
            ],
            (bf.PRETERM, bf.NO_ULTRASOUND): [
                1 - PROB_BELIEVED_TERM_GIVEN_PRETERM_NO_ULTRASOUND,
                PROB_BELIEVED_TERM_GIVEN_PRETERM_NO_ULTRASOUND,
            ],
            (bf.TERM, bf.ULTRASOUND): [
                1 - PROB_BELIEVED_TERM_GIVEN_TERM_ULTRASOUND,
                PROB_BELIEVED_TERM_GIVEN_TERM_ULTRASOUND,
            ],
            (bf.PRETERM, bf.ULTRASOUND): [
                1 - PROB_BELIEVED_TERM_GIVEN_PRETERM_ULTRASOUND,
                PROB_BELIEVED_TERM_GIVEN_PRETERM_ULTRASOUND,
            ],
        }
        self.prob_believed_term_given_term_status_and_ultrasound = pd.DataFrame(
            prob_believed_term_status_dict.values(),
            index=pd.MultiIndex.from_tuples(
                prob_believed_term_status_dict.keys(), names=[bf.TERM_STATUS, bf.ULTRASOUND]
            ),
            columns=pd.CategoricalIndex(
                [bf.PRETERM, bf.TERM], ordered=True, name=bf.BELIEVED_TERM_STATUS
            ),
        )

    def _calculate_derived_parameters(self):
        """Calculate derived parameters based on loaded data"""
        # Preterm prevalence
        self.preterm_cause_prevalence = 0.167175  # cause ID 381 (Neonatal preterm birth)
        self.preterm_prevalence = bf.get_preterm_prevalence_at_birth(
            self.lbwsg_exposure, self.preterm_categories, self.prob_male_birth
        )[self.draw_col]

        # Values from DHS - facility delivery conditional on ANC status
        prob_in_facility_given_anc_dhs = pd.Series(
            {
                bf.ANC0: 0.536443,  # constraint 1
                bf.ANC1: 0.766985,  # constraint 2
            }
        )

        # Rescale P(in-facility|ANC) to be consistent with P(in-facility) from GBD
        self.prob_in_facility_given_anc = bf.get_consistent_conditional_probabilities(
            prob_in_facility_given_anc_dhs,
            self.anc_probabilities,
            self.ifd_proportion_gbd,
        )

        # Values from WomenFirst study - preterm births by delivery location
        prob_preterm_given_facility_raw = pd.Series(
            {
                bf.AT_HOME: 0.153846,  # constraint 3
                bf.IN_FACILITY: 0.203837,  # constraint 4
            }
        )

        # Rescale P(preterm|facility) to be consistent with P(preterm) from GBD
        self.prob_preterm_given_facility = bf.get_consistent_conditional_probabilities(
            prob_preterm_given_facility_raw,
            self.in_facility_probabilities,
            self.preterm_prevalence,
        )

    def _set_target_probabilities(self):
        """Set target probabilities to match in optimization"""
        self.targets = pd.concat(
            [self.prob_in_facility_given_anc, self.prob_preterm_given_facility],
            keys=[bf.IN_FACILITY, bf.PRETERM],
            names=["probability_of", "given"],
        ).rename("target_probabilities")


class OptimalCorrelationFinder:
    """Class to find optimal correlation values between birth model variables"""

    def __init__(self, params, pop_size=10_000, num_initial_conditions=10):
        self.params = params
        self.pop_size = pop_size
        self.num_initial_conditions = num_initial_conditions
        self.pop_index = pd.Index(range(pop_size), name="simulant_id")

        # Specify which variables we'll induce correlation between
        self.correlated_variables = ["anc", "lbwsg_cat", "facility"]
        # Specify which variables we'll sample independently
        self.independent_variables = ["sex", "ultrasound", "ga_error"]
        # Exogenous variables are all propensities
        self.exogenous_variables = self.correlated_variables + self.independent_variables

    def find_optimal_correlations(
        self, anc_lbw_range=(0.8, -0.8, -0.1), lbwsg_facility_range=(0.8, -0.8, -0.1)
    ):
        """
        Find optimal correlations between model variables

        Args:
            anc_lbw_range: Tuple of (start, stop, step) for ANC-LBWSG correlation search
            lbwsg_facility_range: Tuple of (start, stop, step) for LBWSG-facility correlation search

        Returns:
            DataFrame with results for all parameter combinations
        """
        results_data = []

        for anc_lbw_corr in np.arange(*anc_lbw_range):
            for lbwsg_facility_corr in np.arange(*lbwsg_facility_range):
                print(
                    f"Testing ANC-LBWSG correlation: {anc_lbw_corr:.1f}, "
                    f"LBWSG-Facility correlation: {lbwsg_facility_corr:.1f}"
                )

                # Set fixed correlation values
                fixed_x_components = {0: anc_lbw_corr, 2: lbwsg_facility_corr}

                # Find solutions
                solutions, fixed_values, initial_points, error_log = bf.find_solutions(
                    self.num_initial_conditions,
                    self.pop_index,
                    self.exogenous_variables,
                    self.independent_variables,
                    self.params.male_female_birth_probabilities,
                    self.params.anc1_proportion,
                    self.params.prob_ultrasound_given_anc,
                    self.params.sorted_lbwsg_birth_exposure_cdf_by_sex,
                    self.params.preterm_categories,
                    self.params.prob_believed_term_given_term_status_and_ultrasound,
                    self.params.targets,
                    self.params.seed,
                    loss_func=log_loss,
                    fixed_x_components=fixed_x_components,
                    fixed_x_components_on_iteration=(),
                )

                # Sort results by error value
                sorted_indices = sorted(
                    range(self.num_initial_conditions), key=lambda i: solutions[i].fun
                )

                # Get best result
                best_result_idx = sorted_indices[0]
                best_result = solutions[best_result_idx]
                best_xf = bf.fill_x_values(best_result.x, fixed_values[best_result_idx])

                # Store results
                best_dict = {
                    "anc_lbw_corr": anc_lbw_corr,
                    "lbwsg_facility_corr": lbwsg_facility_corr,
                    "anc_facility_corr": best_xf[1],  # Extract from best solution
                    "prob_home_given_believed_preterm": best_xf[
                        3
                    ],  # Extract from best solution
                    "prob_home_given_believed_term": best_xf[4],  # Extract from best solution
                    "error": best_result.fun,
                    # 'best_result': best_result,
                    # 'best_x_values': best_xf
                }
                pprint(best_dict)
                print()
                results_data.append(best_dict)

        # Create results DataFrame
        return pd.DataFrame(results_data)


def main():
    """Main function to run the optimization"""
    # Initialize parameters
    params = BirthModelParameters(location="pakistan", draw=1)

    # Create correlation finder
    finder = OptimalCorrelationFinder(params, pop_size=20_000, num_initial_conditions=25)

    # Run search with smaller ranges for testing
    results_df = finder.find_optimal_correlations(
        anc_lbw_range=(0.7, -0.3, -0.1),  # Reduced range for testing
        lbwsg_facility_range=(0.7, -0.3, -0.1),  # Reduced range for testing
    )

    # Sort by error value to find best overall correlation combination
    results_df_sorted = results_df.sort_values("error")

    # Display best results
    print("\nBest correlation combinations (lowest error first):")
    print(
        results_df_sorted[
            ["anc_lbw_corr", "lbwsg_facility_corr", "anc_facility_corr", "error"]
        ].head(5)
    )

    # Save results to CSV
    results_df.to_csv(f"correlation_search_results_{params.location}_all.csv", index=False)

    # Return best correlation values
    best_row = results_df_sorted.iloc[0]
    print(f"\nBest correlation values found:")
    print(f"ANC-LBWSG correlation: {best_row['anc_lbw_corr']:.3f}")
    print(f"LBWSG-Facility correlation: {best_row['lbwsg_facility_corr']:.3f}")
    print(f"ANC-Facility correlation: {best_row['anc_facility_corr']:.3f}")
    print(f"Error value: {best_row['error']:.6f}")

    return results_df


if __name__ == "__main__":
    results_df = main()
