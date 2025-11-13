from __future__ import annotations

import importlib
import inspect
from collections import namedtuple
from collections.abc import Iterable, Mapping, Sequence

# from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Type, TypeAlias

import numpy as np
import pandas as pd
import scipy
import vivarium
from correlation import extract_upper_3, generate_correlation_matrix
from loss_functions import log_loss
from numpy.random import SeedSequence
from pandas.api.types import CategoricalDtype
from statsmodels.distributions.copula.api import GaussianCopula
from vivarium_helpers.lbwsg.lbwsg import (
    get_category_data,
    validate_exposure_matches_category_data,
)
from vivarium_helpers.prob_distributions.fit import l2_loss, log_loss
from vivarium_helpers.prob_distributions.sampling import (
    sample_categorical_from_propensity,
    sample_series_from_propensity,
)

# We store data received from private correspondence with the GF
# in a separate file, which makes it easier to not include it in
# our public simulation repo.
# Note that as of right now, the used version of this model (BirthFacilityModelWithUltrasoundAndSimpleGAError)
# does not make use of this data.
# It does, however, make use of the facility_choice_data.xlsx
# data file, which also contains such private-correspondence data.
if importlib.util.find_spec("data_values") is not None:
    HAS_PRIVATE_DATA = True
    from data_values import JOINT_PROB_LMP_ULTRASOUND_TERM_STATUS
else:
    HAS_PRIVATE_DATA = False


# TODO: List valid types that can be passed to numpy.random.default_rng:
# https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng
Seed: TypeAlias = Any

### Metadata

# Column names for population
SEX = "sex"  # Male/Female
ANC = "anc"  # anc1/anc0
ULTRASOUND = "ultrasound"  # none, standard, AI
TERM_STATUS = "term_status"  # term/preterm
BELIEVED_TERM_STATUS = "believed_term_status"  # believed_term/believed_preterm
LBWSG_CAT = "lbwsg_category"  # 58 categories
GA = "gestational_age"  # continuous
GA_ERROR = "ga_error"  # continuous
GA_ESTIMATE = "estimated_gestational_age"  # continuous
FACILITY = "facility"  # 2 or 3 categories

# labels/values in columns
MALE = "Male"
FEMALE = "Female"
ANC0 = "anc0"
ANC1 = "anc1"
NO_ULTRASOUND = "no_ultrasound"
STANDARD_ULTRASOUND = "standard_ultrasound"
AT_HOME = "at_home"
IN_FACILITY = "in_facility"
BEMONC = "BEmONC"
CEMONC = "CEmONC"
PRETERM = "preterm"
TERM = "term"
BELIEVED_PRETERM = "believed_preterm"
BELIEVED_TERM = "believed_term"


# Create CategoricalDtypes for all categorical variables, ordered from
# "worst" to "best" so that higher propensities correspond to better
# outcomes.
# For sex, higher propensities arbitrarily correspond to females
SEX_DTYPE = CategoricalDtype([MALE, FEMALE], ordered=True)
ANC_DTYPE = CategoricalDtype([ANC0, ANC1], ordered=True)
ULTRASOUND_DTYPE = CategoricalDtype([NO_ULTRASOUND, STANDARD_ULTRASOUND], ordered=True)
TERM_STATUS_DTYPE = CategoricalDtype([PRETERM, TERM], ordered=True)
BELIEVED_TERM_STATUS_DTYPE = CategoricalDtype([BELIEVED_PRETERM, BELIEVED_TERM], ordered=True)
BINARY_FACILITY_DTYPE = CategoricalDtype([AT_HOME, IN_FACILITY], ordered=True)
TERNARY_FACILITY_DTYPE = CategoricalDtype([AT_HOME, BEMONC, CEMONC], ordered=True)

# Create Index objects using the above dtypes
SEX_INDEX = pd.CategoricalIndex(SEX_DTYPE.categories, dtype=SEX_DTYPE, name=SEX)
ANC_INDEX = pd.CategoricalIndex(ANC_DTYPE.categories, dtype=ANC_DTYPE, name=ANC)
ULTRASOUND_INDEX = pd.CategoricalIndex(
    ULTRASOUND_DTYPE.categories, dtype=ULTRASOUND_DTYPE, name=ULTRASOUND
)
TERM_STATUS_INDEX = pd.CategoricalIndex(
    TERM_STATUS_DTYPE.categories, dtype=TERM_STATUS_DTYPE, name=TERM_STATUS
)
BELIEVED_TERM_STATUS_INDEX = pd.CategoricalIndex(
    BELIEVED_TERM_STATUS_DTYPE.categories,
    dtype=BELIEVED_TERM_STATUS_DTYPE,
    name=BELIEVED_TERM_STATUS,
)
BINARY_FACILITY_INDEX = pd.CategoricalIndex(
    BINARY_FACILITY_DTYPE.categories, dtype=BINARY_FACILITY_DTYPE, name=FACILITY
)
TERNARY_FACILITY_INDEX = pd.CategoricalIndex(
    TERNARY_FACILITY_DTYPE.categories, dtype=TERNARY_FACILITY_DTYPE, name=FACILITY
)

# Name for "mean draw" column
MEAN_DRAW = "mean_draw"

# Column names in Artifacts and in GBD
AGE_END_COL = "age_end"
CHILD_AGE_END_COL = "child_age_end"
AGE_GROUP_ID_COL = "age_group_id"
SEX_COL = "sex"
CHILD_SEX_COL = "sex_of_child"
SEX_ID_COL = "sex_id"

# Values used in Artifacts
BIRTH_AGE_END = 0
ENN_AGE_END = np.round(7 / 365, 8)
LNN_AGE_END = np.round(28 / 365, 8)
# (Also MALE and FEMALE as defined above)

# Values used in GBD
BIRTH_ID = 164
ENN_ID = 2
LNN_ID = 3
MALE_ID = 1
FEMALE_ID = 2

# Flag for re-using last used random seed
LAST_USED = "last_used"

# Gestational age (in weeks) below which babies are considered preterm
PRETERM_CUTOFF = 37

if HAS_PRIVATE_DATA:
    # Use Chris's microdata to compute conditional probabilities of
    # believing your baby is term using last menstrual period, given
    # the term status identified by ultrasound.
    # Below, we'll use the term status by ultrasound as the ground
    # truth (i.e., we'll assume ultrasound is 100% accurate at
    # identifying preterm status) so that we can assume these values
    # are the True negative rate and false negative rate for
    # identifying preterm status using last menstrual period.
    PROB_TERM_LMP_GIVEN_TERM_STATUS_ULTRASOUND = (
        JOINT_PROB_LMP_ULTRASOUND_TERM_STATUS.loc["term_lmp"]
        / JOINT_PROB_LMP_ULTRASOUND_TERM_STATUS.sum()
    ).rename("prob_term_lmp")
# Values I'm making up, specifying true negative rate and false
# negative rate for identifying preterm status via ultrasound:
# P(believed term | term, ultrasound) = true negative rate
PROB_BELIEVED_TERM_GIVEN_TERM_ULTRASOUND = 0.99
# P(believed term | preterm, ultrasound) = false negative rate
PROB_BELIEVED_TERM_GIVEN_PRETERM_ULTRASOUND = 0.02

### Data about error in GA estimation from literature

DAYS_PER_WEEK = 7  # for converting gestational age data

# Standard deviations in gestational age estimate based on values in
# literature
# TODO: Revise these after a more careful reading, and add more
# categories of ultrasound (e.g., ultrasound in 1st trimester,
# ultrasound in later preganancy, ultrasound in both)
# HACK: These are arbitrary large values to give AI ultrasound an extreme advantage,
# to find the plausible upper bound of impact.
ST_DEV_OF_GA_ERROR_GIVEN_ULTRASOUND = (
    pd.Series(
        {NO_ULTRASOUND: 70, STANDARD_ULTRASOUND: 30},  # Units are days
        index=pd.CategoricalIndex(
            ULTRASOUND_DTYPE.categories, dtype=ULTRASOUND_DTYPE, name=ULTRASOUND
        ),
        name="std_of_ga_error",
    )
    / DAYS_PER_WEEK
)  # Convert from days to weeks to match GA from GBD

# Data sources in spreadsheet
PROB_ULTRASOUND_SOURCE = "prob_ultrasound"
PROB_PRETERM_GIVEN_FACILITY_SOURCE = "prob_preterm_given_facility"
PROB_FACILITY_SOURCE = "prob_facility"
DEFAULT_DATA_SOURCES = {
    "Pakistan": {
        PROB_ULTRASOUND_SOURCE: "DHS",
        PROB_PRETERM_GIVEN_FACILITY_SOURCE: "Pooled studies",
        PROB_FACILITY_SOURCE: "WomenFirst",
    },
    "Ethiopia": {
        PROB_ULTRASOUND_SOURCE: "Yetwale et al.",
        PROB_PRETERM_GIVEN_FACILITY_SOURCE: "HU-NeoSurvival",
        PROB_FACILITY_SOURCE: "DHS",
    },
    "Nigeria": {
        PROB_ULTRASOUND_SOURCE: "Ugwu et al.",
        PROB_PRETERM_GIVEN_FACILITY_SOURCE: "Fajolu et al.",
        PROB_FACILITY_SOURCE: "DHS",
    },
}

TargetProbabilities = namedtuple(
    "TargetProbabilities", ["facility_given_anc", "term_status_given_facility"]
)

GBDMetadata = namedtuple(
    "GBDMetadata",
    [
        "age_col",
        "birth",
        "enn",
        "lnn",
        "sex_col",
        "male",
        "female",
    ],
)


def get_gbd_medatada(df):
    """Get appropriate column names and values depending on whether
    the DataFrame df is from an Artifact or from get_draws.
    """
    is_artifact_data = AGE_END_COL in df.index.names and SEX_COL in df.index.names
    # NOTE: Only check sex column because birth exposure data doesn't
    # contain age columns
    is_child_artifact_data = CHILD_SEX_COL in df.index.names
    is_get_draws_data = AGE_GROUP_ID_COL in df and SEX_ID_COL in df
    if is_artifact_data:
        age_col = AGE_END_COL
        birth, enn, lnn = BIRTH_AGE_END, ENN_AGE_END, LNN_AGE_END
        sex_col = SEX_COL
        male, female = MALE, FEMALE
    elif is_child_artifact_data:
        age_col = CHILD_AGE_END_COL
        birth, enn, lnn = BIRTH_AGE_END, ENN_AGE_END, LNN_AGE_END
        sex_col = CHILD_SEX_COL
        male, female = MALE, FEMALE
    elif is_get_draws_data:
        age_col = AGE_GROUP_ID_COL
        birth, enn, lnn = BIRTH_ID, ENN_ID, LNN_ID
        sex_col = SEX_ID_COL
        male, female = MALE_ID, FEMALE_ID
    else:
        raise ValueError("Unknown GBD data format for DataFrame `df`")
    metadata = GBDMetadata(age_col, birth, enn, lnn, sex_col, male, female)
    return metadata


### Functions for getting consistent input parameters


def get_consistent_conditional_probabilities(
    conditional_probabilities,
    stratum_probabilities,
    total_probability,
    min_iter=1,
    max_iter=100,
    **kwargs,  # kwargs for numpy.allclose
):
    """Rescales a DataFrame or Series of conditional probabilities to
    make them consistent with a given set of probabilities for each
    stratum and a given set of total probabilities across strata.

    If `conditional_probabilities` is a DataFrame, entries should be of
    the form .iloc[i, j] = P(B_j|A_i), and if
    `conditional_probabilities` is a Series, its entries should be
    .iloc[i] = P(B_j|A_i), where j is fixed. `stratum_probabilities`
    should be a Series with .iloc[i] = P(A_i). `total_probability`
    should be a Series with .iloc[j] = P(B_j), or a scalar equal to
    P(B_j) if `conditional_probabilities` is a Series.

    For each j, scales the array of conditional probabilities P(B_j|A_i)
    by an approximately constant factor to new probabilities P'(B_j|A_i)
    so that their weighted average over the strata A_i, sum_i
    P'(B_j|A_i) * P(A_i), is equal to the specified total probability
    P(B_j). When `conditional_probabilities` is a DataFrame, also
    guarantees that sum_j P(B_j|A_i) = 1 for each i; if
    `conditional_probabilities` is a Series, there is no such guarantee
    since only one j is considered.

    Note: I think this function might implement a version of the
    iterative proportional fitting procedure (IPF), also known as
    biproportional fitting, the RAS algorithm, raking, or matrix
    scaling:
    https://en.wikipedia.org/wiki/Iterative_proportional_fitting

    But I have not looked into IPF closely enough to determine whether
    it is exactly the same as the iterative algorithm I've implemented
    here.
    """
    done = False
    iteration = 1
    while not done:
        if isinstance(conditional_probabilities, pd.DataFrame):
            # Rescale to ensure conditional probabilities sum to 1
            conditional_probabilities = conditional_probabilities.divide(
                conditional_probabilities.sum(axis=1), axis=0
            )
        # Compute the value of P(B_j) that is implied by the given
        # values of P(A_i) and the current values of P(B_j|A_i) (the sum
        # will be a Series of P(B_j)'s or a scalar for a single B_j).
        total_probability_consistent = conditional_probabilities.multiply(
            stratum_probabilities, axis=0
        ).sum()
        # Compute the correction factor to rescale the conditional
        # probabilities by (here we're either dividing two Series or two
        # scalars)
        correction_factor = total_probability / total_probability_consistent
        # Rescale (here we're multiplying two Series or two scalars)
        new_conditional_probabilities = correction_factor * conditional_probabilities
        iteration += 1
        done = (
            iteration > min_iter
            and np.allclose(
                new_conditional_probabilities, conditional_probabilities, **kwargs
            )
        ) or iteration > max_iter
        # Update conditional probabilities with new values
        conditional_probabilities = new_conditional_probabilities
    return conditional_probabilities


# NOTE: This function is superseded by the one below
def get_consistent_residual_conditional_probability(
    known_conditional_probabilities,
    corresponding_stratum_probabilities,
    residual_stratum_probability,
    total_probability,
):
    residual_conditional_probability = (
        total_probability
        - (known_conditional_probabilities * corresponding_stratum_probabilities).sum()
    ) / residual_stratum_probability
    return residual_conditional_probability


# NOTE: This replaces the above function, with a simpler interface and
# the ability to handle DataFrames as well as Series
def fill_missing_conditional_probabilities(
    conditional_probabilities: pd.Series | pd.DataFrame,
    stratum_probabilities: pd.Series,
    total_probability: float | pd.Series,
):
    """
    Fill in missing conditional probabilities using the known conditional
    probabilities, the stratum probabilities, and the total probability.
    The conditional probability is assumed to be the same for each
    missing stratum.
    """
    # Calculate how much probability is left after we sum over all the
    # known strata: this is P((event of interest) and (missing strata))
    residual_probability = (
        total_probability
        # This sums over non-null values, i.e., non-missing strata
        - (conditional_probabilities.multiply(stratum_probabilities, axis=0)).sum()
    )
    missing = conditional_probabilities.isna()
    if isinstance(conditional_probabilities, pd.DataFrame):
        assert missing.sum(axis=1).isin([0, len(missing.columns)]).all(), (
            "Conditional probabilities must be either all missing"
            " or all present for each stratum."
        )
        # Change missing to a Series indexing the missing rows
        missing = missing.any(axis=1)
        # NOTE: For some reason, we must convert this Series to an array
        # for assignment to work (is this a feature or a bug?); reindex
        # to ensure columns are aligned, since we're losing the index
        # information
        residual_probability = residual_probability.reindex(
            conditional_probabilities.columns
        ).array
    # Make a copy instead of overwriting the input
    filled_conditional_probabilities = conditional_probabilities.copy()
    # Divide the residual probability by the sum of the probabilities of
    # the missing strata to get the conditional probability for the
    # missing strata
    filled_conditional_probabilities.loc[missing] = (
        residual_probability / stratum_probabilities.loc[missing].sum()
    )
    return filled_conditional_probabilities


### Functions for dealing with LBWSG data


def get_preterm_prevalence_at_birth(
    lbwsg_birth_exposure, preterm_categories, prob_male_birth=None
):
    """Return a DataFrame or Series with the prevalence of preterm
    birth for each draw. If `prob_male_birth` is None (default),
    returns a DataFrame with one row for each sex (Male and Female),
    and whose columns are draws from GBD.
    If `prob_male_birth` is a float between 0 and 1, returns a Series
    indexed by draw, whose values are the weighted average of preterm
    prevalence for males and females combined.
    """
    meta = get_gbd_medatada(lbwsg_birth_exposure)
    preterm_prev_by_sex = (
        lbwsg_birth_exposure.query("parameter.isin(@preterm_categories)")
        .groupby(meta.sex_col)
        .sum()
    )
    if prob_male_birth is not None:
        preterm_prev = (
            prob_male_birth * preterm_prev_by_sex.loc[meta.male]
            + (1 - prob_male_birth) * preterm_prev_by_sex.loc[meta.female]
        )
    else:
        preterm_prev = preterm_prev_by_sex
    return preterm_prev


def get_lbwsg_categories_sorted_by_risk(
    lbwsg_rr, preterm_categories=None, sexes_distinct=True
):
    """Returns a DataFrame containing LBWSG categories (in the
    'parameter' column as per GBD and Artifact convention), ordered for
    each sex by decreasing mean relative risk in the early neonatal age
    group. If preterm_categories is also passed, then the categories are
    ordered *first* by preterm status in decreasing order (True then
    False), then by decreasing relative risk among categories with the
    same preterm status.
    """
    meta = get_gbd_medatada(lbwsg_rr)
    enn_rr = lbwsg_rr.query(f"{meta.age_col} == @meta.enn")
    draw_cols = lbwsg_rr.filter(regex=r"^draw_\d{1,3}$").columns
    # Make sure we're not using capped RRs
    assert (
        enn_rr.query("parameter=='cat10'")[draw_cols].nunique(axis=1) == len(draw_cols)
    ).all(), "Not all draws of LBWSG cat10 relative risks are unique"
    lbwsg_category_risks = (
        enn_rr[draw_cols]
        .mean(axis=1)
        .rename("rr_mean")
        .reset_index()
        # Sort by descending RR because low RRs are good
        # .sort_values([meta.sex_col, 'rr_mean'], ascending=False)
    )
    # If we don't want to distinguish between the sexes, take the mean
    # over sex for each category, but keep sex in the index
    if not sexes_distinct:
        lbwsg_category_risks = lbwsg_category_risks.groupby("parameter").transform("mean")
    # If preterm_categories were passed, sort *first* by preterm status,
    # then by mean relative risk
    if preterm_categories is not None:
        lbwsg_category_risks["is_preterm"] = lbwsg_category_risks["parameter"].isin(
            preterm_categories
        )
        # Sort descending by preterm status because True is bad, False
        # is good
        sort_cols = [meta.sex_col, "is_preterm", "rr_mean"]
    else:
        sort_cols = [meta.sex_col, "rr_mean"]

    # Sort by descending RR because low RRs are good
    lbwsg_category_risks.sort_values(sort_cols, ascending=False, inplace=True)
    return lbwsg_category_risks


def get_sorted_lbwsg_birth_exposure_cdf_by_sex(
    lbwsg_birth_exposure: pd.DataFrame,
    lbwsg_categories_sorted_by_risk: dict[str, list[str]],
    draw_col,
):
    """Get the cumulative distribution function for LBWSG exposure
    for the specified sex and draw. If using Artifact data, sex
    should be either 'Male' or 'Female'; if using data from get_draws,
    sex should be GBD's corresponding sex_id (1 or 2).
    """

    def set_columns_dtype(df, dtype):
        return df.set_axis(df.columns.astype(dtype), axis=1)

    # Get format (GBD or Artifact) of exposure DataFrame - the risk
    # dataframe is assumed to have the same format (namely, the same
    # name for the "sex" column: 'sex_id', 'sex', or 'sex_of_child').
    # Note that this is NOT true for the `birth_exposure` artifact
    # (which uses 'sex' for the RRs and exposure, but 'sex_of_child' for
    # birth exposure), but it seems to be true for other versions of the
    # artifact, and it will be true for raw GBD data.
    meta = get_gbd_medatada(lbwsg_birth_exposure)

    # NOTE: Format (GBD or Artifact) of exposure data is assumed to be
    # the same as the format of the RR data in
    # lbwsg_categories_sorted_by_risk DataFrame
    sex_to_lbwsg_category_dtype_dict = {
        sex: CategoricalDtype(lbwsg_categories_sorted_by_risk[sex], ordered=True)
        for sex in [meta.male, meta.female]
    }
    sex_to_exposure_cdf_dict = {
        sex: (
            lbwsg_birth_exposure[draw_col]
            .unstack("parameter")
            .query(f"{meta.sex_col}==@sex")
            .pipe(set_columns_dtype, cat_dtype)
            .sort_index(axis=1)
            .cumsum(axis=1)
        )
        for sex, cat_dtype in sex_to_lbwsg_category_dtype_dict.items()
    }
    return sex_to_exposure_cdf_dict


### Helper functions


def flatten_index(df: pd.DataFrame) -> pd.DataFrame:
    """Replaces a MultiIndex in df with a flattened Index."""
    return df.set_axis(df.index.to_flat_index())


def stack_conditional_probability_table(table: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Converts a DataFrame of conditional probabilities P(Y | X), where
    X is indexed by rows and Y is indexed by columns, into a Series
    where each row corresponds to one of the conditional probabilities
    in the table. The rows will be labeled as
    '{y_value}_given_{x_value}'. If the DataFram's columns are a
    MultiIndex, only the innermost level will be stacked, and the result
    will be a DataFrame instead of a Series.
    """
    # If conditioning on multiple variables (as levels in a MultiIndex),
    # join the different conditions with "and"
    if isinstance(table.index, pd.MultiIndex):
        table = table.pipe(flatten_index).rename(index=lambda t: "_and_".join(t))
    stacked = (
        table.stack()
        .pipe(flatten_index)
        .rename(index=lambda t: "_given_".join(reversed(t)))
        .rename_axis("probability_of")
    )
    return stacked


#### Class for storing birth location model data


class CausalModelData:
    pass


class BirthFacilityChoiceData(CausalModelData):
    def __init__(
        self,
        location,
        birth_facility_data_path="facility_choice_data.xlsx",
        data_sources_overrides=None,
        artifacts_directory="/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/artifacts/",
        model_subdirectory="model19.0.1",
        draw=MEAN_DRAW,
    ):
        self.location = location.title()
        self.sources = DEFAULT_DATA_SOURCES[self.location]
        self.sources.update(data_sources_overrides or {})
        self.draw = draw
        self.draw_col = draw if draw == MEAN_DRAW else f"draw_{draw}"
        self.birth_facility_data = pd.read_excel(
            birth_facility_data_path, self.location
        ).set_index(["Parameter", "Source"])
        self.artifact_path = (
            Path(artifacts_directory) / model_subdirectory / f"{location.lower()}.hdf"
        )
        self.cat_df = get_category_data()

        # Load LBWSG data from artifact
        self._load_lbwsg_data()
        # Record some data from the LBWSG category dataframe
        self._set_lbwsg_category_data()
        # Set demographic parameters
        self._set_demographic_parameters()
        # Calculate derived parameters
        self._calculate_derived_parameters()
        # Store all inputs in a single Series
        self._set_input_probabilities()
        # Set target probabilities
        self._set_target_probabilities()

    def _set_lbwsg_category_data(self):
        """Record data from LBWSG category data DataFrame."""
        self.lbwsg_category_dtype = CategoricalDtype(
            # Categories are ordered lexicographically in the dataframe
            self.cat_df["lbwsg_category"],
            ordered=True,
        )
        self.preterm_categories = self.cat_df.loc[
            self.cat_df["ga_end"] <= PRETERM_CUTOFF, "lbwsg_category"
        ]

    def _load_lbwsg_data(self):
        """Load LBWSG data from artifact file"""

        birth_exposure_key = "risk_factor.low_birth_weight_and_short_gestation.birth_exposure"
        exposure_key = "risk_factor.low_birth_weight_and_short_gestation.exposure"
        rr_key = "risk_factor.low_birth_weight_and_short_gestation.relative_risk"
        sex_specific_ordered_categories_key = (
            "risk_factor.low_birth_weight_and_short_gestation.sex_specific_ordered_categories"
        )

        art = vivarium.Artifact(self.artifact_path)
        self.lbwsg_rr = art.load(rr_key)
        if birth_exposure_key in art.keys:
            self.lbwsg_birth_exposure = art.load(birth_exposure_key)
        else:
            lbwsg_exposure = art.load(exposure_key)
            meta = get_gbd_medatada(lbwsg_exposure)
            self.lbwsg_birth_exposure = lbwsg_exposure.query(
                # Make a copy to avoid SettingWithCopyWarning
                f"{meta.age_col} == {meta.birth}"
            ).copy()

        sex_specific_ordered_categories = art.load(sex_specific_ordered_categories_key)

        # Add a "mean draw" column
        draw_cols = self.lbwsg_birth_exposure.filter(regex=r"^draw_\d{1,3}$").columns
        self.lbwsg_birth_exposure[MEAN_DRAW] = self.lbwsg_birth_exposure[draw_cols].mean(
            axis=1
        )

        validate_exposure_matches_category_data(self.lbwsg_birth_exposure, self.cat_df)

        # For each sex, order the LBWSG categories first by preterm
        # status, then by mean RR in the ENN age group
        self.lbwsg_birth_exposure_cdf_sorted_by_preterm_and_rr = (
            get_sorted_lbwsg_birth_exposure_cdf_by_sex(
                self.lbwsg_birth_exposure, sex_specific_ordered_categories, self.draw_col
            )
        )

    def _set_demographic_parameters(self):
        """Set demographic parameters from GBD and other sources"""
        # Birth sex distribution
        art = vivarium.Artifact(self.artifact_path)
        prob_male_birth = art.load("population.infant_male_percentage")
        self.male_female_birth_probabilities = pd.Series(
            {MALE: prob_male_birth, FEMALE: 1 - prob_male_birth},
            index=SEX_INDEX,
            name="birth_probability",
        )

        anc1_proportion = art.load(
            "covariate.antenatal_care_1_visit_coverage_proportion.estimate"
        ).mean(axis=1)
        assert anc1_proportion.shape == (1,)
        anc1_proportion = anc1_proportion.iloc[0]
        # Note: This must be ordered with ANC0 first and ANC1 second so
        # that larger propensities correspond to the better outcome of
        # ANC1 when this probabilility distribution is used to
        # initialize a population
        self.anc_probabilities = pd.Series(
            {ANC0: 1 - anc1_proportion, ANC1: anc1_proportion},
            index=ANC_INDEX,
            name="proportion",
        )

        # In-facility delivery proportion (covariate ID 51)
        ifd_proportion_gbd = art.load("covariate.in_facility_delivery_proportion.estimate")
        self.in_facility_probabilities = pd.Series(
            {IN_FACILITY: ifd_proportion_gbd, AT_HOME: 1 - ifd_proportion_gbd},
            index=BINARY_FACILITY_INDEX,
            name="proportion_of_births",
        )

        # Ultrasound rates from different source for each country
        PROB_STANDARD_ULTRASOUND_GIVEN_ANC1 = self.birth_facility_data.at[
            ("P(ultrasound|ANC1)", self.sources[PROB_ULTRASOUND_SOURCE]), "Value"
        ]
        self.prob_ultrasound_given_anc = pd.DataFrame.from_dict(
            {
                ANC0: [1.0, 0.0],
                ANC1: [
                    1 - PROB_STANDARD_ULTRASOUND_GIVEN_ANC1,
                    PROB_STANDARD_ULTRASOUND_GIVEN_ANC1,
                ],
            },
            orient="index",
            columns=ULTRASOUND_INDEX,
        ).reindex(ANC_INDEX)

        # Raw trichotomous facility type probabilities from DHS (or
        # WomenFirst study, since the values seem more reasonable)
        # (We'll rescale these later so that IFD proportion matches GBD)
        self.prob_facility_raw = pd.Series(
            [
                self.birth_facility_data.at[
                    ("P(home)", self.sources[PROB_FACILITY_SOURCE]), "Value"
                ],
                self.birth_facility_data.at[
                    ("P(clinic)", self.sources[PROB_FACILITY_SOURCE]), "Value"
                ],
                self.birth_facility_data.at[
                    ("P(hospital)", self.sources[PROB_FACILITY_SOURCE]), "Value"
                ],
            ],
            index=TERNARY_FACILITY_INDEX,
            name="proportion_of_births",
        )

        if HAS_PRIVATE_DATA:
            # Probability of believed term status given true term status and
            # type of ultrasound
            prob_believed_term_status_dict = {
                # P(believed term | term, no ultrasound) = true negative
                # rate for no ultrasound
                (TERM, NO_ULTRASOUND): [
                    1 - PROB_TERM_LMP_GIVEN_TERM_STATUS_ULTRASOUND["term_ultrasound"],
                    PROB_TERM_LMP_GIVEN_TERM_STATUS_ULTRASOUND["term_ultrasound"],
                ],
                # P(believed term | preterm, no ultrasound) = false negative
                # rate for no ultrasound
                (PRETERM, NO_ULTRASOUND): [
                    1 - PROB_TERM_LMP_GIVEN_TERM_STATUS_ULTRASOUND["preterm_ultrasound"],
                    PROB_TERM_LMP_GIVEN_TERM_STATUS_ULTRASOUND["preterm_ultrasound"],
                ],
                # P(believed term | term, ultrasound) = true negative rate
                # for ultrasound
                (TERM, STANDARD_ULTRASOUND): [
                    1 - PROB_BELIEVED_TERM_GIVEN_TERM_ULTRASOUND,
                    PROB_BELIEVED_TERM_GIVEN_TERM_ULTRASOUND,
                ],
                # P(believed term | preterm, ultrasound) = false negative
                # rate for ultrasound
                (PRETERM, STANDARD_ULTRASOUND): [
                    1 - PROB_BELIEVED_TERM_GIVEN_PRETERM_ULTRASOUND,
                    PROB_BELIEVED_TERM_GIVEN_PRETERM_ULTRASOUND,
                ],
            }
            self.prob_believed_term_status_given_term_status_and_ultrasound = (
                pd.DataFrame.from_dict(
                    prob_believed_term_status_dict,
                    orient="index",
                    columns=BELIEVED_TERM_STATUS_INDEX,
                ).reindex(pd.MultiIndex.from_product([TERM_STATUS_INDEX, ULTRASOUND_INDEX]))
            )

        self.st_dev_of_ga_error_given_ultrasound = ST_DEV_OF_GA_ERROR_GIVEN_ULTRASOUND

    def _calculate_derived_parameters(self):
        """Calculate derived parameters based on loaded data"""
        # Calculate preterm prevalence from LBWSG data
        self.preterm_prevalence = get_preterm_prevalence_at_birth(
            self.lbwsg_birth_exposure,
            self.preterm_categories,
            self.male_female_birth_probabilities[MALE],
        )[self.draw_col]

        self.term_status_probabilities = pd.Series(
            {PRETERM: self.preterm_prevalence, TERM: 1 - self.preterm_prevalence},
            index=TERM_STATUS_INDEX,
            name="birth_prevalence",
        )

        # Rescale DHS facility probabilities to match GBD IFD proportion
        prob_in_facility_raw = self.prob_facility_raw.loc[[BEMONC, CEMONC]].sum()
        self.prob_facility = self.prob_facility_raw.copy()
        self.prob_facility.at[AT_HOME] = self.in_facility_probabilities.at[AT_HOME]
        self.prob_facility.loc[[BEMONC, CEMONC]] *= (
            self.in_facility_probabilities.at[IN_FACILITY] / prob_in_facility_raw
        )
        assert np.isclose(
            self.prob_facility.sum(), 1.0
        ), "Error rescaling DHS facility probabilities"

        # Values from DHS - facility delivery conditional on ANC status
        self.prob_in_facility_given_anc_dhs = pd.Series(
            {
                ANC0: self.birth_facility_data.at[
                    ("P(facility|ANC0)", "DHS"), "Value"
                ],  # constraint 1
                ANC1: self.birth_facility_data.at[
                    ("P(facility|ANC1)", "DHS"), "Value"
                ],  # constraint 2
            },
            index=ANC_INDEX,
            name="prob_in_facility_given_anc",
        )
        # OR (for 3 facility types):
        prob_facility_given_anc_dhs_dict = {
            ANC0: self.birth_facility_data.loc[
                [
                    ("P(home|ANC0)", "DHS"),  # constraint 1
                    ("P(clinic|ANC0)", "DHS"),  # constraint 2
                    ("P(hospital|ANC0)", "DHS"),
                ],
                "Value",
            ].array,
            ANC1: self.birth_facility_data.loc[
                [
                    ("P(home|ANC1)", "DHS"),  # constraint 3
                    ("P(clinic|ANC1)", "DHS"),  # constraint 4
                    ("P(hospital|ANC1)", "DHS"),
                ],
                "Value",
            ].array,
        }
        self.prob_facility_given_anc_dhs = pd.DataFrame(
            prob_facility_given_anc_dhs_dict.values(),
            index=ANC_INDEX,
            columns=TERNARY_FACILITY_INDEX,
        )
        # Rescale P(in-facility|ANC) to be consistent with
        # P(in-facility) from GBD
        self.prob_in_facility_given_anc = get_consistent_conditional_probabilities(
            self.prob_in_facility_given_anc_dhs,
            self.anc_probabilities,
            self.in_facility_probabilities[IN_FACILITY],
        )
        # OR (for 3 facility types):
        self.prob_facility_given_anc = get_consistent_conditional_probabilities(
            self.prob_facility_given_anc_dhs,
            self.anc_probabilities,
            self.prob_facility,
        )

        # Get preterm births by delivery location
        # 2 facility types - Series
        self.prob_preterm_given_in_facility_raw = pd.Series(
            {
                AT_HOME: self.birth_facility_data.at[  # constraint 3
                    ("P(preterm|home)", self.sources[PROB_PRETERM_GIVEN_FACILITY_SOURCE]),
                    "Value",
                ],
                IN_FACILITY: self.birth_facility_data.at[  # constraint 4
                    ("P(preterm|facility)", self.sources[PROB_PRETERM_GIVEN_FACILITY_SOURCE]),
                    "Value",
                ],
            },
            index=BINARY_FACILITY_INDEX,
            name="prob_preterm_given_in_facility",
        )
        # 2 facility types - DataFrame
        self.prob_term_status_given_in_facility_raw = pd.DataFrame(
            {
                PRETERM: self.prob_preterm_given_in_facility_raw,
                TERM: 1 - self.prob_preterm_given_in_facility_raw,
            },
            columns=TERM_STATUS_INDEX,
        )
        # If we're missing P(preterm|home), for example, fill it in
        # using P(preterm|facility) and GBD data (this is the case for
        # Nigeria, and for Ethiopia if we use the EmONC assessment data)
        if self.prob_preterm_given_in_facility_raw.isna().any():
            # Fill in Series
            self.prob_preterm_given_in_facility_filled = (
                fill_missing_conditional_probabilities(
                    self.prob_preterm_given_in_facility_raw,
                    self.in_facility_probabilities,
                    self.preterm_prevalence,
                )
            )
            # Fill in corresponding DataFrame
            self.prob_term_status_given_in_facility_filled = (
                fill_missing_conditional_probabilities(
                    self.prob_term_status_given_in_facility_raw,
                    self.in_facility_probabilities,
                    self.term_status_probabilities,
                )
            )
        else:
            self.prob_preterm_given_in_facility_filled = (
                self.prob_preterm_given_in_facility_raw
            )
            self.prob_term_status_given_in_facility_filled = (
                self.prob_term_status_given_in_facility_raw
            )

        # 3 facility types - Series
        prob_preterm_given_facility_raw = pd.Series(
            {
                AT_HOME: self.birth_facility_data.at[  # constraint 5
                    ("P(preterm|home)", self.sources[PROB_PRETERM_GIVEN_FACILITY_SOURCE]),
                    "Value",
                ],
                BEMONC: self.birth_facility_data.at[  # constraint 6
                    ("P(preterm|clinic)", self.sources[PROB_PRETERM_GIVEN_FACILITY_SOURCE]),
                    "Value",
                ],
                CEMONC: self.birth_facility_data.at[  # constraint 7
                    ("P(preterm|hospital)", self.sources[PROB_PRETERM_GIVEN_FACILITY_SOURCE]),
                    "Value",
                ],
            },
            index=TERNARY_FACILITY_INDEX,
        )
        # 3 facility types - DataFrame
        self.prob_term_status_given_facility_raw = pd.DataFrame(
            {
                PRETERM: prob_preterm_given_facility_raw,
                TERM: 1 - prob_preterm_given_facility_raw,
            },
            columns=TERM_STATUS_INDEX,
        )

        # Rescale P(preterm|facility) to be consistent with P(preterm)
        # from GBD
        # 2 facility types - Series version:
        self.prob_preterm_given_in_facility = get_consistent_conditional_probabilities(
            self.prob_preterm_given_in_facility_filled,
            self.in_facility_probabilities,
            self.preterm_prevalence,
        )
        # 2 facility types - DataFrame version:
        self.prob_term_status_given_in_facility = get_consistent_conditional_probabilities(
            self.prob_term_status_given_in_facility_filled,
            self.in_facility_probabilities,
            self.term_status_probabilities,
        )
        # OR for 3 facility types - DataFrame:
        self.prob_term_status_given_facility = get_consistent_conditional_probabilities(
            self.prob_term_status_given_facility_raw,
            self.prob_facility,
            self.term_status_probabilities,
        )

    def _set_input_probabilities(self):
        """Save a single Series of probabilities that are used for input
        to the birth facility choice model.
        """
        self.input_probabilities = (
            pd.concat(
                [
                    self.male_female_birth_probabilities,
                    self.anc_probabilities,
                    self.term_status_probabilities,
                    self.in_facility_probabilities,
                    self.prob_facility.loc[[BEMONC, CEMONC]],
                    stack_conditional_probability_table(self.prob_ultrasound_given_anc),
                ]
            )
            .rename_axis("subpopulation")
            .rename("input_probability")
        )

    def _set_target_probabilities(self):
        """Set target probabilities to match in optimization"""
        self.targets_2_facility_types = pd.concat(
            [self.prob_in_facility_given_anc, self.prob_preterm_given_in_facility],
            keys=[IN_FACILITY, PRETERM],
            names=["probability_of", "given"],
        ).rename("target_probabilities")

        self.targets_3_facility_types = TargetProbabilities(
            self.prob_facility_given_anc,
            self.prob_term_status_given_facility,
        )

    def targets_to_series(self, target_probabilities: TargetProbabilities):
        targets_series = (
            pd.concat([df.T.stack() for df in target_probabilities])
            .rename_axis(self.targets_2_facility_types.index.names)
            .rename(self.targets_2_facility_types.name)
        )
        return targets_series


### Functions for initializing propensities


def create_correlation_matrix(
    variable_names,
    # TODO: Allow passing either a dict or a list of correlations.
    # If given a list, then we should be able to initialize the
    # correlation matrix by using np.triu_indices to index the
    # submatrix corresponding to the correlated variables. I'm
    # guessing this could be more efficient than using dictionaries.
    # Hmm, to make this work, we'd need to also pass in the correct
    # indices for the submatrix.
    correlation_map,
    independent_variables=None,
):
    """Create a pandas DataFrame representing the correlation matrix
    determined by the given variable names and correlation map,
    optionally verifying that the correlation map is consistent with
    a given set of variables that are supposed to be independent.
    """
    n_variables = len(variable_names)
    # Initialize a correlation matrix for the variables
    corr = pd.DataFrame(
        np.zeros((n_variables, n_variables)),
        index=variable_names,
        columns=variable_names,
    )
    # Fill in correlations based on the map
    for (row, col), value in correlation_map.items():
        if row not in variable_names or col not in variable_names:
            raise ValueError(
                "correlation_map contains a key with invalid variables: "
                f"{(row, col)}. {variable_names=}"
            )
        corr.at[row, col] = value
    # Symmetrize
    corr = corr + corr.T + np.identity(n_variables)

    assert corr.equals(corr.T), "Correlation matrix not symmetric!"
    if independent_variables is not None:
        # Insure that each independent variable is actually independent
        # of all other variables
        assert (((corr - np.identity(n_variables)).loc[independent_variables]) == 0).all(
            None
        ), "Independence specification violated!"
    return corr


def get_copula(correlation_matrix) -> GaussianCopula:
    """Return a Gaussian copula defined by the given correlation matrix."""
    k_dim = len(correlation_matrix)
    copula = GaussianCopula(correlation_matrix, k_dim, allow_singular=True)
    assert (copula.corr == correlation_matrix).all(None), "Copula initialized incorrectly!"
    return copula


### Functions to calculate calibration targets


def prob_x_given_y(x: pd.Series, y: pd.Series):
    """Generates a table of the empirical probabilities of each value
    of x (in rows) given each value of y (in columns).
    x and y are assumed to be pandas Series of categorical data with
    matching indices, and y must be named.
    """
    counts = pd.concat([x, y], axis=1).value_counts().unstack(y.name)
    return counts / counts.sum()


def prob_y_given_x(y: pd.Series, x: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Generates a table of the empirical probabilities of each value
    of y (in columns) given each value of x (in rows).
    x and y are assumed to have matching indices and contain categorical
    data. y must be a Series, and x can be a Series or a DataFrame with
    multiple columns to condition on.
    """
    counts = (
        pd.concat([x, y], axis=1).value_counts()
        # unstack y values, which should be in the 2nd index level
        .unstack()
        # When we unstack, some strata may be missing, indicating that
        # the count was 0, so fill NA with 0
        .fillna(0)
    )
    # Need to use .divide instead of / to align along axis=0
    return counts.divide(counts.sum(axis=1), axis=0)


### Loss functions
def loss_2_facility_types(target_probabilities, empirical_probabilities):
    return log_loss(target_probabilities, empirical_probabilities)


def loss_3_facility_types(
    target_probabilities: TargetProbabilities,
    empirical_probabilities: TargetProbabilities,
):
    loss = log_loss(target_probabilities[0], empirical_probabilities[0]) + log_loss(
        target_probabilities[1], empirical_probabilities[1]
    )
    # OR:
    # loss = sum(map(lambda t: log_loss(*t),
    #                zip(*(target_probabilities, empirical_probabilities))))
    return loss


### Base class to encapsulate a birth location choice model


class CausalModelNanosim:
    # This will get overwritten with a real copula as long as the list
    # of model variables is nonempty
    copula: GaussianCopula | None = None
    # Declare propensity DataFrame but don't initialize since we may
    # not have passed in the desired correlation map yet
    propensity: pd.DataFrame | None = None
    # Initialize population to None to flag that it needs to be
    # created from scratch rather than overwritten
    population: pd.DataFrame | None = None
    seed: Seed

    def __init__(
        self,
        data: CausalModelData,
        population_size: int,
        endogenous_variables: Sequence[Type[ModelVariable]],
        correlated_propensities: Iterable,
        correlation_map: dict | None = None,
        # Can be any seed that can be passed to
        # numpy.random.default_rng()
        seed: Seed | None = None,
    ):
        self.data = data
        self.pop_index = pd.Index(range(population_size), name="simulant_id")

        # Initialize all model variables, passing self as the model
        self.endogenous_variables = [
            variable_type(self) for variable_type in endogenous_variables
        ]
        # self._set_propensity_names(correlated_propensities)
        self.propensity_names = pd.Index(
            [
                variable.propensity_name
                for variable in self.endogenous_variables
                if variable.has_propensity
            ]
        )
        self.correlated_propensities = pd.Index(correlated_propensities)
        self.independent_propensities = self.propensity_names.difference(
            self.correlated_propensities
        )
        self.num_correlated_propensities = len(self.correlated_propensities)
        # There's one correlation for each pair of correlated variables
        self.num_correlations = scipy.special.comb(
            self.num_correlated_propensities, 2, exact=True
        )
        # These will be initialized to zero correlation (independence)
        # if correlation_map is None
        self.set_correlation_matrix_and_copula(correlation_map)
        if seed is None:
            self.seed = SeedSequence()
            print(f"seed={self.seed}")
        else:
            self.seed = seed

    def names_to_variables_map(self) -> dict[str, ModelVariable]:
        """Get a mapping from variable names to variable objects."""
        return {variable.name: variable for variable in self.endogenous_variables}

    def correlated_pairs(self) -> list[tuple[str, str]]:
        """Returns a list of pairs of names of correlated propensities."""
        correlated_pairs = [
            (propensity_i, propensity_j)
            for i, propensity_i in enumerate(self.correlated_propensities)
            for propensity_j in self.correlated_propensities[i + 1 :]
        ]
        return correlated_pairs

    def set_correlation_matrix_and_copula(
        self,
        correlation_map: Mapping[tuple[str, str], float] | None,
    ):
        """Initializes the correlation matrix and corresponding Gaussian
        copula using the given correlation map, which should map pairs
        (2-tuples) of variable names (strings) to their correlations.
        """
        # If correlation_map is None, set it to an empty dict (i.e., no
        # correlations) so that the correlation matrix will be the
        # identity and the copula will be the independence copula.
        if correlation_map is None:
            correlation_map = {}

        self.correlation_matrix = create_correlation_matrix(
            self.propensity_names,
            correlation_map,
            self.independent_propensities,
        )
        # This check is only here to enable instantiating a model with
        # no variables, for testing purposes. If this check fails,
        # self.copula will be None by default
        if len(self.propensity_names) > 0:
            self.copula = get_copula(self.correlation_matrix)

    # TODO: Maybe instead of annotating with int, specify only the
    # literals 1, 0, or -1 are allowed
    def set_correlations(self, *correlations: float | int):
        """Set the correlations in this model's correlation matrix to
        the numbers passed in the list
        """
        # correlation_map = {
        #     (vi, vj): correlations[
        #         # NOTE: This formula maps a pair of indices i, j to the
        #         # index where it will appear if the pairs are listed
        #         # lexicographically.
        #         # TODO: Explain why this formula works. In particular,
        #         # why do you have to subtract off a triangular number?
        #         i * self.num_correlated_propensities + j - i * (i + 1) // 2]
        #     for i, vi in enumerate(self.correlated_propensities)
        #     for j, vj in enumerate(self.correlated_propensities[i+1:])
        # }
        # Here's an easier to understand algorithm:
        correlated_pairs = self.correlated_pairs()
        if len(correlations) != len(correlated_pairs):
            return ValueError(
                f"{len(correlations)} correlations were passed,"
                f" but {len(correlated_pairs)} are needed"
            )
        correlation_map = {
            pair: correlation for pair, correlation in zip(correlated_pairs, correlations)
        }
        # TODO: See if I can figure out a way to adapt
        # create_correlation_matrix to directly accept the list of
        # correlations so we don't have to convert to a dictionary first
        self.set_correlation_matrix_and_copula(correlation_map)

    def correlation_map(self) -> dict[tuple[str, str], float]:
        """Return the correlation map (dict) corresponding to this
        model's correlation matrix. The returned dictionary maps pairs
        of distinct correlated variables to their numerical correlation.
        """
        correlation_map = {
            (propensity1, propensity2): self.correlation_matrix.at[propensity1, propensity2]
            for propensity1, propensity2 in self.correlated_pairs()
        }
        return correlation_map

    def sample_propensities(self, seed=LAST_USED):
        """Sample propensities for a population using a Gaussian copula,
        and store them in self.propensity.
        """
        if seed != LAST_USED:
            self.seed = seed
        # Convert seed to a Generator - then scipy and statsmodels will
        # use the Generator object instead of the old-school RandomState
        rng = np.random.default_rng(self.seed)

        self.propensity = pd.DataFrame(
            self.copula.rvs(len(self.pop_index), random_state=rng),
            index=self.pop_index,
            columns=self.propensity_names,
        )

    def empirical_propensity_correlations(self) -> pd.DataFrame | None:
        """Computes the empirical correlation matrix (Pearson
        product-moment correlation coefficients) for this model's
        sampled propensities using numpy.corrcoef, or returns None if
        the propensities haven't been sampled yet.
        """
        if self.propensity is None:
            # Return None instead of raising an error
            return None
        empirical_correlation_matrix = pd.DataFrame(
            # Variables correspond to columns not rows, so rowvar=False
            np.corrcoef(self.propensity, rowvar=False),
            index=self.propensity.columns,
            columns=self.propensity.columns,
        )
        return empirical_correlation_matrix

    def assign_population_variables(self):
        """ "Initialize the population DataFrame by following the order
        of the causal graph to construct the columns.
        """
        # If population has already been initialized (i.e., is not
        # None), just write columns directly to the population
        # DataFrame. Otherwise, write them to a new dictionary and then
        # concatenate to create the population dataframe.
        if self.population is None:
            population_columns = {}
        else:
            population_columns = self.population

        for variable in self.endogenous_variables:
            # Get parent columns, which need to have already been
            # initialized
            parent_columns = [population_columns[name] for name in variable.parents]
            # Construct this variable's column, passing the values of
            # the parents' columns as data
            population_columns[variable.name] = variable.get_population_column(
                *parent_columns
            )
        # If population hadn't been initialized, create it by
        # concatenating
        if self.population is None:
            self.population = pd.DataFrame(population_columns, index=self.pop_index)

    def process_input_vector(self, x):
        # NOTE: All current models call
        # self.set_correlations(*x[:self.num_correlations]), so maybe I
        # could do that here by default, then they call
        # self.set_facility_choice_probabilities(*x[self.num_correlations:]),
        # which needs to be defined in subclasses.
        raise NotImplementedError("Subclasses must define how to process x")

    def do_all_steps(self, x, seed=LAST_USED):
        # NOTE: Should this accept seed as a second argument?
        self.process_input_vector(x)
        self.sample_propensities(seed)
        self.assign_population_variables()


### Subclasses definining birth location choice models


class BirthFacilityChoiceModel(CausalModelNanosim):
    # The names of these correlated propensities are provided as
    # defaults so that subclasses don't have to define them.
    correlated_propensities: tuple[str, ...] = (ANC, LBWSG_CAT, FACILITY)
    # Subclasses should define the endogenous_variables class variable.
    # This default value of () only exists to allow instantiating this
    # base class without errors.
    endogenous_variables: tuple[Type[ModelVariable], ...] = ()

    def __init__(
        self,
        data,
        population_size,
        num_facility_types=2,
        correlation_map=None,
        seed=None,
        # NOTE: Parameter order is different from superclass --
        # endogenous_variables is moved to last because it is
        # expected we'll usually use the default sequence defined by
        # each subclass.
        endogenous_variables: Sequence[Type[ModelVariable]] | None = None,
        correlated_propensities: Sequence[str] | None = None,
    ):
        # If variables are not explicitly specified, use variables
        # defined by the class as defaults for the instance variables.
        # These class variables can and should be overridden by
        # subclasses to provide meaningful defaults.
        if endogenous_variables is None:
            endogenous_variables = self.endogenous_variables
        if correlated_propensities is None:
            correlated_propensities = self.correlated_propensities

        super().__init__(
            data,
            population_size,
            endogenous_variables,
            correlated_propensities,
            correlation_map,
            seed,
        )
        if num_facility_types == 2:
            self.facility_index = BINARY_FACILITY_INDEX
            self.targets = self.data.targets_2_facility_types
            # for compatibility with 3-type model
            self.targets_series = self.targets
            self.loss_func = loss_2_facility_types
        elif num_facility_types == 3:
            self.facility_index = TERNARY_FACILITY_INDEX
            self.targets = self.data.targets_3_facility_types
            # for conveience, also store targets as a single series
            # instead of a tuple
            self.targets_series = self.data.targets_to_series(self.targets)
            self.loss_func = loss_3_facility_types
        else:
            raise ValueError(
                "Must have either 2 or 3 facility types," f" not {num_facility_types!r}."
            )

        self.num_facility_types = num_facility_types
        # Initialize conditional facility probabilitiy DataFrame filled
        # with zeros for two or three facility types
        self.prob_facility_given_believed_term_status = pd.DataFrame(
            0.0, index=BELIEVED_TERM_STATUS_INDEX, columns=self.facility_index
        )
        # Store data about parameters this model optimizes over
        self.parameter_data = self._get_parameter_data()
        # Compute and store minimum possible loss, which for log loss is
        # the average entropy of the target probability distributions
        self.minimum_loss = self.loss_func(self.targets, self.targets)

    def _get_parameter_data(self):
        # Count the number of probabability parameters: For n-1
        # independent facility types F, need P(F|preterm) and P(F|term)
        num_probs = len(BELIEVED_TERM_STATUS_INDEX) * (self.num_facility_types - 1)
        # Get names of facility choice probabilities using introspection
        facility_prob_parameter_names = [
            name
            for name in inspect.signature(self.set_facility_choice_probabilities).parameters
        ]
        # Truncate to correct number of probabilities depending on
        # number of facility types
        facility_prob_parameter_names = facility_prob_parameter_names[:num_probs]
        # Get data about correlations
        correlated_pairs = self.correlated_pairs()
        correlation_names = ["corr(" + ", ".join(pair) + ")" for pair in correlated_pairs]
        # Specify bounds for valid correlations and valid probabilities
        correlation_bounds = [(-1, 1)] * len(correlated_pairs)
        probability_bounds = [(0, 1)] * num_probs
        # Get names of all parameters
        parameter_names = correlation_names + facility_prob_parameter_names
        parameter_data = pd.DataFrame(
            {
                "correlated_pair": correlated_pairs + [pd.NA] * num_probs,
                "bounds": correlation_bounds + probability_bounds,
                "position": range(len(parameter_names)),
            },
            index=pd.Index(parameter_names, name="parameter_name"),
        ).rename_axis(columns="attribute")
        return parameter_data

    def set_facility_choice_probabilities(
        self,
        prob_home_given_believed_preterm,
        prob_home_given_believed_term,
        prob_bemonc_given_believed_preterm=None,
        prob_bemonc_given_believed_term=None,
    ):
        # NOTE: Order must be preterm, term
        self.prob_facility_given_believed_term_status[AT_HOME] = (
            prob_home_given_believed_preterm,
            prob_home_given_believed_term,
        )
        columns_assigned = [AT_HOME]
        column_to_assign = IN_FACILITY
        if self.num_facility_types == 3:
            # NOTE: Order must be preterm, term
            self.prob_facility_given_believed_term_status[BEMONC] = (
                prob_bemonc_given_believed_preterm,
                prob_bemonc_given_believed_term,
            )
            columns_assigned.append(BEMONC)
            column_to_assign = CEMONC
        elif (
            prob_bemonc_given_believed_preterm is not None
            or prob_bemonc_given_believed_term is not None
        ):
            raise TypeError(
                "Only two probabilities can be specified when"
                " the number of facility types is 2"
            )

        self.prob_facility_given_believed_term_status[
            column_to_assign
        ] = 1 - self.prob_facility_given_believed_term_status[columns_assigned].sum(axis=1)

    def process_input_vector(self, x):
        # Correlations need to come first in x
        self.set_correlations(*x[: self.num_correlations])
        # Probabilities need to come last in x
        self.set_facility_choice_probabilities(*x[self.num_correlations :])

    def get_observed_facility_choice_probabilities(self):
        observed_probabilities = prob_y_given_x(
            self.population[FACILITY], self.population[BELIEVED_TERM_STATUS]
        )
        return observed_probabilities

    def get_observed_vs_causal_facility_choice_probabilities(self):
        causal_probabilities = stack_conditional_probability_table(
            self.prob_facility_given_believed_term_status
        )
        observed_probabilities = stack_conditional_probability_table(
            self.get_observed_facility_choice_probabilities()
        )
        all_probabilities = pd.concat(
            [causal_probabilities, observed_probabilities],
            keys=["causal", "observed"],
            axis=1,
        )
        return all_probabilities

    def calculate_targets(self, as_series=False):
        # Name of returned Series in 2-facility-type or as_series cases
        name = "empirical_probabilities"
        index = self.targets_series.index
        # NOTE: I ran into an issue where the first iteration of an
        # optimization failed, leaving self.population uninitialized,
        # causing an error to be thrown when calling calculate_targets.
        # So I'm explicitly testing for None here, which seems a bit
        # hacky, but it should do the job...
        if self.population is None:
            if self.num_facility_types == 2 or as_series:
                return pd.Series(np.nan, index=index, name=name)
            else:
                empirical_targets = TargetProbabilities(*(df.copy() for df in self.targets))
                for df in empirical_targets:
                    df.loc[:, :] = np.nan
                return empirical_targets

        # NOTE: Using .loc instead of [] because .loc is optimized:
        # https://pandas.pydata.org/docs/user_guide/indexing.html
        prob_facility_given_anc = prob_y_given_x(
            self.population.loc[:, FACILITY], self.population.loc[:, ANC]
        )
        prob_term_status_given_facility = prob_y_given_x(
            self.population.loc[:, TERM_STATUS],
            self.population.loc[:, FACILITY],
        )
        if self.num_facility_types == 2:
            empirical_targets = (
                pd.concat(
                    [
                        prob_facility_given_anc.loc[:, IN_FACILITY],
                        prob_term_status_given_facility.loc[:, PRETERM],
                    ],
                    keys=[IN_FACILITY, PRETERM],
                    names=index.names,
                    # Reindex in case there are any strata missing
                )
                .rename(name)
                .reindex(index, fill_value=0)
            )
        else:
            empirical_targets = TargetProbabilities(
                # Reindex in case there are any strata missing
                prob_facility_given_anc.reindex_like(self.targets.facility_given_anc).fillna(
                    0
                ),
                prob_term_status_given_facility.reindex_like(
                    self.targets.term_status_given_facility
                ).fillna(0),
            )
            if as_series:
                empirical_targets = self.data.targets_to_series(empirical_targets)

        # empirical_targets = calculate_targets(
        #     self.population
        # # Reindex in case there are any strata missing
        # ).reindex(self.targets.index, fill_value=0)
        return empirical_targets

    def get_population_proportions(self) -> pd.Series:
        """Compute population proportions that are expected to match
        input values.
        """
        index = self.data.input_probabilities.index
        name = "population_proportion"
        if self.population is None:
            return pd.Series(np.nan, index=index, name=name)

        # if self.num_facility_types == 2:
        #     facility_dtype = BINARY_FACILITY_DTYPE
        # elif self.num_facility_types == 3:
        #     facility_dtype = TERNARY_FACILITY_DTYPE

        proportions = []
        # For each variable of interest, check if it's in the model, and
        # if so, compute the empirical probability of each of its
        # categories. Use the category order in the corresponding dtype
        # to get a consistent ordering regardless of the order returned
        # by .value_counts. OOPS: If one of the categories is missing by
        # chance, using .loc like I had been would result in a KeyError
        if SEX in self.population:
            proportions.append(self.population.loc[:, SEX].value_counts(normalize=True))
        if ANC in self.population:
            proportions.append(self.population.loc[:, ANC].value_counts(normalize=True))
        if TERM_STATUS in self.population:
            proportions.append(
                self.population.loc[:, TERM_STATUS].value_counts(normalize=True)
            )
        if FACILITY in self.population:
            proportions.append(self.population.loc[:, FACILITY].value_counts(normalize=True))
            if self.num_facility_types == 3:
                # Add a row for in-facility delivery proportion
                proportions.append(
                    pd.Series(
                        proportions[-1].loc[[BEMONC, CEMONC]].sum(), index=[IN_FACILITY]
                    )
                )
        if ULTRASOUND in self.population:
            # Get DataFrame of conditional probabilities of ultrasound
            # given ANC, then convert to a Series with a simple Index
            proportions.append(
                stack_conditional_probability_table(
                    prob_y_given_x(
                        self.population.loc[:, ULTRASOUND], self.population.loc[:, ANC]
                    )
                )
            )
        # NOTE: Reindexing to input_probabilities Series may result in
        # NaNs for two reasons: 1) The model is missing one of the
        # variables (e.g., sex), or 2) The model has the variable, but
        # by random chance happens to be missing one of the strata
        # (e.g., all simulants are female). To distinguish between these
        # cases, we could instead .reindex each of the above Series by
        # the categories of the corresponding dtype (instead of using
        # .loc like I had been doing before), and fill NaN's wtih 0.
        # Then 0s would indicate a missing stratum, while NaNs would
        # indicate a missing variable.

        # Use index of input_probabilities to get a consistent ordering
        proportions_series = pd.concat(proportions).reindex(index).rename(name)
        return proportions_series

    def get_population_proportions_vs_input(self) -> pd.DataFrame:
        proportions = pd.concat(
            [
                self.data.input_probabilities,
                self.get_population_proportions(),
            ],
            axis=1,
        )
        return proportions

    def loss(self) -> float:
        """Return the loss for the current population."""
        return self.loss_func(self.targets, self.calculate_targets())

    def normalized_loss(self) -> float:
        """Return the normalized (0-based) loss for the current
        population.
        """
        return self.loss() - self.minimum_loss


def uniform_from_propensity(propensity, low, high):
    """Scale a Uniform(0,1) random variable to a Uniform(low, high)
    random variable.
    """
    return low + propensity * (high - low)


# Sentinel indicating that propensity name should be the same as the
# model variable name. See
# https://peps.python.org/pep-0661/#use-a-single-valued-enum
class DefaultNameType(Enum):
    DEFAULT_NAME = "DEFAULT_NAME"


DEFAULT_NAME = DefaultNameType.DEFAULT_NAME


class ModelVariable:
    # Each subclass must define its name and parents
    # NOTE: When using dataclasses, uncommenting
    # these two fields caused an error when trying to initialize them in
    # a subclass:
    # ValueError: __init__() requires a code object with 1 free vars, not 0
    name: str
    parents: tuple[str, ...]
    propensity_name: str | DefaultNameType | None = DEFAULT_NAME
    # has_propensity: bool = True

    def __init__(self, model: CausalModelNanosim):
        self.model = model
        if self.propensity_name is DEFAULT_NAME and hasattr(self, "name"):
            self.propensity_name = self.name

    def get_population_column(self, *parent_values: pd.Series) -> pd.Series:
        raise NotImplementedError("Must be defined by subclasses.")

    @property
    def has_propensity(self) -> bool:
        return self.propensity_name is not None


class Sex(ModelVariable):
    name: str = SEX
    parents: tuple[()] = ()

    def get_population_column(self) -> pd.Series:
        # Arbitrarily, larger propensities correspond to Female
        # because data.male_female_birth_probabilities is
        # ordered ['Male', 'Female']
        sex_col = sample_series_from_propensity(
            self.model.propensity[self.propensity_name],
            SEX_DTYPE,
            # Categories are guaranteed to be in correct order because
            # the order of the categories in
            # male_female_birth_probabilities was defined to be the same
            # as SEX_DTYPE.categories
            self.model.data.male_female_birth_probabilities.cumsum(),
            method="select",
            index=self.model.pop_index,
            name=self.name,
        )
        return sex_col


class ANCAttendance(ModelVariable):
    name: str = ANC
    parents: tuple[()] = ()

    def get_population_column(self) -> pd.Series:
        # Note: data.anc_probabilities needs to be defined
        # with ANC0 first and ANC1 second, so that larger
        # propensities correspond to better outcomes (i.e. ANC1)
        anc_col = sample_series_from_propensity(
            self.model.propensity[self.propensity_name],
            ANC_DTYPE,
            # Categories are guaranteed to be in correct order because
            # the order of the categories in anc_probabilities was
            # defined to be the same as ANC_DTYPE.categories
            self.model.data.anc_probabilities.cumsum(),
            method="select",
            index=self.model.pop_index,
            name=self.name,
        )
        return anc_col


class TermStatusFromPropensity(ModelVariable):
    """Get term status directly from GBD's preterm prevalence"""

    name: str = TERM_STATUS
    parents: tuple[()] = ()
    # Use LBWSG category propensity for consistency between models
    propensity_name: str = LBWSG_CAT

    def get_population_column(self):
        term_status_col = sample_series_from_propensity(
            self.model.propensity[self.propensity_name],
            TERM_STATUS_DTYPE,
            # Create CDF on the fly
            [self.model.data.preterm_prevalence, 1],
            method="array",
            index=self.model.pop_index,
            name=self.name,
        )
        return term_status_col


class BelievedTermStatusGivenTermStatusWithNoUltrasound(ModelVariable):
    name: str = BELIEVED_TERM_STATUS
    parents: tuple[str] = (TERM_STATUS,)
    # Use GA error propensity for consistency between models
    propensity_name: str = GA_ERROR

    def get_population_column(self, term_status):
        pop_believed_term_cdf = term_status.to_frame().join(
            self.model.data.prob_believed_term_status_given_term_status_and_ultrasound.xs(
                NO_ULTRASOUND, level=ULTRASOUND
            ).cumsum(axis=1),
            on=TERM_STATUS,
        )
        believed_term_status_col = sample_series_from_propensity(
            self.model.propensity[self.propensity_name],
            BELIEVED_TERM_STATUS_DTYPE,
            pop_believed_term_cdf,
            method="select",
            index=self.model.pop_index,
            name=self.name,
        )
        return believed_term_status_col


class UltrasoundTypeGivenANC(ModelVariable):
    name: str = ULTRASOUND
    parents: tuple[str] = (ANC,)

    def get_population_column(self, anc: pd.Series) -> pd.Series:
        pop_ultrasound_cdf = (
            anc.to_frame().join(
                self.model.data.prob_ultrasound_given_anc.cumsum(axis=1), on=ANC
            )
            # NOTE: Explicitly subsetting to the ultrasound columns
            # (i.e., dropping the ANC column) is not necessary if using
            # method='select' below, but it is necessary if using
            # method='array'
            [ULTRASOUND_DTYPE.categories]
        )
        ultrasound_col = sample_series_from_propensity(
            self.model.propensity[self.propensity_name],
            ULTRASOUND_DTYPE,
            # Note that the columns in the ultrasound CDF were defined
            # to be in the same order as in ULTRASOUND_DTYPE, which is
            # necessary for this to work correctly
            pop_ultrasound_cdf,
            method="select",
            index=self.model.pop_index,
            name=self.name,
        )
        return ultrasound_col


class LBWSGCategoryGivenSex(ModelVariable):
    name: str = LBWSG_CAT
    parents: tuple[str] = (SEX,)

    def get_population_column(self, sex: pd.Series) -> pd.Series:
        # Create an empty Series to store the data since we have to fill
        # in rows for males and females separately
        lbwsg_cat_col = pd.Series(
            index=self.model.pop_index,
            dtype=self.model.data.lbwsg_category_dtype,
            name=self.name,
        )
        # Need to assign categories for males and females separately
        # because the category ordering is different, so we need to pass
        # different CDFs to assign categories from propensities
        for sex_value, cat_cdf in (
            self.model.data
            # TODO: Add an option (e.g. to this variable's constructor)
            # to specify whether we want to include preterm in the
            # sorting or not, in order to rerun previous model versions
            .lbwsg_birth_exposure_cdf_sorted_by_preterm_and_rr.items()
        ):
            sex_index = sex == sex_value
            # Forget ordering of categories to avoid an error,
            # because orders for Male and Female are incompatible
            sampled_categories = sample_categorical_from_propensity(
                self.model.propensity.loc[sex_index, self.propensity_name],
                cat_cdf.columns,
                cat_cdf,
                method="select",
            ).astype(lbwsg_cat_col.dtype)
            lbwsg_cat_col.loc[sex_index] = sampled_categories
        return lbwsg_cat_col


class TermStatusFromLBWSGCategory(ModelVariable):
    name: str = TERM_STATUS
    parents: tuple[str] = (LBWSG_CAT,)
    propensity_name: None = None

    def get_population_column(self, lbwsg_category: pd.Series) -> pd.Series:
        term_status_col = pd.Series(
            pd.Categorical.from_codes(
                (~lbwsg_category.isin(self.model.data.preterm_categories)).astype("uint8"),
                dtype=TERM_STATUS_DTYPE,
            ),
            index=self.model.pop_index,
            name=self.name,
        )
        return term_status_col


class GestationalAgeGivenLBWSGCategory(ModelVariable):
    name: str = GA
    parents: tuple[str] = (LBWSG_CAT,)

    def get_population_column(self, lbwsg_category):
        # Get upper and lower bounds for GA from LBWSG category
        ga_cols = ["ga_start", "ga_end"]
        ga_endpoints = self.model.data.cat_df.set_index("lbwsg_category")[ga_cols]
        pop_ga_bounds = lbwsg_category.to_frame().join(ga_endpoints, on=LBWSG_CAT)[ga_cols]
        # Scale the propensity to be uniform between the bounds
        gestational_age_col = uniform_from_propensity(
            self.model.propensity[self.propensity_name],
            pop_ga_bounds[ga_cols[0]],
            pop_ga_bounds[ga_cols[1]],
        )
        return gestational_age_col


class TermStatusFromGestationalAge(ModelVariable):
    name: str = TERM_STATUS
    parents: tuple[str] = (GA,)
    propensity_name: None = None

    def get_population_column(self, gestational_age: pd.Series) -> pd.Series:
        term_status_col = pd.Series(
            pd.Categorical.from_codes(
                (gestational_age >= PRETERM_CUTOFF).astype("uint8"), dtype=TERM_STATUS_DTYPE
            ),
            index=self.model.pop_index,
            name=self.name,
        )
        return term_status_col


class BelievedTermStatusGivenTermStatusAndUltrasound(ModelVariable):
    name: str = BELIEVED_TERM_STATUS
    parents: tuple[str, str] = (TERM_STATUS, ULTRASOUND)

    def get_population_column(
        self,
        term_status: pd.Series,
        ultrasound: pd.Series,
    ) -> pd.Series:
        pop_believed_term_cdf = pd.concat([term_status, ultrasound], axis=1).join(
            (
                self.model.data.prob_believed_term_status_given_term_status_and_ultrasound.cumsum(
                    axis=1
                )
            ),
            on=[TERM_STATUS, ULTRASOUND],
        )
        believed_term_status_col = sample_series_from_propensity(
            self.model.propensity[self.propensity_name],
            BELIEVED_TERM_STATUS_DTYPE,
            pop_believed_term_cdf,
            method="select",
            index=self.model.pop_index,
            name=self.name,
        )
        return believed_term_status_col


class GestationalAgeErrorGivenUltrasound(ModelVariable):
    name: str = GA_ERROR
    parents: tuple[str] = (ULTRASOUND,)

    def get_population_column(self, ultrasound: pd.Series) -> pd.Series:
        pop_std = ultrasound.map(self.model.data.st_dev_of_ga_error_given_ultrasound)
        # Use inverse transform sampling to sample normal random
        # variables with the specified standard deviation for each row
        ga_error_col = scipy.stats.norm.ppf(
            self.model.propensity[self.propensity_name], scale=pop_std
        )
        return ga_error_col


class EstimatedGestationalAgeFromGestationalAgeAndGAError(ModelVariable):
    name: str = GA_ESTIMATE
    parents: tuple[str, str] = (GA, GA_ERROR)
    propensity_name: None = None

    def get_population_column(self, gestational_age, ga_error):
        return gestational_age + ga_error


class BelievedTermStatusFromEstimatedGestationalAge(ModelVariable):
    name: str = BELIEVED_TERM_STATUS
    parents: tuple[str] = (GA_ESTIMATE,)
    propensity_name: None = None

    def get_population_column(self, estimated_gestational_age: pd.Series) -> pd.Series:
        term_status_col = pd.Series(
            pd.Categorical.from_codes(
                (estimated_gestational_age >= PRETERM_CUTOFF).astype("uint8"),
                dtype=BELIEVED_TERM_STATUS_DTYPE,
            ),
            index=self.model.pop_index,
            name=self.name,
        )
        return term_status_col


# NOTE: This class is slightly different from the others because its
# data comes from the model itself rather than model.data. That's
# because it's the model's job to store the facility choice
# probabilities since they're continuously updated during optimization,
# rather than the job of the Data class, which stores fixed input data.
class BirthFacilityGivenBelievedTermStatus(ModelVariable):
    name: str = FACILITY
    parents: tuple[str] = (BELIEVED_TERM_STATUS,)

    def get_population_column(self, believed_term_status):
        facility_cdf = (
            # Note probabilities come from model, not model.data
            self.model.prob_facility_given_believed_term_status.cumsum(axis=1)
        )
        pop_cdf = (
            believed_term_status.to_frame().join(facility_cdf, on=BELIEVED_TERM_STATUS)[
                facility_cdf.columns
            ]
            # .drop(columns=BELIEVED_TERM_STATUS)
        )
        facility_col = sample_series_from_propensity(
            self.model.propensity[self.propensity_name],
            # facility dtype could be binary or ternary
            facility_cdf.columns.dtype,
            pop_cdf,
            method="select",
            # Pass NaN as a default in case this gets called with a CDF
            # of all 0's, which indicates that the facility choice
            # probabilities haven't been specified yet
            default_category=np.nan,
            index=self.model.pop_index,
            name=self.name,
        )
        return facility_col


class SimplifiedBirthFacilityModel(BirthFacilityChoiceModel):
    # correlated_propensities: tuple[str, str, str] = (
    #     ANC, TERM_STATUS, FACILITY)
    endogenous_variables: tuple[Type[ModelVariable], ...] = (
        ANCAttendance,
        TermStatusFromPropensity,
        BelievedTermStatusGivenTermStatusWithNoUltrasound,
        BirthFacilityGivenBelievedTermStatus,
    )


class SimplifiedBirthFacilityModelWithLBWSG(BirthFacilityChoiceModel):
    # correlated_propensities: tuple[str, str, str] = (
    #     ANC, TERM_STATUS, FACILITY)
    endogenous_variables: tuple[Type[ModelVariable], ...] = (
        Sex,
        ANCAttendance,
        LBWSGCategoryGivenSex,
        TermStatusFromLBWSGCategory,
        BelievedTermStatusGivenTermStatusWithNoUltrasound,
        BirthFacilityGivenBelievedTermStatus,
    )


class SimplifiedBirthFacilityModelWithUltrasound(BirthFacilityChoiceModel):
    # correlated_propensities: tuple[str, str, str] = (
    #     ANC, TERM_STATUS, FACILITY)
    endogenous_variables: tuple[Type[ModelVariable], ...] = (
        ANCAttendance,
        UltrasoundTypeGivenANC,
        TermStatusFromPropensity,
        BelievedTermStatusGivenTermStatusAndUltrasound,
        BirthFacilityGivenBelievedTermStatus,
    )


class BirthFacilityModelWithLBWSGandUltrasound(BirthFacilityChoiceModel):
    endogenous_variables: tuple[Type[ModelVariable], ...] = (
        Sex,
        ANCAttendance,
        UltrasoundTypeGivenANC,
        LBWSGCategoryGivenSex,
        TermStatusFromLBWSGCategory,
        BelievedTermStatusGivenTermStatusAndUltrasound,
        BirthFacilityGivenBelievedTermStatus,
    )


class BirthFacilityModelWithContinuousGA(BirthFacilityChoiceModel):
    endogenous_variables: tuple[Type[ModelVariable], ...] = (
        Sex,
        ANCAttendance,
        UltrasoundTypeGivenANC,
        LBWSGCategoryGivenSex,
        GestationalAgeGivenLBWSGCategory,
        TermStatusFromGestationalAge,
        BelievedTermStatusGivenTermStatusAndUltrasound,
        BirthFacilityGivenBelievedTermStatus,
    )


class BirthFacilityModelWithUltrasoundAndSimpleGAError(BirthFacilityChoiceModel):
    endogenous_variables: tuple[Type[ModelVariable], ...] = (
        Sex,
        ANCAttendance,
        UltrasoundTypeGivenANC,
        LBWSGCategoryGivenSex,
        GestationalAgeGivenLBWSGCategory,
        TermStatusFromGestationalAge,
        GestationalAgeErrorGivenUltrasound,
        EstimatedGestationalAgeFromGestationalAgeAndGAError,
        BelievedTermStatusFromEstimatedGestationalAge,
        BirthFacilityGivenBelievedTermStatus,
    )
