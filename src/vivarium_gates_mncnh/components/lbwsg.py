from __future__ import annotations

import itertools
import math
import pickle
import re
from typing import Any

import numpy as np
import pandas as pd
from vivarium.component import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium_public_health.risks.data_transformations import (
    get_exposure_post_processor,
)
from vivarium_public_health.risks.distributions import MissingDataError
from vivarium_public_health.risks.implementations.low_birth_weight_and_short_gestation import (
    LBWSGDistribution,
)
from vivarium_public_health.risks.implementations.low_birth_weight_and_short_gestation import (
    LBWSGRisk as LBWSGRisk_,
)
from vivarium_public_health.risks.implementations.low_birth_weight_and_short_gestation import (
    LBWSGRiskEffect as LBWSGRiskEffect_,
)
from vivarium_public_health.utilities import TargetString, to_snake_case

from vivarium_gates_mncnh.constants import data_keys, metadata
from vivarium_gates_mncnh.constants.data_values import (
    CHILD_LOOKUP_COLUMN_MAPPER,
    COLUMNS,
    PIPELINES,
    PREGNANCY_OUTCOMES,
    SIMULATION_EVENT_NAMES,
)

CATEGORICAL = "categorical"
BIRTH_WEIGHT = "birth_weight"
GESTATIONAL_AGE = "gestational_age"


class OrderedLBWSGDistribution(LBWSGDistribution):
    """This class allows us to use sex-specific custom ordering for our LBWSG categories
    when determining exposure."""

    AXES = [BIRTH_WEIGHT, GESTATIONAL_AGE]

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.ordered_categories = builder.data.load(
            data_keys.LBWSG.SEX_SPECIFIC_ORDERED_CATEGORIES
        )

    ##################
    # Public methods #
    ##################

    def single_axis_ppf(
        self,
        axis: str,
        propensity: pd.Series,
        categorical_propensity: pd.Series | None = None,
        categorical_exposure: pd.Series | None = None,
    ) -> pd.Series:
        if (categorical_propensity is None) == (categorical_exposure is None):
            raise ValueError(
                "Exactly one of categorical propensity or categorical exposure "
                "must be provided."
            )

        # get categorical exposure
        if categorical_exposure is None:
            categorical_exposures = [
                self.categorical_sex_specific_ppf(categorical_propensity, sex)
                for sex in ["Male", "Female"]
            ]
            categorical_exposure = pd.concat(categorical_exposures).sort_index()

        # get continuous exposure (copied from LBWSGDistribution)
        exposure_intervals = categorical_exposure.apply(
            lambda category: self.category_intervals[axis][category]
        )
        exposure_left = exposure_intervals.apply(lambda interval: interval.left)
        exposure_right = exposure_intervals.apply(lambda interval: interval.right)
        continuous_exposure = propensity * (exposure_right - exposure_left) + exposure_left
        continuous_exposure = continuous_exposure.rename(f"{axis}.exposure")
        # TODO: uncomment once we allow null gestational age exposure
        # set exposures to null for preterm pregnancies
        # continuous_exposure[
        #     self.population_view.get(propensity.index)[COLUMNS.PREGNANCY_OUTCOME]
        #     == PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME
        # ] = np.nan
        return continuous_exposure

    def categorical_sex_specific_ppf(self, quantiles: pd.Series, sex: str) -> pd.Series:
        """Takes a possibly full set of propensities and returns category exposures for the provided sex."""
        pop = self.population_view.get(
            quantiles.index, [COLUMNS.SEX_OF_CHILD, COLUMNS.PREGNANCY_OUTCOME]
        )
        pop = pop.loc[pop[COLUMNS.SEX_OF_CHILD] == sex]
        exposures = self.get_simulant_specific_probabilities(pop.index)
        # reorder categories according to sex-specific order
        exposures = exposures[self.ordered_categories[sex]]

        if not np.allclose(1, np.sum(exposures, axis=1)):
            raise MissingDataError("Exposure data does not sum to 1.")

        quantiles = quantiles.loc[exposures.index]
        exposure_sum = exposures.cumsum(axis="columns")
        category_index = pd.concat(
            [exposure_sum[c] < quantiles for c in exposure_sum.columns], axis=1
        ).sum(axis=1)
        sex_specific_ordering = self.ordered_categories[sex]
        return pd.Series(
            np.array(sex_specific_ordering)[category_index],
            name=self.risk + ".exposure",
            index=quantiles.index,
        )

    def get_simulant_specific_probabilities(self, index: pd.Index) -> pd.DataFrame:
        """Return the appropriate exposure probabilities given whether a simulant has a
        live birth or stillbirth (abortion/miscarriage/ectopic pregnancies are assigned
        live birth exposures). Stillbirth exposures are the same as live birth exposures
        but with the two LBWSG categories with a max gestational age below 24 weeks
        (cat2 and cat8) dropped."""
        pop = self.population_view.get(
            index, [COLUMNS.SEX_OF_CHILD, COLUMNS.PREGNANCY_OUTCOME]
        )
        exposure = self.population_view.get_frame(index, self.exposure_params_pipeline)
        non_stillbirth_categories = self.get_non_stillbirth_categories()

        is_stillbirth = (
            pop[COLUMNS.PREGNANCY_OUTCOME] == PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME
        )
        stillbirth_exposures = exposure.loc[is_stillbirth].copy()
        stillbirth_exposures.loc[:, non_stillbirth_categories] = 0.0
        stillbirth_exposures = stillbirth_exposures.div(
            stillbirth_exposures.sum(axis=1), axis=0
        )

        is_not_stillbirth = ~is_stillbirth  # include abortion/miscarriage/ectopic pregnancy
        non_stillbirth_exposures = exposure.loc[is_not_stillbirth]

        return pd.concat([non_stillbirth_exposures, stillbirth_exposures]).sort_index()

    ##################
    # Helper methods #
    ##################

    @staticmethod
    def _parse_description(description: str) -> tuple[pd.Interval, pd.Interval]:
        """Parses a string corresponding to a low birth weight and short gestation
        category to an Interval. Sets the minimum gestational age value to 20
        instead of 0.

        An example of a standard description:
        'Neonatal preterm and LBWSG (estimation years) - [0, 24) wks, [0, 500) g'
        An example of an edge case for gestational age:
        'Neonatal preterm and LBWSG (estimation years) - [40, 42+] wks, [2000, 2500) g'
        An example of an edge case of birth weight:
        'Neonatal preterm and LBWSG (estimation years) - [36, 37) wks, [4000, 9999] g'
        """
        lbwsg_values = [float(val) for val in re.findall(r"(\d+)", description)]
        if len(list(lbwsg_values)) != 4:
            raise ValueError(
                f"Could not parse LBWSG description '{description}'. Expected 4 numeric values."
            )
        # update gestational age to never be below 20
        # in practice, all this does is convert cat2 and cat8 to be bounded between 20 and 24 weeks
        # https://vivarium-research.readthedocs.io/en/latest/models/risk_exposures/low_birthweight_short_gestation/gbd_2021/index.html#converting-gbd-s-categorical-exposure-distribution-to-a-continuous-exposure-distribution
        lbwsg_values[0] = max(lbwsg_values[0], 20.0)
        return (
            pd.Interval(*lbwsg_values[:2], closed="left"),  # Gestational Age
            pd.Interval(*lbwsg_values[2:], closed="left"),  # Birth Weight
        )

    def get_non_stillbirth_categories(self) -> list[str]:
        """Get a list of LBWSG categories which we cannot assign to stillbirths
        (those that have a max gestational age of 24)."""
        return [
            cat
            for (cat, interval) in self.category_intervals["gestational_age"].items()
            if interval.right == 24.0
        ]


class LBWSGRisk(LBWSGRisk_):
    exposure_distributions = {"lbwsg": OrderedLBWSGDistribution}

    @property
    def time_step_prepare_priority(self) -> int:
        return 1

    #####################
    # Lifecycle methods #
    #####################

    AXES = [BIRTH_WEIGHT, GESTATIONAL_AGE]

    def __init__(self):
        super().__init__()
        self.continuous_propensity_column_name = {
            axis: f"{self.name}.{axis}.continuous_propensity" for axis in self.AXES
        }
        self.categorical_propensity_name_correlated = f"{self.name}.correlated_propensity"

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self._sim_step_name = builder.time.simulation_event_name()
        # We have to override the age_end due to the wide state table and this is easier than
        # adding an extra population configuration key
        self.configuration_age_end = 0.0

        # The parent registers initializers for categorical_propensity, continuous propensities,
        # and exposure columns. We need an additional initializer for our custom continuous
        # propensity columns (the parent uses different column names).
        builder.population.register_initializer(
            initializer=self._initialize_continuous_propensities,
            columns=[self.continuous_propensity_column_name[axis] for axis in self.AXES],
            required_resources=[self.randomness],
        )

    #################
    # Setup methods #
    #################

    def register_birth_exposure_pipeline(self, builder: Builder) -> None:
        """Override parent to handle partial term pregnancies in the gestational age axis."""
        builder.value.register_attribute_producer(
            self.birth_exposure_pipeline,
            source=self.get_birth_exposure,
            preferred_post_processor=get_exposure_post_processor(builder, self.name),
            required_resources=[
                self.exposure_distribution.exposure_ppf_pipeline,
                self.categorical_propensity_name_correlated,
                COLUMNS.PARTIAL_TERM_PREGNANCY_DURATION,
                *[self.continuous_propensity_column_name[axis] for axis in self.AXES],
            ],
        )

    ########################
    # Event-driven methods #
    ########################

    def initialize_exposure(self, pop_data: SimulantData) -> None:
        """Initialize exposure columns to NaN; values are assigned at ultrasound."""
        exposures = pd.DataFrame(
            {
                self.get_exposure_name(axis): pd.Series(np.nan, index=pop_data.index)
                for axis in self.AXES
            }
        )
        self.population_view.initialize(exposures)

    def _initialize_continuous_propensities(self, pop_data: SimulantData) -> None:
        propensity = {
            self.continuous_propensity_column_name[axis]: self.get_continuous_propensity(
                pop_data, axis
            )
            for axis in self.AXES
        }
        self.population_view.initialize(pd.DataFrame(propensity))

    def get_birth_exposure(self, index: pd.Index) -> pd.DataFrame:
        """Compute birth exposure for both axes.

        For gestational age, use partial term pregnancy duration where available.
        For all other simulants, use the PPF from the LBWSG distribution.
        """
        pop = self.population_view.get(
            index,
            [
                COLUMNS.SEX_OF_CHILD,
                COLUMNS.PARTIAL_TERM_PREGNANCY_DURATION,
                self.continuous_propensity_column_name[BIRTH_WEIGHT],
                self.continuous_propensity_column_name[GESTATIONAL_AGE],
            ],
        )

        partial_term_durations = pop[COLUMNS.PARTIAL_TERM_PREGNANCY_DURATION]
        is_partial_term = partial_term_durations.notna()

        result = {}
        for axis in self.AXES:
            exposure = pd.Series(np.nan, index=index)

            if axis == GESTATIONAL_AGE and is_partial_term.any():
                exposure.loc[is_partial_term] = partial_term_durations.loc[is_partial_term]
                ppf_index = index[~is_partial_term]
            else:
                ppf_index = index

            if not ppf_index.empty:
                categorical_propensity = self.population_view.get(
                    ppf_index, self.categorical_propensity_name_correlated
                )
                continuous_propensity = pop.loc[
                    ppf_index, self.continuous_propensity_column_name[axis]
                ]
                exposure.loc[ppf_index] = self.exposure_distribution.single_axis_ppf(
                    axis, continuous_propensity, categorical_propensity
                )

            result[axis] = exposure

        return pd.DataFrame(result)

    def get_continuous_propensity(self, pop_data: SimulantData, axis: str) -> pd.Series:
        return pd.Series(
            self.randomness.get_draw(pop_data.index, additional_key=axis),
            name=self.continuous_propensity_column_name[axis],
        )

    def on_time_step_prepare(self, event: Event) -> None:
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.ULTRASOUND:
            return

        exposure_columns = [self.get_exposure_name(axis) for axis in self.AXES]

        def _update_exposures(pop: pd.DataFrame) -> pd.DataFrame:
            birth_exposures = self.population_view.get_frame(
                event.index, self.birth_exposure_pipeline
            )
            col_mapping = {axis: self.get_exposure_name(axis) for axis in self.AXES}
            return birth_exposures.rename(columns=col_mapping)

        self.population_view.update(exposure_columns, _update_exposures)


class LBWSGRiskEffect(LBWSGRiskEffect_):
    """Subclass of LBWSGRiskEffect to be compatible with the wide state table, meaning it
    will query on child lookup columns. This also exposes the PAF as a pipeline so it is
    accessible by the neonatal causes component. The ACMR PAF will be used to calculate a
    normalizing constant to modify CSMR pipelines for neonatal causes."""

    @property
    def lbwsg_exposure_column_names(self) -> list[str]:
        return [
            LBWSGRisk_.get_exposure_name(axis) for axis in [BIRTH_WEIGHT, GESTATIONAL_AGE]
        ]

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.paf_pipeline_name = f"lbwsg_paf_on_{self.target.name}.{self.target.measure}.paf"
        # age_intervals must be set before super().setup() since it's used by
        # register_relative_risk_pipeline and initialize_relative_risk
        self.age_intervals = self.get_age_intervals(builder)
        # super().setup() calls build_paf_lookup_table, register_relative_risk_pipeline, etc.
        super().setup(builder)
        # Register a separate PAF pipeline that exposes the PAF for other components
        builder.value.register_attribute_producer(
            self.paf_pipeline_name,
            source=self.paf_table,
            required_resources=[self.paf_table],
        )

    def register_paf_modifier(self, builder: Builder) -> None:
        builder.value.register_attribute_modifier(
            self.target_paf_name,
            modifier=lambda index: self.population_view.get(index, self.paf_pipeline_name),
            required_resources=[self.paf_pipeline_name],
        )

    def get_age_intervals(self, builder: Builder) -> dict[str, pd.Interval]:
        age_bins = builder.data.load("population.age_bins").set_index("age_start")
        # Map to child column. Can't map in artifact since it is used for both mothers and children
        age_bins = age_bins.rename(columns=CHILD_LOOKUP_COLUMN_MAPPER)
        relative_risks = builder.data.load(f"{self.risk}.relative_risk")

        # Filter groups where all 'value' entries are not equal to 1
        filtered_groups = relative_risks.groupby("child_age_start").filter(
            lambda x: (x["value"] != 1).any()
        )
        # Get unique 'age_start' values from the filtered groups
        exposed_age_group_starts = filtered_groups["child_age_start"].unique()

        return {
            to_snake_case(age_bins.loc[age_start, "age_group_name"]): pd.Interval(
                age_start, age_bins.loc[age_start, "child_age_end"]
            )
            for age_start in exposed_age_group_starts
        }

    def register_relative_risk_pipeline(self, builder: Builder) -> None:
        builder.value.register_attribute_producer(
            self.relative_risk_name,
            source=self._relative_risk_source,
            required_resources=[COLUMNS.CHILD_AGE] + self.rr_column_names,
        )

    def get_interpolator(self, builder: Builder) -> pd.Series:
        age_start_to_age_group_name_map = {
            interval.left: to_snake_case(age_group_name)
            for age_group_name, interval in self.age_intervals.items()
        }

        # get relative risk data for target
        interpolators = builder.data.load(f"{self.risk}.relative_risk_interpolator")
        interpolators = (
            # isolate RRs for target and drop non-neonatal age groups since they have RR == 1.0
            interpolators[
                interpolators["child_age_start"].isin(
                    [interval.left for interval in self.age_intervals.values()]
                )
            ]
            .drop(columns=["child_age_end", "year_start", "year_end"])
            .set_index([COLUMNS.SEX_OF_CHILD, "value"])
            .apply(
                lambda row: (age_start_to_age_group_name_map[row["child_age_start"]]), axis=1
            )
            .rename("age_group_name")
            .reset_index()
            .set_index([COLUMNS.SEX_OF_CHILD, "age_group_name"])
        )["value"]

        interpolators = interpolators.apply(lambda x: pickle.loads(bytes.fromhex(x)))
        return interpolators

    def _get_relative_risk(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get(index, self.rr_column_names + [COLUMNS.CHILD_AGE])
        relative_risk = pd.Series(1.0, index=index, name=self.relative_risk_name)

        for age_group, interval in self.age_intervals.items():
            age_group_mask = (interval.left <= pop[COLUMNS.CHILD_AGE]) & (
                pop[COLUMNS.CHILD_AGE] < interval.right
            )
            relative_risk[age_group_mask] = pop.loc[
                age_group_mask, self.get_relative_risk_column_name(age_group)
            ]
        return relative_risk

    ########################
    # Event-driven methods #
    ########################

    def initialize_relative_risk(self, pop_data: SimulantData) -> None:
        pop = self.population_view.get(
            pop_data.index,
            [COLUMNS.SEX_OF_CHILD] + self.lbwsg_exposure_column_names,
        )
        birth_weight = pop[LBWSGRisk_.get_exposure_name(BIRTH_WEIGHT)]
        gestational_age = pop[LBWSGRisk_.get_exposure_name(GESTATIONAL_AGE)]

        is_male = pop[COLUMNS.SEX_OF_CHILD] == "Male"
        is_tmrel = (self.TMREL_GESTATIONAL_AGE_INTERVAL.left <= gestational_age) & (
            self.TMREL_BIRTH_WEIGHT_INTERVAL.left <= birth_weight
        )

        def get_relative_risk_for_age_group(age_group: str) -> pd.Series:
            column_name = self.get_relative_risk_column_name(age_group)
            log_relative_risk = pd.Series(0.0, index=pop_data.index, name=column_name)

            male_interpolator = self.interpolator["Male", age_group]
            log_relative_risk[is_male & ~is_tmrel] = male_interpolator(
                gestational_age[is_male & ~is_tmrel],
                birth_weight[is_male & ~is_tmrel],
                grid=False,
            )
            female_interpolator = self.interpolator["Female", age_group]
            log_relative_risk[~is_male & ~is_tmrel] = female_interpolator(
                gestational_age[~is_male & ~is_tmrel],
                birth_weight[~is_male & ~is_tmrel],
                grid=False,
            )

            return np.exp(log_relative_risk)

        relative_risk_columns = [
            get_relative_risk_for_age_group(age_group) for age_group in self.age_intervals
        ]
        self.population_view.initialize(pd.concat(relative_risk_columns, axis=1))

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.ULTRASOUND:
            return

        pop = self.population_view.get(
            event.index,
            [COLUMNS.SEX_OF_CHILD] + self.lbwsg_exposure_column_names,
        )
        birth_weight = pop[LBWSGRisk_.get_exposure_name(BIRTH_WEIGHT)]
        gestational_age = pop[LBWSGRisk_.get_exposure_name(GESTATIONAL_AGE)]

        is_male = pop[COLUMNS.SEX_OF_CHILD] == "Male"
        is_tmrel = (self.TMREL_GESTATIONAL_AGE_INTERVAL.left <= gestational_age) & (
            self.TMREL_BIRTH_WEIGHT_INTERVAL.left <= birth_weight
        )

        def get_relative_risk_for_age_group(age_group: str) -> pd.Series:
            column_name = self.get_relative_risk_column_name(age_group)
            log_relative_risk = pd.Series(0.0, index=event.index, name=column_name)

            male_interpolator = self.interpolator["Male", age_group]
            log_relative_risk[is_male & ~is_tmrel] = male_interpolator(
                gestational_age[is_male & ~is_tmrel],
                birth_weight[is_male & ~is_tmrel],
                grid=False,
            )
            female_interpolator = self.interpolator["Female", age_group]
            log_relative_risk[~is_male & ~is_tmrel] = female_interpolator(
                gestational_age[~is_male & ~is_tmrel],
                birth_weight[~is_male & ~is_tmrel],
                grid=False,
            )

            return np.exp(log_relative_risk)

        rr_columns = [self.get_relative_risk_column_name(ag) for ag in self.age_intervals]

        def _update_rr(pop: pd.DataFrame) -> pd.DataFrame:
            return pd.concat(
                [get_relative_risk_for_age_group(ag) for ag in self.age_intervals],
                axis=1,
            )

        self.population_view.update(rr_columns, _update_rr)


####################################
# LBWSG PAF Calculation Components #
####################################
class LBWSGPAFRiskEffect(LBWSGRiskEffect):
    def setup(self, builder: Builder) -> None:
        from vivarium_public_health.risks import RiskEffect

        # subclass setup so we don't define sim step name attribute
        self._sim_step_name = None
        self.paf_pipeline_name = f"lbwsg_paf_on_{self.target.name}.{self.target.measure}.paf"
        self.age_intervals = self.get_age_intervals(builder)
        # Call RiskEffect.setup() directly — this builds paf_table, rr_table etc.
        # We skip LBWSGRiskEffect_.setup() because it registers an initializer
        # with required_resources=["sex"] which doesn't exist in the PAF sim.
        RiskEffect.setup(self, builder)
        self.interpolator = self.get_interpolator(builder)
        exposure_columns = [LBWSGRisk_.get_exposure_name(axis) for axis in [BIRTH_WEIGHT, GESTATIONAL_AGE]]
        builder.population.register_initializer(
            initializer=self.initialize_relative_risk,
            columns=self.rr_column_names,
            required_resources=exposure_columns + [COLUMNS.SEX_OF_CHILD],
        )
        # Register a separate PAF pipeline after super has built paf_table
        builder.value.register_attribute_producer(
            self.paf_pipeline_name,
            source=self.paf_table,
            required_resources=[self.paf_table],
        )

    def register_relative_risk_pipeline(self, builder: Builder) -> None:
        """Override to avoid depending on the exposure pipeline."""
        exposure_columns = [LBWSGRisk_.get_exposure_name(axis) for axis in [BIRTH_WEIGHT, GESTATIONAL_AGE]]
        builder.value.register_attribute_producer(
            self.relative_risk_name,
            self._relative_risk_source,
            required_resources=exposure_columns,
        )

    def initialize_relative_risk(self, pop_data: SimulantData) -> None:
        pop = self.population_view.get(
            pop_data.index,
            [COLUMNS.SEX_OF_CHILD] + self.lbwsg_exposure_column_names,
        )
        birth_weight = pop[LBWSGRisk_.get_exposure_name(BIRTH_WEIGHT)]
        gestational_age = pop[LBWSGRisk_.get_exposure_name(GESTATIONAL_AGE)]

        is_male = pop[COLUMNS.SEX_OF_CHILD] == "Male"
        is_tmrel = (self.TMREL_GESTATIONAL_AGE_INTERVAL.left <= gestational_age) & (
            self.TMREL_BIRTH_WEIGHT_INTERVAL.left <= birth_weight
        )

        def get_relative_risk_for_age_group(age_group: str) -> pd.Series:
            column_name = self.get_relative_risk_column_name(age_group)
            log_relative_risk = pd.Series(0.0, index=pop_data.index, name=column_name)

            male_interpolator = self.interpolator["Male", age_group]
            log_relative_risk[is_male & ~is_tmrel] = male_interpolator(
                gestational_age[is_male & ~is_tmrel],
                birth_weight[is_male & ~is_tmrel],
                grid=False,
            )
            female_interpolator = self.interpolator["Female", age_group]
            log_relative_risk[~is_male & ~is_tmrel] = female_interpolator(
                gestational_age[~is_male & ~is_tmrel],
                birth_weight[~is_male & ~is_tmrel],
                grid=False,
            )

            return np.exp(log_relative_risk)

        relative_risk_columns = [
            get_relative_risk_for_age_group(age_group) for age_group in self.age_intervals
        ]
        self.population_view.initialize(pd.concat(relative_risk_columns, axis=1))

    def on_time_step(self, event: Event) -> None:
        pass


class LBWSGPAFCalculationExposure(LBWSGRisk):
    def setup(self, builder: Builder) -> None:
        # Skip the entire Risk/LBWSGRisk_ setup chain because it creates the
        # LBWSGDistribution which registers pipelines requiring categorical_propensity
        # — a column that doesn't exist in the PAF simulation.
        # Instead, we only register the pipelines and initializers that the PAF sim
        # actually needs.
        self._components = builder.components
        self.register_birth_exposure_pipeline(builder)
        self.configuration_age_end = 0.0
        self.lbwsg_categories = builder.data.load(data_keys.LBWSG.CATEGORIES)
        self.age_bins = builder.data.load(data_keys.POPULATION.AGE_BINS)

        # Phase 1: initialize category/bin columns (no pipeline dependency)
        builder.population.register_initializer(
            initializer=self._initialize_categories,
            columns=["lbwsg_category", "age_bin"],
            required_resources=["child_age", "sex_of_child"],
        )
        # Phase 2: initialize exposure columns via the birth exposure pipeline,
        # which reads lbwsg_category and age_bin from the state table.
        exposure_columns = [self.get_exposure_name(axis) for axis in self.AXES]
        builder.population.register_initializer(
            initializer=self._initialize_birth_exposures,
            columns=exposure_columns,
            required_resources=[
                self.birth_exposure_pipeline,
                "lbwsg_category",
                "age_bin",
            ],
        )

    #################
    # Setup methods #
    #################

    def register_birth_exposure_pipeline(self, builder: Builder) -> None:
        """Register a birth exposure pipeline with resources available in the PAF sim."""
        builder.value.register_attribute_producer(
            self.birth_exposure_pipeline,
            source=self.get_birth_exposure,
            preferred_post_processor=get_exposure_post_processor(builder, self.name),
            required_resources=["child_age", "sex_of_child", "lbwsg_category", "age_bin"],
        )

    ########################
    # Event-driven methods #
    ########################

    def _initialize_categories(self, pop_data: SimulantData) -> None:
        """Assign LBWSG categories and age bins to simulants."""
        pop = self.population_view.get(pop_data.index, ["child_age", "sex_of_child"])
        pop["age_bin"] = pd.cut(pop["child_age"], self.age_bins["age_start"])
        pop = pop.sort_values(["sex_of_child", "child_age"])

        num_sexes = pop.sex_of_child.nunique()
        assert num_sexes == 2

        lbwsg_categories = self.lbwsg_categories.keys()
        num_repeats, remainder = divmod(len(pop), num_sexes * len(lbwsg_categories))
        if remainder != 0:
            raise ValueError(
                "Population size should be multiple of the number of LBWSG categories times the number of sexes."
                f"Population size is {len(pop)}, but should be a multiple of "
                f"{num_sexes * len(lbwsg_categories)}."
            )

        assigned_categories = list(lbwsg_categories) * num_sexes * num_repeats
        pop["lbwsg_category"] = assigned_categories

        self.population_view.initialize(pop[["age_bin", "lbwsg_category"]])

    def _initialize_birth_exposures(self, pop_data: SimulantData) -> None:
        """Compute birth exposures from the pipeline and write to state table."""
        birth_exposures = self.population_view.get_frame(
            pop_data.index, self.birth_exposure_pipeline
        )
        col_mapping = {axis: self.get_exposure_name(axis) for axis in self.AXES}
        birth_exposures.rename(columns=col_mapping, inplace=True)
        self.population_view.initialize(birth_exposures)

    def on_time_step_prepare(self, event: Event) -> None:
        """Update the age bins to match the simulants' ages."""
        pop = self.population_view.get(event.index, ["child_age", "sex_of_child"])
        pop["age_bin"] = pd.cut(pop["child_age"], self.age_bins["age_start"])

        self.population_view.update(["age_bin"], lambda _: pop["age_bin"])

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def get_birth_exposure(self, index: pd.Index) -> pd.DataFrame:
        pop = self.population_view.get(index, ["age_bin", "sex_of_child", "lbwsg_category"])
        lbwsg_categories = self.lbwsg_categories.keys()
        num_simulants_in_category = int(len(pop) / (len(lbwsg_categories) * 2))
        num_points_in_interval = int(math.sqrt(num_simulants_in_category))

        result = {axis: pd.Series(np.nan, index=pop.index) for axis in self.AXES}

        for age_bin, sex, cat in itertools.product(
            pop["age_bin"].unique(), ["Male", "Female"], lbwsg_categories
        ):
            description = self.lbwsg_categories[cat]

            birthweight_endpoints = [
                float(val)
                for val in description.split(", [")[1].split(")")[0].split("]")[0].split(", ")
            ]
            birthweight_interval_values = np.linspace(
                birthweight_endpoints[0],
                birthweight_endpoints[1],
                num=num_points_in_interval + 2,
            )[1:-1]

            gestational_age_endpoints = [
                float(val)
                for val in description.split("- [")[1].split(")")[0].split("+")[0].split(", ")
            ]
            gestational_age_interval_values = np.linspace(
                gestational_age_endpoints[0],
                gestational_age_endpoints[1],
                num=num_points_in_interval + 2,
            )[1:-1]

            birthweight_points, gestational_age_points = np.meshgrid(
                birthweight_interval_values, gestational_age_interval_values
            )
            lbwsg_exposures = pd.DataFrame(
                {
                    "birth_weight": birthweight_points.flatten(),
                    "gestational_age": gestational_age_points.flatten(),
                }
            )

            subset_index = pop[
                (pop["lbwsg_category"] == cat)
                & (pop["age_bin"] == age_bin)
                & (pop["sex_of_child"] == sex)
            ].index
            for axis in self.AXES:
                result[axis].loc[subset_index] = lbwsg_exposures[axis].values

        return pd.DataFrame(result)


class LBWSGPAFObserver(Component):
    CONFIGURATION_DEFAULTS = {
        "stratification": {
            "lbwsg_paf": {
                "exclude": [],
                "include": [],
            }
        }
    }

    def __init__(self, target: str):
        super().__init__()
        self.target = TargetString(target)

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.birth_exposure = builder.data.load(data_keys.LBWSG.BIRTH_EXPOSURE)
        self.risk_effect = builder.components.get_component(
            f"risk_effect.low_birth_weight_and_short_gestation_on_{self.target}"
        )
        self.config = builder.configuration.stratification.lbwsg_paf
        self.pop_size = builder.configuration.population.population_size
        self.step_number = 1

        builder.results.register_stratified_observation(
            name=f"calculated_lbwsg_paf_on_{self.target}",
            aggregator=self.calculate_paf,
            results_updater=self.results_updater,
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="time_step",
        )
        # Add observer to get paf for preterm birth population
        builder.results.register_stratified_observation(
            name=f"calculated_lbwsg_paf_on_{self.target}_preterm",
            pop_filter="`gestational_age.exposure` < 37",
            aggregator=self.calculate_paf,
            results_updater=self.results_updater,
            requires_attributes=[COLUMNS.GESTATIONAL_AGE_EXPOSURE],
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="time_step",
        )

    def on_time_step_cleanup(self, event: Event) -> None:
        """Increment step number at the end of each time step."""
        self.step_number += 1

    def results_updater(self, old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        if self.step_number == 1:  # early neonatal time step
            df = new
        else:  # late neonatal time step
            # use PAFs from first time step for ENN and second time step for LNN
            df = pd.concat(
                [
                    old.query("child_age_group=='early_neonatal'"),
                    new.query("child_age_group=='late_neonatal'"),
                ],
            )
        return df

    def calculate_paf(self, x: pd.DataFrame) -> float:
        relative_risk = self.risk_effect.adjust_target(x.index, pd.Series(1, index=x.index))
        relative_risk.name = "relative_risk"
        lbwsg_category = self.population_view.get(x.index, "lbwsg_category")
        unique_sexes = x["sex_of_child"].unique()
        if len(unique_sexes) != 1:
            raise ValueError(
                "Stratified data contains more than one sex, but this observer (LBWSGPAFObserver) needs sex-stratified data."
            )
        sex = unique_sexes[0]

        # Use exposure prevalence at birth
        lbwsg_prevalence = self.birth_exposure.rename(
            {"parameter": "lbwsg_category", "value": "prevalence"}, axis=1
        )
        lbwsg_prevalence = lbwsg_prevalence.loc[lbwsg_prevalence["sex_of_child"] == sex]

        # weight LBWSG prevalence by fraction of simulants who survived to late neonatal period
        # within a given LBWSG category
        # this fraction will be 1 at the first time step because no one has died yet, which is
        # what we want
        weights = calculate_mortality_weights(self, sex)
        lbwsg_prevalence = lbwsg_prevalence.merge(weights)
        lbwsg_prevalence["prevalence"] = (
            lbwsg_prevalence["prevalence"] * lbwsg_prevalence["proportion_alive"]
        )
        lbwsg_prevalence = lbwsg_prevalence.drop(columns=["proportion_alive"])

        mean_rrs = (
            pd.concat([lbwsg_category, relative_risk], axis=1)
            .groupby("lbwsg_category", as_index=False)
            .mean()
        )
        mean_rrs = mean_rrs.merge(lbwsg_prevalence, on="lbwsg_category")

        mean_rr = np.average(mean_rrs["relative_risk"], weights=mean_rrs["prevalence"])
        paf = (mean_rr - 1) / mean_rr

        return paf


class PretermPrevalenceObserver(Component):
    CONFIGURATION_DEFAULTS = {
        "stratification": {
            "preterm_prevalence": {
                "exclude": [],
                "include": [],
            }
        }
    }

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.birth_exposure = builder.data.load(data_keys.LBWSG.BIRTH_EXPOSURE)
        self.lbwsg_categories = builder.data.load(data_keys.LBWSG.CATEGORIES)
        self.config = builder.configuration.stratification.preterm_prevalence
        self.pop_size = builder.configuration.population.population_size
        self.step_number = 1

        builder.results.register_stratified_observation(
            name=f"calculated_late_neonatal_preterm_prevalence",
            aggregator=self.calculate_preterm_prevalence,
            results_updater=self.results_updater,
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="time_step",
            to_observe=self.to_observe,
        )

    def on_time_step_cleanup(self, event: Event) -> None:
        """Increment step number at the end of each time step."""
        self.step_number += 1

    def calculate_preterm_prevalence(self, x: pd.DataFrame) -> float:
        # weight preterm prevalence by fraction of simulants who survived to late neonatal period
        # within a given LBWSG category
        # this fraction will be 1 at the first time step because no one has died yet, which is
        # what we want
        unique_sexes = x["sex_of_child"].unique()
        if len(unique_sexes) != 1:
            raise ValueError(
                "Stratified data contains more than one sex, but this observer (PretermPrevalenceObserver) needs sex-stratified data."
            )
        sex = unique_sexes[0]

        # Use exposure prevalence at birth
        lbwsg_prevalence = self.birth_exposure.rename(
            {"parameter": "lbwsg_category", "value": "prevalence"}, axis=1
        )
        lbwsg_prevalence = lbwsg_prevalence.loc[lbwsg_prevalence["sex_of_child"] == sex]

        weights = calculate_mortality_weights(self, sex)
        lbwsg_prevalence = lbwsg_prevalence.merge(weights)
        lbwsg_prevalence["mortality_weighted_prevalence"] = (
            lbwsg_prevalence["prevalence"] * lbwsg_prevalence["proportion_alive"]
        )

        # Get preterm categories
        preterm_cats = []
        for cat, description in self.lbwsg_categories.items():
            i = parse_short_gestation_description(description)
            if i.right <= metadata.PRETERM_AGE_CUTOFF:
                preterm_cats.append(cat)

        preterm_prevalences = lbwsg_prevalence[
            lbwsg_prevalence["lbwsg_category"].isin(preterm_cats)
        ]
        preterm_prevalence = (
            preterm_prevalences["mortality_weighted_prevalence"].sum()
            / lbwsg_prevalence["mortality_weighted_prevalence"].sum()
        )
        return preterm_prevalence

    def results_updater(self, old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        return new

    def to_observe(self, event: Event) -> pd.DataFrame:
        """Only observe the late neonatal time step."""
        if self.step_number == 2:
            return True
        else:
            return False


class LBWSGMortality(Component):
    """A component to handle neonatal mortality."""

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            self.name: {
                "data_sources": {
                    "all_cause_mortality_risk": self.load_all_causes_mortality_data,
                    "life_expectancy": self.load_life_expectancy_data,
                }
            }
        }

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.randomness = builder.randomness.get_stream(self.name, self)
        self.causes_of_death = ["other_causes"]

        self.all_cause_mortality_risk = self.build_lookup_table(
            builder, "all_cause_mortality_risk", value_columns="value"
        )
        self.life_expectancy = self.build_lookup_table(
            builder, "life_expectancy", value_columns="value"
        )

        self.acmr_paf_pipeline_name = PIPELINES.ACMR_PAF
        # Register pipelines
        self.get_acmr_paf_pipeline(builder)

        builder.value.register_attribute_producer(
            PIPELINES.ACMR,
            source=self.get_acmr_pipeline,
            required_resources=[self.all_cause_mortality_risk]
            + [self.acmr_paf_pipeline_name],
        )

        builder.population.register_initializer(
            self.initialize_mortality_columns,
            columns=[COLUMNS.CHILD_ALIVE, COLUMNS.CHILD_CAUSE_OF_DEATH, COLUMNS.CHILD_YEARS_OF_LIFE_LOST],
        )

    def initialize_mortality_columns(self, pop_data: SimulantData) -> None:
        pop_update = pd.DataFrame(
            {
                COLUMNS.CHILD_ALIVE: True,
                COLUMNS.CHILD_CAUSE_OF_DEATH: "not_dead",
                COLUMNS.CHILD_YEARS_OF_LIFE_LOST: 0.0,
            },
            index=pop_data.index,
        )
        self.population_view.initialize(pop_update)

    def on_time_step(self, event: Event) -> None:
        child_alive = self.population_view.get(event.index, COLUMNS.CHILD_ALIVE)
        alive_idx = child_alive.index[child_alive]
        acmr = self.population_view.get(alive_idx, PIPELINES.ACMR)

        # Determine which neonates die and update metadata
        dead_idx = self.randomness.filter_for_probability(
            alive_idx,
            acmr,
            "lbwsg_mortality",
        )
        if not dead_idx.empty:
            cols_to_update = [
                COLUMNS.CHILD_ALIVE,
                COLUMNS.CHILD_CAUSE_OF_DEATH,
                COLUMNS.CHILD_YEARS_OF_LIFE_LOST,
            ]

            def _update_mortality(pop: pd.DataFrame) -> pd.DataFrame:
                pop.loc[dead_idx, COLUMNS.CHILD_ALIVE] = False
                pop.loc[dead_idx, COLUMNS.CHILD_CAUSE_OF_DEATH] = "other_causes"
                pop.loc[dead_idx, COLUMNS.CHILD_YEARS_OF_LIFE_LOST] = self.life_expectancy(
                    dead_idx
                )
                return pop

            self.population_view.update(cols_to_update, _update_mortality)

    ##################
    # Helper methods #
    ##################

    def load_all_causes_mortality_data(self, builder: Builder) -> pd.DataFrame:
        """Load all-cause mortality rate data."""
        acmrisk = builder.data.load(data_keys.POPULATION.ALL_CAUSES_MORTALITY_RISK)
        child_acmrisk = acmrisk.rename(columns=CHILD_LOOKUP_COLUMN_MAPPER)
        return child_acmrisk

    def load_life_expectancy_data(self, builder: Builder) -> pd.DataFrame:
        """Load life expectancy data."""
        life_expectancy = builder.data.load(
            "population.theoretical_minimum_risk_life_expectancy"
        )
        # This needs to remain here since it gets used for both maternal and neonatal mortality
        child_life_expectancy = life_expectancy.rename(columns=CHILD_LOOKUP_COLUMN_MAPPER)
        return child_life_expectancy

    def get_acmr_pipeline(self, index: pd.Index) -> pd.Series:
        # NOTE: This will be modified by the LBWSGRiskEffect
        acmr = self.all_cause_mortality_risk(index)
        paf = self.population_view.get(index, self.acmr_paf_pipeline_name)
        return acmr * (1 - paf)

    def get_acmr_paf_pipeline(self, builder: Builder) -> None:
        acmr_paf = builder.lookup.build_table(0)
        builder.value.register_attribute_producer(
            PIPELINES.ACMR_PAF,
            source=lambda index: acmr_paf(index),
        )


######################
## Utility functions #
######################


def parse_short_gestation_description(description: str) -> pd.Interval:
    # descriptions look like this: 'Birth prevalence - [34, 36) wks, [2000, 2500) g'
    endpoints = pd.Interval(
        *[
            float(val)
            for val in description.split("- [")[1].split(")")[0].split("+")[0].split(", ")
        ],
        closed="left",
    )
    return endpoints


def calculate_mortality_weights(component: Component, sex: str) -> pd.Series:
    """Calculate percentage of simulants alive within a LBWSG category for a given sex."""
    full_index = pd.Index(range(component.pop_size))
    pop_data = component.population_view.get(
        full_index, ["lbwsg_category", "child_alive", "sex_of_child"]
    )
    pop_data = pop_data.loc[pop_data["sex_of_child"] == sex]
    weights = (
        pop_data.groupby(["lbwsg_category", "sex_of_child"])["child_alive"]
        .agg(proportion_alive=lambda x: x.mean())
        .reset_index()
    )
    return weights
