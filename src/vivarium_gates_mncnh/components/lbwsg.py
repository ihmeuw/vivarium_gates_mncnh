from __future__ import annotations

import itertools
import math
import pickle
from typing import Any

import numpy as np
import pandas as pd
from layered_config_tree import ConfigurationError
from vivarium.component import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import SimulantData
from vivarium.framework.resource import Resource
from vivarium.framework.values import Pipeline, list_combiner, union_post_processor
from vivarium_public_health.risks.data_transformations import (
    get_exposure_post_processor,
)
from vivarium_public_health.risks.implementations.low_birth_weight_and_short_gestation import (
    LBWSGRisk as LBWSGRisk_,
)
from vivarium_public_health.risks.implementations.low_birth_weight_and_short_gestation import (
    LBWSGRiskEffect as LBWSGRiskEffect_,
)
from vivarium_public_health.utilities import (
    TargetString,
    get_lookup_columns,
    to_snake_case,
)

from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.constants.data_values import (
    CHILD_LOOKUP_COLUMN_MAPPER,
    COLUMNS,
    PIPELINES,
)

CATEGORICAL = "categorical"
BIRTH_WEIGHT = "birth_weight"
GESTATIONAL_AGE = "gestational_age"


class LBWSGRisk(LBWSGRisk_):
    @property
    def columns_required(self) -> list[str]:
        return [
            COLUMNS.SEX_OF_CHILD,
        ]

    @property
    def initialization_requirements(self) -> list[str | Resource]:
        return [
            COLUMNS.SEX_OF_CHILD,
        ]

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        # We have to override the age_end due to the wide state table and this is easier than
        # adding an extra population configuratio key
        self.configuration_age_end = 0.0


class LBWSGRiskEffect(LBWSGRiskEffect_):
    """Subclass of LBWSGRiskEffect to be compatible with the wide state table, meaning it
    will query on child lookup columns. This also exposes the PAF as a pipeline so it is
    accessible by the neonatal causes component. The ACMR PAF will be used to calculate a
    normalizing constant to modify CSMR pipelines for neonatal causes."""

    @property
    def columns_required(self) -> list[str] | None:
        return [COLUMNS.CHILD_AGE, COLUMNS.SEX_OF_CHILD] + self.lbwsg_exposure_column_names

    @property
    def initialization_requirements(self) -> list[str | Resource]:
        return [COLUMNS.SEX_OF_CHILD] + self.lbwsg_exposure_column_names

    def setup(self, builder: Builder) -> None:
        # Paf pipeline needs to be registered before the super setup is called
        self.paf = builder.value.register_value_producer(
            f"lbwsg_paf_on_{self.target.name}.{self.target.measure}.paf",
            source=self.lookup_tables["population_attributable_fraction"],
            component=self,
            required_resources=get_lookup_columns(
                [self.lookup_tables["population_attributable_fraction"]]
            ),
        )
        super().setup(builder)

    def register_paf_modifier(self, builder):
        required_columns = get_lookup_columns(
            [self.lookup_tables["population_attributable_fraction"]]
        )
        builder.value.register_value_modifier(
            self.target_paf_pipeline_name,
            modifier=self.paf,
            component=self,
            required_resources=required_columns,
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

    def get_relative_risk_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.relative_risk_pipeline_name,
            source=self._relative_risk_source,
            component=self,
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
        pop = self.population_view.get(index)
        relative_risk = pd.Series(1.0, index=index, name=self.relative_risk_pipeline_name)

        for age_group, interval in self.age_intervals.items():
            age_group_mask = (interval.left <= pop[COLUMNS.CHILD_AGE]) & (
                pop[COLUMNS.CHILD_AGE] < interval.right
            )
            relative_risk[age_group_mask] = pop.loc[
                age_group_mask, self.relative_risk_column_name(age_group)
            ]
        return relative_risk

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop = self.population_view.subview(
            [COLUMNS.SEX_OF_CHILD] + self.lbwsg_exposure_column_names
        ).get(pop_data.index)
        birth_weight = pop[LBWSGRisk_.get_exposure_column_name(BIRTH_WEIGHT)]
        gestational_age = pop[LBWSGRisk_.get_exposure_column_name(GESTATIONAL_AGE)]

        is_male = pop[COLUMNS.SEX_OF_CHILD] == "Male"
        is_tmrel = (self.TMREL_GESTATIONAL_AGE_INTERVAL.left <= gestational_age) & (
            self.TMREL_BIRTH_WEIGHT_INTERVAL.left <= birth_weight
        )

        def get_relative_risk_for_age_group(age_group: str) -> pd.Series:
            column_name = self.relative_risk_column_name(age_group)
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
        self.population_view.update(pd.concat(relative_risk_columns, axis=1))


####################################
# LBWSG PAF Calculation Components #
####################################


class LBWSGPAFCalculationRiskEffect(LBWSGRiskEffect):
    """Risk effect component for calculating PAFs for LBWSG. This is only used in a
    separate simulation to calculate the PAFs for the LBWSG risk effect."""

    def get_population_attributable_fraction_source(self, builder: Builder) -> LookupTable:
        return 0, []


class LBWSGPAFCalculationExposure(LBWSGRisk):
    @property
    def columns_required(self) -> list[str] | None:
        return ["child_age", "sex_of_child"]

    @property
    def columns_created(self) -> list[str]:
        return [self.get_exposure_column_name(axis) for axis in self.AXES] + [
            "lbwsg_category",
            "age_bin",
        ]

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.lbwsg_categories = builder.data.load(data_keys.LBWSG.CATEGORIES)
        self.age_bins = builder.data.load(data_keys.POPULATION.AGE_BINS)

    def get_birth_exposure_pipelines(self, builder: Builder) -> dict[str, Pipeline]:
        def get_pipeline(axis_: str):
            return builder.value.register_value_producer(
                self.birth_exposure_pipeline_name(axis_),
                source=lambda index: self.get_birth_exposure(axis_, index),
                requires_columns=["child_age", "sex_of_child"],
                preferred_post_processor=get_exposure_post_processor(builder, self.risk),
            )

        return {
            self.birth_exposure_pipeline_name(axis): get_pipeline(axis) for axis in self.AXES
        }

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop = self.population_view.subview(["child_age", "sex_of_child"]).get(pop_data.index)
        pop["age_bin"] = pd.cut(pop["child_age"], self.age_bins["age_start"])
        pop = pop.sort_values(["sex_of_child", "child_age"])

        lbwsg_categories = self.lbwsg_categories.keys()
        num_repeats, remainder = divmod(len(pop), 2 * len(lbwsg_categories))
        if remainder != 0:
            raise ValueError(
                "Population size should be multiple of double the number of LBWSG categories."
                f"Population size is {len(pop)}, but should be a multiple of "
                f"{2*len(lbwsg_categories)}."
            )

        assigned_categories = list(lbwsg_categories) * (2 * num_repeats)
        pop["lbwsg_category"] = assigned_categories
        self.population_view.update(pop[["age_bin", "lbwsg_category"]])

        birth_exposures = {
            self.get_exposure_column_name(axis): self.birth_exposures[
                self.birth_exposure_pipeline_name(axis)
            ](pop_data.index)
            for axis in self.AXES
        }
        self.population_view.update(pd.DataFrame(birth_exposures))

    def on_time_step_prepare(self, event: Event) -> None:
        """Update the age bins to match the simulants' ages."""
        pop = self.population_view.subview(["child_age", "sex_of_child"]).get(event.index)
        pop["age_bin"] = pd.cut(pop["child_age"], self.age_bins["age_start"])
        self.population_view.update(pop["age_bin"])

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def get_birth_exposure(self, axis: str, index: pd.Index) -> pd.DataFrame:
        pop = self.population_view.subview(["age_bin", "sex_of_child", "lbwsg_category"]).get(
            index
        )
        lbwsg_categories = self.lbwsg_categories.keys()
        num_simulants_in_category = int(len(pop) / (len(lbwsg_categories) * 2))
        num_points_in_interval = int(math.sqrt(num_simulants_in_category))

        exposure_values = pd.Series(name=axis, index=pop.index, dtype=float)

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
            exposure_values.loc[subset_index] = lbwsg_exposures[axis].values

        return exposure_values


class LBWSGPAFObserver(Component):
    CONFIGURATION_DEFAULTS = {
        "stratification": {
            "lbwsg_paf": {
                "exclude": [],
                "include": [],
            }
        }
    }

    @property
    def columns_required(self) -> list[str] | None:
        return ["lbwsg_category", "gestational_age_exposure", "age_bin", "child_alive", "sex_of_child"]

    def __init__(self, target: str):
        super().__init__()
        self.target = TargetString(target)

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.lbwsg_exposure = self._load_custom_paf_exposure(builder)
        self.risk_effect = builder.components.get_component(
            f"risk_effect.low_birth_weight_and_short_gestation_on_{self.target}"
        )
        self.config = builder.configuration.stratification.lbwsg_paf
        self.pop_size = builder.configuration.population.population_size

        self.mortality_weights = builder.value.register_value_producer(
            f"mortality_weights",
            source=self.calculate_mortality_weights,
            component=self,
            required_resources=["lbwsg_category", "child_alive"],
        )

        builder.results.register_adding_observation(
            name=f"calculated_lbwsg_paf_on_{self.target}",
            aggregator=self.calculate_paf,
            requires_columns=["child_alive"],
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="time_step",
        )
        # Add observer to get paf for preterm birth population
        builder.results.register_adding_observation(
            name=f"calculated_lbwsg_paf_on_{self.target}_preterm",
            pop_filter='gestational_age_exposure < 37',
            aggregator=self.calculate_paf,
            requires_columns=["child_alive", "gestational_age_exposure"],
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="time_step",
        )

    def calculate_paf(self, x: pd.DataFrame) -> float:
        relative_risk = self.risk_effect.adjust_target(x.index, pd.Series(1, index=x.index))
        relative_risk.name = "relative_risk"
        lbwsg_category = self.population_view.get(x.index)["lbwsg_category"]
        age_start = self.population_view.get(x.index).age_bin.min().left
        sex = x["sex_of_child"].unique()[0]
        lbwsg_prevalence = self.lbwsg_exposure.rename(
            {"parameter": "lbwsg_category", "value": "prevalence"}, axis=1
        )

        # Subset to age group for exposure - have to use np.isclose because age_start is rounded
        lbwsg_prevalence = lbwsg_prevalence[
            np.isclose(self.lbwsg_exposure.child_age_start, age_start, atol=0.001)
        ]
        lbwsg_prevalence = lbwsg_prevalence.loc[lbwsg_prevalence["sex_of_child"] == sex]

        # weight LBWSG prevalence by fraction of simulants who survived to late neonatal period
        # within a given LBWSG category
        # this fraction will be 1 at the first time step because no one has died yet, which is
        # what we want
        weights = self.mortality_weights(x.index)
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

    def _load_custom_paf_exposure(self, builder: Builder) -> pd.DataFrame:
        """Loads custom exposure data for the PAF simulation. This swaps the birth
        exposure with the early neonatal exposure for the early neonatal age group.
        TODO: potentially switch ENN for LNN exposure depending on how this validates.
        """
        birth_exposure = builder.data.load(data_keys.LBWSG.BIRTH_EXPOSURE)
        neonatal_exposure = builder.data.load(data_keys.LBWSG.EXPOSURE)
        col_order = neonatal_exposure.columns.tolist()
        enn_exposure = neonatal_exposure.loc[neonatal_exposure["child_age_end"] < 8 / 365.0]
        enn_exposure = enn_exposure.drop(columns=["value"])
        new_enn_exposure = pd.merge(
            birth_exposure,
            enn_exposure,
            on=["sex_of_child", "year_start", "year_end", "parameter"],
        )
        exposure = neonatal_exposure.loc[neonatal_exposure["child_age_start"] > 6 / 365.0]

        final_exposure = pd.concat([new_enn_exposure, exposure])
        return final_exposure[col_order]

    def calculate_mortality_weights(self, index: pd.Index) -> pd.Series:
        """Calculate percentage of simulants alive within a LBWSG category."""
        full_index = pd.Index(range(self.pop_size))
        # sex is guaranteed to be the same for all simulants in the index because
        # we stratify by sex of child
        sex_of_subset = self.population_view.get(index)['sex_of_child'].unique()[0]
        pop_data = self.population_view.get(full_index)[["lbwsg_category", "child_alive", "sex_of_child"]]
        pop_data = pop_data[pop_data['sex_of_child'] == sex_of_subset].drop('sex_of_child', axis=1)
        weights = (
            pop_data.groupby("lbwsg_category")["child_alive"]
            .agg(proportion_alive=lambda x: (x == "alive").mean())
            .reset_index()
        )
        return weights


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

    @property
    def columns_created(self) -> list[str]:
        return [COLUMNS.CHILD_CAUSE_OF_DEATH, COLUMNS.CHILD_YEARS_OF_LIFE_LOST]

    @property
    def columns_required(self) -> list[str]:
        return [
            COLUMNS.CHILD_AGE,
            COLUMNS.CHILD_ALIVE,
        ]

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.randomness = builder.randomness.get_stream(self.name, self)
        self.causes_of_death = ["other_causes"]

        # Register pipelines
        self.acmr_paf = self.get_acmr_paf_pipeline(builder)

        self.all_cause_mortality_risk = builder.value.register_value_producer(
            PIPELINES.ACMR,
            source=self.get_acmr_pipeline,
            component=self,
            required_resources=get_lookup_columns(
                [self.lookup_tables["all_cause_mortality_risk"]]
            )
            + [self.acmr_paf],
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop_update = pd.DataFrame(
            {
                COLUMNS.CHILD_CAUSE_OF_DEATH: "not_dead",
                COLUMNS.CHILD_YEARS_OF_LIFE_LOST: 0.0,
            },
            index=pop_data.index,
        )
        self.population_view.update(pop_update)

    def on_time_step(self, event: Event) -> None:
        pop = self.population_view.get(event.index)
        alive_idx = pop.index[pop[COLUMNS.CHILD_ALIVE] == "alive"]
        mortality_risk = self.all_cause_mortality_risk(alive_idx)

        # Determine which neonates die and update metadata
        dead_idx = self.randomness.filter_for_probability(
            alive_idx,
            mortality_risk,
            "lbwsg_mortality",
        )
        if not dead_idx.empty:
            pop.loc[dead_idx, COLUMNS.CHILD_ALIVE] = "dead"
            pop.loc[dead_idx, COLUMNS.CHILD_CAUSE_OF_DEATH] = "other_causes"
            pop.loc[dead_idx, COLUMNS.CHILD_YEARS_OF_LIFE_LOST] = self.lookup_tables[
                "life_expectancy"
            ](dead_idx)

        self.population_view.update(pop)

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

    def get_acmr_pipeline(self, index: pd.Index) -> Pipeline:
        # NOTE: This will be modified by the LBWSGRiskEffect
        acmr = self.lookup_tables["all_cause_mortality_risk"](index)
        paf = self.acmr_paf(index)
        return acmr * (1 - paf)

    def get_acmr_paf_pipeline(self, builder: Builder) -> Pipeline:
        acmr_paf = builder.lookup.build_table(0)
        return builder.value.register_value_producer(
            PIPELINES.ACMR_PAF,
            source=lambda index: [acmr_paf(index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )
