import itertools
import math
import pickle
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from vivarium.component import Component
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import SimulantData
from vivarium.framework.resource import Resource
from vivarium.framework.values import Pipeline
from vivarium_public_health.risks.data_transformations import (
    get_exposure_post_processor,
)
from vivarium_public_health.risks.implementations.low_birth_weight_and_short_gestation import (
    LBWSGRisk
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
)

CATEGORICAL = "categorical"
BIRTH_WEIGHT = "birth_weight"
GESTATIONAL_AGE = "gestational_age"


class LBWSGRiskEffect(LBWSGRiskEffect_):
    """Subclass of LBWSGRiskEffect to be compatible with the wide state table, meaning it
    will query on child lookup columns. This also exposes the PAF as a pipeline so it is
    accessible by the neonatal causes component. The ACMR PAF will be used to calculate a
    normalizing constant to modify CSMR pipelines for neonatal causes."""

    # @property
    # def configuration_defaults(self) -> dict[str, Any]:
    #     """Default values for any configurations managed by this component."""
    #     return {
    #         self.name: {
    #             "data_sources": {
    #                 **{
    #                     key: partial(
    #                         self.load_child_data_from_artifact, data_key=key
    #                     )
    #                     for key in ["relative_risk", "population_attributable_fraction"]
    #                 },
    #             },
    #             "data_source_parameters": {
    #                 "relative_risk": {},
    #             },
    #         }
    #     }

    @property
    def columns_required(self) -> list[str] | None:
        return [COLUMNS.CHILD_AGE, COLUMNS.SEX_OF_CHILD] + self.lbwsg_exposure_column_names

    @property
    def initialization_requirements(self) -> dict[str, list[str]]:
        return {
            "requires_columns": [COLUMNS.SEX_OF_CHILD] + self.lbwsg_exposure_column_names,
            "requires_values": [],
            "requires_streams": [],
        }

    def setup(self, builder: Builder) -> None:
        # Paf pipeline needs to be registered before the super setup is called
        self.acmr_paf = builder.value.register_value_producer(
            f"lbwsg_paf_on_{self.target.name}.{self.target.measure}",
            source=self.lookup_tables["population_attributable_fraction"],
            component=self,
            required_resources=get_lookup_columns(
                [self.lookup_tables["population_attributable_fraction"]]
            ),
        )
        super().setup(builder)
        breakpoint()

    def get_population_attributable_fraction_source(
        self, builder: Builder
    ) -> tuple[pd.DataFrame, list[str]]:
        paf_key = f"{self.risk}.population_attributable_fraction"
        paf_data = builder.data.load(paf_key)
        # Map to child columns
        paf_data = paf_data.rename(columns=CHILD_LOOKUP_COLUMN_MAPPER)
        return paf_data, builder.data.value_columns()(paf_key)

    def get_age_intervals(self, builder: Builder) -> dict[str, pd.Interval]:
        age_bins = builder.data.load("population.age_bins").set_index("age_start")
        relative_risks = builder.data.load(f"{self.risk}.relative_risk")
        # Map to child columns
        age_bins = age_bins.rename(columns=CHILD_LOOKUP_COLUMN_MAPPER)
        relative_risks = relative_risks.rename(columns=CHILD_LOOKUP_COLUMN_MAPPER)

        # Filter groups where all 'value' entries are not equal to 1
        filtered_groups = relative_risks.groupby('child_age_start').filter(lambda x: (x['value'] != 1).any())
        # Get unique 'age_start' values from the filtered groups
        exposed_age_group_starts = filtered_groups['child_age_start'].unique()
        
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
        # Map to child columns
        interpolators = interpolators.rename(columns=CHILD_LOOKUP_COLUMN_MAPPER)
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

    def load_child_data_from_artifact(self, builder: Builder, data_key: str) -> pd.DataFrame:
        data = builder.data.load(data_key)
        data = data.rename(columns=CHILD_LOOKUP_COLUMN_MAPPER)
        return data

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop = self.population_view.subview(
            [COLUMNS.SEX_OF_CHILD] + self.lbwsg_exposure_column_names
        ).get(pop_data.index)
        birth_weight = pop[LBWSGRisk.get_exposure_column_name(BIRTH_WEIGHT)]
        gestational_age = pop[LBWSGRisk.get_exposure_column_name(GESTATIONAL_AGE)]

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


class LBWSGPAFCalculationRiskEffect(LBWSGRiskEffect_):
    """Risk effect component for calculating PAFs for LBWSG. This is only used in a
    separate simulation to calculate the PAFs for the LBWSG risk effect."""

    def get_population_attributable_fraction_source(self, builder: Builder) -> LookupTable:
        return 0, []


class LBWSGPAFCalculationExposure(LBWSGRisk):
    @property
    def columns_required(self) -> list[str] | None:
        return ["age", "sex"]

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
                requires_columns=["age", "sex"],
                preferred_post_processor=get_exposure_post_processor(builder, self.risk),
            )

        return {
            self.birth_exposure_pipeline_name(axis): get_pipeline(axis) for axis in self.AXES
        }

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop = self.population_view.subview(["age", "sex"]).get(pop_data.index)
        pop["age_bin"] = pd.cut(pop["age"], self.age_bins["age_start"])
        pop = pop.sort_values(["sex", "age"])

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

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def get_birth_exposure(self, axis: str, index: pd.Index) -> pd.DataFrame:
        pop = self.population_view.subview(["age_bin", "sex", "lbwsg_category"]).get(index)
        lbwsg_categories = self.lbwsg_categories.keys()
        num_simulants_in_category = int(len(pop) / (len(lbwsg_categories) * 4))
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
                & (pop["sex"] == sex)
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
        return ["lbwsg_category"]

    def __init__(self, target: str):
        super().__init__()
        self.target = TargetString(target)

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.lbwsg_exposure = builder.data.load(data_keys.LBWSG.EXPOSURE)
        self.risk_effect = builder.components.get_component(
            f"risk_effect.low_birth_weight_and_short_gestation_on_{self.target}"
        )
        self.config = builder.configuration.stratification.lbwsg_paf

        builder.results.register_adding_observation(
            name=f"calculated_lbwsg_paf_on_{self.target}",
            pop_filter='alive == "alive"',
            aggregator=self.calculate_paf,
            requires_columns=["alive"],
            additional_stratifications=self.config.include,
            excluded_stratifications=self.config.exclude,
            when="time_step__prepare",
        )

    def calculate_paf(self, x: pd.DataFrame) -> float:
        relative_risk = self.risk_effect.adjust_target(x.index, pd.Series(1, index=x.index))
        relative_risk.name = "relative_risk"
        lbwsg_category = self.population_view.get(x.index)["lbwsg_category"]
        lbwsg_prevalence = self.lbwsg_exposure.rename(
            {"parameter": "lbwsg_category", "value": "prevalence"}, axis=1
        )
        lbwsg_prevalence = lbwsg_prevalence.groupby("lbwsg_category", as_index=False)[
            "prevalence"
        ].sum()

        mean_rrs = (
            pd.concat([lbwsg_category, relative_risk], axis=1)
            .groupby("lbwsg_category", as_index=False)
            .mean()
        )
        mean_rrs = mean_rrs.merge(lbwsg_prevalence, on="lbwsg_category")

        mean_rr = np.average(mean_rrs["relative_risk"], weights=mean_rrs["prevalence"])
        paf = (mean_rr - 1) / mean_rr

        return paf
