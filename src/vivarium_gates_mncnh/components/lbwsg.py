import itertools
import math

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
    LBWSGRisk,
)
from vivarium_public_health.risks.implementations.low_birth_weight_and_short_gestation import (
    LBWSGRiskEffect as LBWSGRiskEffect_,
)
from vivarium_public_health.utilities import TargetString, get_lookup_columns

from vivarium_gates_mncnh.constants import data_keys


class LBWSGRiskEffect(LBWSGRiskEffect_):
    """Subclass of LBWSGRiskEffect to expose the PAF pipeline to be accessable by other components."""

    def setup(self, builder: Builder) -> None:
        # Paf pipeline needs to be registered before the super setup is called
        self.paf = builder.value.register_value_producer(
            f"lbwsg_paf_on_{self.target.name}.{self.target.measure}",
            source=self.lookup_tables["population_attributable_fraction"],
            component=self,
            required_resources=get_lookup_columns(
                [self.lookup_tables["population_attributable_fraction"]]
            ),
        )
        super().setup(builder)

    # NOTE: We will be manually handling the paf effect so the target_paf_pipeline
    # has not been created and will throw a warning
    def register_paf_modifier(self, builder: Builder) -> None:
        builder.value.register_value_modifier(
            self.target_paf_pipeline_name,
            modifier=self.paf,
            component=self,
        )


class LBWSGPAFCalculationRiskEffect(LBWSGRiskEffect_):
    """Risk effect component for calculating PAFs for LBWSG."""

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
        relative_risk = self.risk_effect.target_modifier(x.index, pd.Series(1, index=x.index))
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
