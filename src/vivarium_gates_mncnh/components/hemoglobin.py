from typing import Any

import numpy as np
import pandas as pd
import scipy
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import SimulantData
from vivarium.framework.resource import Resource
from vivarium_public_health.utilities import get_lookup_columns

from vivarium_gates_mncnh.constants.data_keys import HEMOGLOBIN
from vivarium_gates_mncnh.constants.data_values import COLUMNS, HEMOGLOBIN_X_MAX


class Hemoglobin(Component):
    """Component for managing hemoglobin exposure."""

    @property
    def configuration_defaults(self) -> dict[str, dict[str, Any]]:
        return {
            self.name: {
                "data_sources": {
                    "exposure_mean": HEMOGLOBIN.EXPOSURE_MEAN,
                    "exposure_sd": HEMOGLOBIN.EXPOSURE_SD,
                }
            }
        }

    @property
    def columns_created(self) -> list[str]:
        return [
            COLUMNS.HEMOGLOBIN_DISTRIBUTION_PROPENSITY,
            COLUMNS.HEMOGLOBIN_VALUE_PROPENSITY,
        ]

    @property
    def columns_required(self) -> list[str]:
        return [COLUMNS.MOTHER_ALIVE]

    @property
    def initialization_requirements(self) -> list[str | Resource]:
        return [self.randomness]

    #####################
    # Lifecycle Methods #
    #####################

    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)
        self._sim_step_name = builder.time.simulation_event_name()
        self.distribution_weights = builder.data.load(HEMOGLOBIN.DISTRIBUTION_WEIGHTS)
        self.raw_hemoglobin = builder.value.register_value_producer(
            "hemoglobin.exposure_parameters",
            source=self.get_raw_hemoglobin,
            requires_columns=get_lookup_columns(
                [
                    self.lookup_tables["exposure_mean"],
                    self.lookup_tables["exposure_sd"],
                ]
            ),
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop_update = pd.DataFrame(
            {
                COLUMNS.HEMOGLOBIN_DISTRIBUTION_PROPENSITY: self.randomness.get_draw(
                    pop_data.index, additional_key="hemoglobin_distribution_propensity"
                ),
                COLUMNS.HEMOGLOBIN_VALUE_PROPENSITY: self.randomness.get_draw(
                    pop_data.index, additional_key="hemoglobin_value_propensity"
                ),
            },
            index=pop_data.index,
        )
        self.population_view.update(pop_update)

    ##################
    # Helper methods #
    ##################

    def get_raw_hemoglobin(self, idx: pd.Index) -> pd.Series:
        """Takes the hemoglobin propensity for given simulants and returns
        their hemoglobin value from a custom ensemble distribution."""

        pop = self.population_view.get(idx)
        mean = self.lookup_tables["exposure_mean"](idx)
        sd = self.lookup_tables["exposure_sd"](idx)

        return self.sample_from_hemoglobin_distribution(
            pop["hemoglobin_distribution_propensity"],
            pop["hemoglobin_value_propensity"],
            mean,
            sd,
        )

    @staticmethod
    def _gamma_ppf(propensity, mean, sd):
        """Returns the quantile for the given quantile rank (`propensity`) of a Gamma
        distribution with the specified mean and standard deviation.
        """
        shape = (mean / sd) ** 2
        scale = sd**2 / mean
        return scipy.stats.gamma(a=shape, scale=scale).ppf(propensity)

    @staticmethod
    def _mirrored_gumbel_ppf_2017(propensity, mean, sd):
        """Returns the quantile for the given quantile rank (`propensity`) of a mirrored Gumbel
        distribution with the specified mean and standard deviation.
        """
        x_max = HEMOGLOBIN_X_MAX
        alpha = x_max - mean - (sd * np.euler_gamma * np.sqrt(6) / np.pi)
        scale = sd * np.sqrt(6) / np.pi
        return x_max - scipy.stats.gumbel_r(alpha, scale=scale).ppf(1 - propensity)

    def sample_from_hemoglobin_distribution(
        self,
        propensity_distribution: pd.Series,
        propensity_value: pd.Series,
        mean: pd.Series,
        sd: pd.Series,
    ) -> pd.Series:
        """
        Returns a sample from an ensemble distribution with the specified mean and
        standard deviation (stored in `exposure_parameters`) that is 40% Gamma and
        60% mirrored Gumbel. The sampled value is a function of the two propensities
        `prop_dist` (used to choose whether to sample from the Gamma distribution or
        the mirrored Gumbel distribution) and `propensity` (used as the quantile rank
        for the selected distribution).
        """

        gamma = propensity_distribution < self.distribution_weights["gamma"]
        gumbel = ~gamma

        hemoglobin_value = pd.Series(
            index=propensity_distribution.index, name="value", dtype=float
        )
        hemoglobin_value.loc[gamma] = self._gamma_ppf(propensity_value, mean, sd)[gamma]
        hemoglobin_value.loc[gumbel] = self._mirrored_gumbel_ppf_2017(
            propensity_value, mean, sd
        )[gumbel]

        return hemoglobin_value
