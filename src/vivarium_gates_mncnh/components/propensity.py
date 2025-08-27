from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.distributions.copula.api import GaussianCopula
from vivarium import Component
from vivarium.framework.engine import Builder

from vivarium_gates_mncnh.constants import data_values


class CorrelatedPropensities(Component):
    """A component that takes components of any type as arguments and produces propensities
    such that they are correlated according to the correlation values we define for each pair
    of compoents in data_values. These propensities will be accessible as pipelines called
    `<component_name>.correlated_propensity` in the simulation.
    These propensities will not be used by any components as a result of creating these
    pipelines, including the passed components, and it is the responsibility of each component
    that wants to use these propensities to access them explicitly."""

    ##############
    # Properties #
    ##############

    def setup(self, builder: Builder):
        self.component_names = builder.configuration.correlated_propensity_components
        self.pop_size = builder.configuration.population.population_size
        self.propensities = self.get_all_propensities()

        for component in self.component_names:
            builder.value.register_value_producer(
                f"{component}.correlated_propensity",
                source=lambda index: self.propensities.loc[index, component],
                component=self,
            )

    def get_all_propensities(self) -> pd.DataFrame:
        correlation_matrix = np.eye(len(self.component_names))  # matrix with 1s on diagonal
        name_to_index = {name: idx for idx, name in enumerate(self.component_names)}
        pairs = list(combinations(self.component_names, 2))
        # create symmetric matrix with diagonals on the 1
        for (name1, name2) in pairs:
            # first key lexiographically is first element of tuple
            correlation = data_values.PROPENSITY_CORRELATIONS[
                (min(name1, name2), max(name1, name2))
            ]
            i, j = name_to_index[name1], name_to_index[name2]
            # symmetric matrix
            correlation_matrix[i, j] = correlation_matrix[j, i] = correlation

        copula = GaussianCopula(correlation_matrix)
        propensities = copula.rvs(self.pop_size)
        propensities_df = pd.DataFrame(propensities, columns=self.component_names)
        return propensities_df
