from collections import Counter

# from typing import TypeAlias, Any
from collections.abc import Callable, Iterable
from dataclasses import InitVar, dataclass, field
from pprint import pprint

import numpy as np
import pandas as pd
import scipy
from birth_facility import (
    LAST_USED,
    BirthFacilityChoiceData,
    BirthFacilityChoiceModel,
    BirthFacilityModelWithLBWSGandUltrasound,
    BirthFacilityModelWithUltrasoundAndSimpleGAError,
    Seed,
)
from correlation import generate_correlation_matrix
from numpy.random import SeedSequence
from vivarium_helpers.prob_distributions.fit import log_loss

# Flag to indicate that a keyword argument was not passed
NOT_PASSED = "not-passed"

# Minimization methods for scipy.optimize.minimize that accept the
# `bounds` parameter
METHODS_WITH_BOUNDS = [
    "Nelder-Mead",
    "L-BFGS-B",
    "TNC",
    "SLSQP",
    "Powell",
    "trust-constr",
    "COBYLA",
    "COBYQA",
]


def elementwise_log_loss(y_true, y_pred, epsilon=1e-15):
    """
    Calculates the log loss (binary cross-entropy) between true and predicted values.

    Args:
        y_true (array-like): True labels (0 or 1) or probabilities.
        y_pred (array-like): Predicted probabilities (values between 0 and 1).
        epsilon (float, optional): A small value to prevent log(0) errors. Defaults to 1e-15.

    Returns:
        float: The calculated log loss.
    """
    # NOTE: For elementwise operations, it should be ok to propagate
    # NaNs like pandas objects do, so we shouldn't need to convert to
    # NumPy arrays; broadcasting should be easier by using aligned
    # pandas objects.
    # y_true, y_pred = map(np.asarray, (y_true, y_pred))
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # Since elementwise, don't take mean like in ordinary log loss
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


def normalized_elementwise_loss(
    targets,
    empirical_targets,
    elementwise_loss_func=elementwise_log_loss,
):
    """Computes normalized loss of each element instead of taking the
    mean. Useful for diagnosing if some target probabilities are more
    problematic than others.
    """
    normalized_loss = (
        elementwise_loss_func(targets, empirical_targets)
        # Normalize by subtracting off minimum loss
        - elementwise_loss_func(targets, targets)
    )
    return normalized_loss


# Create a simple logger to log NumPy floating point errors.
# Namely, I got a bunch of these errors the first time I tried
# optimizing:
# RuntimeWarning: invalid value encountered in subtract
#   np.max(np.abs(fsim[0] - fsim[1:])) <= fatol):
class ErrorLog:
    def __init__(self):
        self.err_counts = Counter()

    def write(self, message):
        self.err_counts[message] += 1


### Class for finding optimum values of missing parameters for birth facility model


class OptimalSolutionFinder:
    def __init__(
        self,
        model: BirthFacilityChoiceModel,
        loss_func: Callable | None = None,
        method: str = "Nelder-Mead",
        **kwargs,  # keyword args to pass to scipy.optimize.minimize
    ):
        self.model = model
        self.loss_func = loss_func or self.model.loss_func
        self.method = method
        self.kwargs = kwargs

        # Total number of unknown parameters in model
        self._num_parameters = len(self.model.parameter_data)
        # Number of unknown parameters that represent correlations
        self._num_correlations = self.model.parameter_data["correlated_pair"].notna().sum()
        self._num_correlated_vars = len(self.model.correlated_propensities)
        assert self._num_correlations == scipy.special.binom(
            self._num_correlated_vars, 2
        ), "Wrong number of correlations..."
        self._param_name_to_position = self.model.parameter_data["position"]
        self._all_bounds = (
            model.parameter_data[["bounds", "position"]].set_index("position").squeeze()
        )
        self.error_log = ErrorLog()

    def _convert_to_position(self, name_or_position):
        """Convert a parameter name or position (index in solution
        vector) into its position.
        """
        return (
            self._param_name_to_position[name_or_position]
            if name_or_position in self._param_name_to_position
            else name_or_position
        )

    def minimum_loss(self):
        """Compute the minimum possible loss for the model, which is the
        loss between the target values and themselves. This would be 0
        for L2 loss, but for log loss, is the mean entropy of the target
        probabilities, treated as a sequence of Bernoulli distributions.
        """
        return self.loss_func(self.model.targets, self.model.targets)

    def reset_error_log(self):
        self.error_log = ErrorLog()

    def objective_function(self, x, seed):
        """args are the arguments besides x to pass to `do_all_steps`."""
        # Get the returned population table (with facility choice added),
        # ignore propensities
        try:
            self.model.do_all_steps(x, seed)
        except ValueError as e:
            if e.args[0] == "The input matrix must be symmetric positive semidefinite.":
                self.error_log.write("In objective_function: " + e.args[0])
                return np.inf
            else:
                raise
        empirical_values = self.model.calculate_targets()
        targets = self.model.targets
        # NOTE: Lengths can be different if empirical_values is missing
        # one of the strata of targets. We could fix this by instead
        # doing this:
        # empirical_values = (
        #     self.model.calculate_targets()
        #     .reindex(targets.index, fill_value=0.0)
        # )
        # But that might be slower than simply returning infinity,
        # without much benefit, so maybe it's better to keep it simple.
        if len(empirical_values) != len(targets):
            print("Different lengths?")
            loss = np.inf
        else:
            loss = self.loss_func(targets, empirical_values)
        return loss

    def find_solution(
        self,
        x0_reduced,
        fixed_x_components={},
        seed=LAST_USED,
        reset_log=True,
        **kwargs,  # keyword args to pass to scipy.optimize.minimize
    ):
        if reset_log:
            self.reset_error_log()
        # Convert parameter names to x-component positions if necessary
        fixed_x_components = {
            self._convert_to_position(k): v for k, v in fixed_x_components.items()
        }
        # Override (but don't overwrite) stored kwargs with passed
        # kwargs
        kwargs = self.kwargs | kwargs
        method = kwargs.pop("method", self.method)
        # Use NOT_PASSED flag instead of None to allow explicitly
        # passing None in order to override these default bounds
        bounds = kwargs.pop("bounds", NOT_PASSED)
        if bounds == NOT_PASSED:
            if method in METHODS_WITH_BOUNDS:
                # Get bounds for components we'll be optimizing over
                bounds = self._all_bounds.loc[
                    self._all_bounds.index.difference(fixed_x_components.keys())
                ]
            else:
                # Use default value of None for methods that don't
                # accept `bounds` parameter
                bounds = None

        # Optimize - be sure to pass a constant seed and to fix the
        # appropriate parameters
        # print(x0_reduced)
        # print(bounds)
        result = scipy.optimize.minimize(
            fix_x_components(
                self.objective_function, self._num_parameters, fixed_x_components
            ),
            x0_reduced,
            args=(seed,),  # Additional args to objective_function
            method=method,
            bounds=bounds,
            **kwargs,  # Additional keyword args for minimize
        )
        return result

    def _get_random_generator(self, seed):
        """Create a Generator from a seed since we don't know what
        format the seed is in, but all valid seeds can be passed to
        the Generator constructor. Use the seed currently stored in
        self.model if seed==LAST_USED.
        """
        if seed == LAST_USED:
            rng = np.random.default_rng(self.model.seed)
        else:
            rng = np.random.default_rng(seed)
        return rng

    def find_solutions(
        self,
        num_initial_conditions,  # number of initial conditions to try
        fixed_x_components={},
        fixed_x_components_on_iteration=(),
        seed=LAST_USED,
        # NOTE: Seems like re-using the same seed in each iteration
        # may work slightly better than spawning new seeds, but it's
        # hard to tell for sure...
        spawn_new_seeds=False,
        reset_log=True,
        **kwargs,
    ):
        """Use scipi.optimize.minimize to find solutions that satisfy the
        constraints, starting from multiple randomly chosen initial
        conditions.
        """
        if reset_log:
            self.reset_error_log()
        # Convert parameter names to x-component positions if necessary
        fixed_x_components_on_iteration = [
            self._convert_to_position(i) for i in fixed_x_components_on_iteration
        ]

        # Create a Generator from the seed since we don't know what
        # format the seed is in, but all valid seeds can be passed to
        # the Generator constructor
        rng = self._get_random_generator(seed)
        # Record the initial SeedSequence
        initial_seed_seq = rng.bit_generator.seed_seq
        if spawn_new_seeds:
            # Spawn a new SeedSequence (not a new Generator, because we need
            # to keep the seed constant during optimization) for each
            # initial condition
            child_seed_seqs = initial_seed_seq.spawn(num_initial_conditions)
        else:
            # Reuse the same seed for all initial points
            child_seed_seqs = num_initial_conditions * [initial_seed_seq]

        # Record initial points so we can recreate results
        initial_points = np.empty(
            (num_initial_conditions, self._num_parameters), dtype="float"
        )
        # Record full list of results
        results = []
        # Record fixed values so we can recreate results
        fixed_values = []
        # Record empirical targets so we can evaluate performance
        empirical_targets = []
        # Record population proportions so we can validate the model
        population_proportions = []

        # Log "invalid value" errors to the log instead of printing
        orig_settings = np.seterr(invalid="log")
        orig_handler = np.seterrcall(self.error_log)

        for i in range(num_initial_conditions):
            # Generate a random correlation matrix as the
            # "projection" of a positive semidefinite matrix
            # drawn from a Wishart distribution
            gaussian_matrix = rng.normal(
                size=(self._num_correlated_vars, self._num_correlated_vars)
            )
            corr0 = generate_correlation_matrix(gaussian_matrix)

            # Set initial conditions
            x0 = [
                # List all correlations from the upper triangle
                # NOTE: It is assumed that all correlations come before
                # all other parameters in the parameter list
                *corr0[np.triu_indices_from(corr0, k=1)],
                # Generate initial probabilities for the other parameters
                *rng.uniform(size=self._num_parameters - self._num_correlations),
            ]
            # Fix specified components of x with components of random x0
            # during optimization
            fixed_values.append({i: x0[i] for i in fixed_x_components_on_iteration})
            # Fix specified components of x according to global fixed
            # values
            fixed_values[i].update(fixed_x_components)
            # print(fixed_values)
            # print(x0)

            x0_reduced = [x0_j for j, x0_j in enumerate(x0) if j not in fixed_values[i]]
            # print(x0_reduced)
            # print(fill_x_values(x0_reduced, fixed_values))

            # Run the optimization
            result = self.find_solution(
                x0_reduced,
                fixed_values[i],
                child_seed_seqs[i],
                # Use the same error log across all iterations
                reset_log=False,
                **kwargs,
            )
            initial_points[i] = fill_x_values(x0_reduced, fixed_values[i])
            results.append(result)
            # Record latest objective probabilities, reindexing in case
            # there are any strata missing
            # print(results)
            # print(self.model.population)
            empirical_targets.append(self.model.calculate_targets(as_series=True))
            population_proportions.append(self.model.get_population_proportions())

        # Reset NumPy error handling to original settings
        np.seterr(**orig_settings)
        np.seterrcall(orig_handler)

        return Solutions(
            results,
            fixed_values,
            initial_points,
            self.model.parameter_data.index,
            empirical_targets,
            population_proportions,
            initial_seed_seq,
            child_seed_seqs,
            self.error_log.err_counts.copy(),
            self.minimum_loss(),
        )

    def find_solutions_for_parameter_grid(
        self,
        num_initial_conditions,
        anc_lbw_range=(0.8, -0.8, -0.1),
        lbwsg_facility_range=(0.8, -0.8, -0.1),
        seed=LAST_USED,
        spawn_new_seeds=False,
        spawn_new_seeds_on_iteration=False,
        reset_log=True,
        **kwargs,
    ):
        """
        For each pair of (ANC-LBWSG, LBWSG-facility) correlations in the
        specified grid, find optimal values for the remaining three
        model parameters. Fixing these two correlations seems to result
        in the best convergence properties for the optimization.

        Args:
            anc_lbw_range: Tuple of (start, stop, step) for ANC-LBWSG
                correlation search
            lbwsg_facility_range: Tuple of (start, stop, step) for
                LBWSG-facility correlation search

        Returns:
            DataFrame with results for all parameter combinations
        """
        if reset_log:
            self.reset_error_log()

        solutions_dict = {}

        # Convert seed to a generator to get a known type
        rng = self._get_random_generator(seed)
        next_seed = rng.bit_generator.seed_seq

        # print(anc_lbw_range, lbwsg_facility_range)
        for anc_lbw_corr in np.arange(*anc_lbw_range):
            for lbwsg_facility_corr in np.arange(*lbwsg_facility_range):
                print(
                    f"Testing ANC-LBWSG correlation: {anc_lbw_corr:.1f}, "
                    f"LBWSG-Facility correlation: {lbwsg_facility_corr:.1f}"
                )

                # Set fixed correlation values
                fixed_x_components = {0: anc_lbw_corr, 2: lbwsg_facility_corr}

                # Spawn a new seed for each grid point if requested
                if spawn_new_seeds:
                    next_seed = rng.bit_generator.seed_seq.spawn(1)

                # Find solutions
                solutions = self.find_solutions(
                    num_initial_conditions,
                    fixed_x_components,
                    seed=next_seed,
                    spawn_new_seeds=spawn_new_seeds_on_iteration,
                    reset_log=False,
                    **kwargs,
                )
                pprint(solutions.sorted_x_values.loc[0])
                print("Loss = ", solutions.sorted_losses[0])
                print()
                solutions_dict[(anc_lbw_corr, lbwsg_facility_corr)] = solutions
        return solutions_dict


@dataclass
class Solutions:
    """Class for returning a set of solutions from `find_solutions`."""

    results: scipy.optimize.OptimizeResult
    fixed_values: list[dict[int, float]]
    initial_points: np.ndarray
    parameter_names: InitVar[Iterable[str]]
    empirical_targets: list[pd.Series]
    population_proportions: list[pd.Series]
    initial_seed: Seed
    child_seeds: list[Seed]
    error_counts: Counter
    min_loss: InitVar[float] = 0.0
    sorted_indices: np.ndarray = field(init=False)
    sorted_losses: list = field(init=False)
    sorted_x_values: pd.DataFrame = field(init=False)
    sorted_targets: pd.DataFrame = field(init=False)
    sorted_pop_proportions: pd.DataFrame = field(init=False)

    def __post_init__(self, parameter_names, min_loss):
        # NOTE: Using numpy.argsort here to sort NaNs to the end
        self.sorted_indices = np.argsort([result.fun for result in self.results])
        self.sorted_losses = [
            float(self.results[i].fun) - min_loss for i in self.sorted_indices
        ]
        self.sorted_x_values = pd.DataFrame(
            # TODO: Looks like maybe the np.array is not necessary here?
            [
                fill_x_values(self.results[i].x, self.fixed_values[i])
                for i in self.sorted_indices
            ],
            columns=parameter_names,
        )
        self.sorted_targets = pd.concat(
            [self.empirical_targets[i] for i in self.sorted_indices],
            axis=1,
            ignore_index=True,
        )
        self.sorted_pop_proportions = pd.concat(
            [self.population_proportions[i] for i in self.sorted_indices],
            axis=1,
            ignore_index=True,
        )


### Functions for removing and filling in components of an array x


def fix_x_components(function, num_components, value_map):
    """Takes a function of the form f(x, *args, **kwargs), where
    x is a NumPy array with n elements, and returns a function
    of the same form except that some elements of x have been fixed
    as specified by the mapping `value_map`.

    Namely, if `value_map` has k items of the form `(i, value)`,
    the returned function
    g has the same form as f, execpt that its first parameter is a
    NumPy array `x` with n-k elements, and g is defined by

    g(x, *args, **args) = f(x_expanded, *args, **args),

    where x_expanded[i] = value for each i, and for each i,
    x_expanded[i-1:i] is filled in with elements of x in the same
    order they appear in x. Essentially, g is a partially
    applied version of f where the different compnents of x are
    treated as separate sub-arguments, and we have applied f on the
    i^th of these sub-arguments, specifying that that argument is
    `value`.

    This function is designed to wrap a function f(x, *args) passed
    to scipy.optimize.minimize, enabling fixing one or more of the
    components of x during the minimization routine, making scipy
    minimize over the remaining parameters only.
    """
    # Do work outside returned function if possible since the returned
    # function will be called repeatedly during optimization
    full_x = np.empty(num_components, dtype="float")
    # Sort (index, value) pairs by index (then value)
    pairs = sorted(dict(value_map).items())
    for i, value in pairs:
        # full_x has missing values filled by value_map
        full_x[i] = value
    # print(full_x)
    # The x passed to the returned function has values omitted
    def partially_applied_function(x, *args, **kwargs):
        prev_i, pos = -1, 0
        for i, value in pairs:
            # number of elements to fill in before position i
            num_elements = i - (prev_i + 1)
            full_x[prev_i + 1 : i] = x[pos : pos + num_elements]
            prev_i = i
            pos += num_elements
        full_x[prev_i + 1 :] = x[pos:]
        # print(x, full_x)
        return function(full_x, *args, **kwargs)

    return partially_applied_function


def fill_x_values(reduced_x, value_map):
    """Takes a 'reduced' x-array with values that have been removed,
    and fills in the missing values with value_map.
    """
    # Note: We could call this function from fix_x_components, but I'm
    # not sure how to do that without repeating unnecessary work
    num_components = len(reduced_x) + len(value_map)
    full_x = np.empty(num_components, dtype="float")
    pairs = sorted(dict(value_map).items())
    prev_i, pos = -1, 0
    for i, value in pairs:
        full_x[i] = value
        # number of elements to fill in before position i
        num_elements = i - (prev_i + 1)
        full_x[prev_i + 1 : i] = reduced_x[pos : pos + num_elements]
        prev_i = i
        pos += num_elements
    full_x[prev_i + 1 :] = reduced_x[pos:]
    return full_x


def main(*args):
    location = "pakistan"
    pop_size = 20_000
    seed = 281780587996846863749789528493434682754

    print(args)

    model_name, num_facility_types = args
    # Convert command line argument from string to int
    num_facility_types = int(num_facility_types)

    if model_name == "ultrasound":
        model_class = BirthFacilityModelWithLBWSGandUltrasound
    elif model_name == "ga_error":
        model_class = BirthFacilityModelWithUltrasoundAndSimpleGAError
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    # Reduced range for testing
    anc_lbw_range = (0.7, -0.3, -0.1)
    lbwsg_facility_range = (0.7, -0.3, -0.1)

    output_filename = (
        f"parameter_search_results_{location}_{model_name}_{num_facility_types}_all.csv"
    )

    data = BirthFacilityChoiceData(location)
    model = model_class(data, pop_size, num_facility_types, seed=seed)
    finder = OptimalSolutionFinder(model)

    num_initial_points = 150
    all_solutions = finder.find_solutions_for_parameter_grid(
        num_initial_points, anc_lbw_range, lbwsg_facility_range
    )

    best_solutions = pd.concat(
        # Use slice to create a DataFrame not Series, to get horizontal
        # orientation automatically
        [solutions.sorted_x_values.loc[0:0] for solutions in all_solutions.values()],
        ignore_index=True,  # Want RangeIndex, not all 0s
    )
    losses = pd.Series(
        [solutions.sorted_losses[0] for solutions in all_solutions.values()], name="error"
    )
    best_solutions = best_solutions.join(losses)
    best_solutions.to_csv(output_filename, index=False)
    print(f"Done!\n{seed=}")
    return all_solutions


if __name__ == "__main__":
    import sys

    print(sys.argv)
    main(*sys.argv[1:])
