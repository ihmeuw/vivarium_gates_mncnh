"""
Compare model outputs from two environments.

These functions answer one question after an environment update: did the
artifacts, the simulation results, or the executed notebooks change? Each
returns a ``Comparison``, which later CI stages can act on or save.

Two outputs are equal when they have the same set of files and each pair of
files is equal. Which files count, and how a pair is compared, depends on the
output type:

- artifacts: the keys in two artifact files, as vivarium's ``Artifact`` reads them
- results: ``<measure>/*.parquet`` in a psimulate ``results`` directory, as
  ``pd.read_parquet`` reads them
- notebooks: ``**/executed/*.ipynb`` in a notebooks directory, by the source and
  outputs of each cell

This package imports nothing from the model, so a broken model cannot make the
comparison itself fail.
"""

from vivarium_gates_mncnh.compare.artifacts import compare_artifacts
from vivarium_gates_mncnh.compare.base import Comparison, ComparisonError
from vivarium_gates_mncnh.compare.notebooks import compare_notebooks
from vivarium_gates_mncnh.compare.results import compare_results

__all__ = [
    "Comparison",
    "ComparisonError",
    "compare_artifacts",
    "compare_notebooks",
    "compare_results",
]
