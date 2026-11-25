"""
Compare model outputs from two environments.

These functions answer one question after an environment update: did the
artifacts, the simulation results, or the executed notebooks change? Each
returns a ``Comparison``, which later CI stages can act on or save.

This package imports nothing from the model, so a broken model cannot make the
comparison itself fail.
"""

from vivarium_gates_mncnh.compare.artifacts import compare_artifacts
from vivarium_gates_mncnh.compare.base import Comparison, ComparisonError
from vivarium_gates_mncnh.compare.notebooks import compare_notebooks
from vivarium_gates_mncnh.compare.results import compare_results
