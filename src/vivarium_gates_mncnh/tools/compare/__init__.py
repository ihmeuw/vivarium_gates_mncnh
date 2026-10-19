"""Compare two runs' outputs value by value.

These tools answer one question for an environment check: did anything move?
Each comparator reads both sides, compares their values rather than their
files, and returns a :class:`ComparisonResult` that :func:`render_report`
turns into Markdown for a pull request or a Slack message.

"""

from vivarium_gates_mncnh.tools.compare.artifacts import compare_artifacts
from vivarium_gates_mncnh.tools.compare.core import (
    ComparisonInputError,
    ComparisonResult,
    Difference,
    DifferenceKind,
    PackageChange,
    ValueDiff,
    compare_values,
)
from vivarium_gates_mncnh.tools.compare.notebooks import (
    PLOT_DATA_MIMETYPE,
    compare_notebooks,
    discover_notebook_paths,
    execute_notebooks,
)
from vivarium_gates_mncnh.tools.compare.report import render_report
from vivarium_gates_mncnh.tools.compare.results import compare_results
