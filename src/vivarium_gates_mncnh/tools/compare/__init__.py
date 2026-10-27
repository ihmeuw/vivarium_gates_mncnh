"""Compare two runs' outputs value by value.

These tools answer one question for an environment check: did anything move?
Each comparator reads both sides, compares their values rather than their
files, and returns a :class:`ComparisonResult` that :func:`render_report`
turns into Markdown for a pull request or a Slack message.

"""

from vivarium_gates_mncnh.tools.compare.core import (
    DEFAULT_RTOL,
    ComparisonInputError,
    ComparisonResult,
    Difference,
    DifferenceKind,
    PackageChange,
    ValueDiff,
    compare_values,
)
from vivarium_gates_mncnh.tools.compare.report import render_report
