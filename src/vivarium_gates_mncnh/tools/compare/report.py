"""Render comparison results as a report."""

from typing import Sequence

from vivarium_gates_mncnh.tools.compare.core import ComparisonResult


def render_report(results: Sequence[ComparisonResult]) -> str:
    """Render comparisons as Markdown fit for a pull request or Slack.

    Each comparison gets a heading naming its title, then its reference and
    candidate, a one-line verdict (clean, or how many items moved out of how
    many compared), any environment changes, any differences, and any notes.
    Differences and environment changes are laid out as fixed-width tables in
    fenced code blocks, which render the same in GitHub and Slack. Each
    difference row names the item, the kind of change, and, for value changes,
    how many values moved and the largest absolute and relative change.

    Args:
        results: The comparisons to report, in order.

    Returns:
        The report text.
    """
    raise NotImplementedError
