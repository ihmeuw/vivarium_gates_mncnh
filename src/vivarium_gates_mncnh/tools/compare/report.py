"""Render comparison results as a report."""

import math
from typing import Optional, Sequence

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
    return "\n\n".join(_render_result(result) for result in results) + "\n"


def _render_result(result: ComparisonResult) -> str:
    lines = [
        f"## {result.title}",
        "",
        f"Reference: `{result.reference}`  ",
        f"Candidate: `{result.candidate}`",
        "",
    ]
    if result.moved:
        n_items = len({difference.item for difference in result.differences})
        lines.append(
            f"**Moved** — {len(result.differences)} difference(s) in {n_items} item(s); "
            f"{result.n_compared} compared"
        )
    else:
        lines.append(f"**Clean** — {result.n_compared} compared")

    if result.environment_changes:
        lines += ["", "Environment changes:"]
        lines += _table(
            ["package", "reference", "candidate"],
            [
                [change.name, change.reference or "(absent)", change.candidate or "(absent)"]
                for change in result.environment_changes
            ],
        )
    if result.differences:
        lines += ["", "Differences:"]
        rows = []
        for difference in result.differences:
            value_diff = difference.value_diff
            rows.append(
                [
                    difference.item,
                    difference.kind.value,
                    str(value_diff.n_changed) if value_diff else "",
                    _format_float(value_diff.max_abs_diff) if value_diff else "",
                    _format_float(value_diff.max_rel_diff) if value_diff else "",
                    difference.detail,
                ]
            )
        lines += _table(["item", "kind", "n changed", "max abs", "max rel", "detail"], rows)
    if result.notes:
        lines += ["", "Notes:"]
        for note in result.notes:
            lines += ["", note]
    return "\n".join(lines)


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    """Lay out rows as a fixed-width table in a fenced code block."""
    widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip()

    rule = ["-" * width for width in widths]
    return ["```", line(header), line(rule)] + [line(row) for row in rows] + ["```"]


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.3g}"
