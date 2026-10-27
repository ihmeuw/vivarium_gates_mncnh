"""Render comparison results as a report."""

import math
import os
from pathlib import Path
from typing import Optional, Sequence, Union

from vivarium_gates_mncnh.tools.compare.core import ComparisonResult


def render_report(
    results: Sequence[ComparisonResult], report_dir: Optional[Union[str, Path]] = None
) -> str:
    """Render comparisons as Markdown fit for a pull request or Slack.

    Each comparison gets a heading naming its title, then its reference and
    candidate, the tolerance values were compared with, a one-line verdict (clean, or how many items moved out of how
    many compared), any environment changes, any differences, and any notes.
    Differences and environment changes are laid out as fixed-width tables in
    fenced code blocks, which render the same in GitHub and Slack. Each
    difference row names the item, the kind of change, and, for value changes,
    how many values moved, the largest absolute change, the change the
    tolerance allowed for that same value (``atol + rtol * |reference|``),
    and the largest relative change. A comparison with diff pages lists a
    link to each.

    Args:
        results: The comparisons to report, in order.
        report_dir: The directory the report will be saved in. Links to diff
            pages are relative to it; without it they are the pages' paths.

    Returns:
        The report text.
    """
    return "\n\n".join(_render_result(result, report_dir) for result in results) + "\n"


def _render_result(result: ComparisonResult, report_dir: Optional[Union[str, Path]]) -> str:
    lines = [
        f"## {result.title}",
        "",
        f"Reference: `{result.reference}`  ",
        f"Candidate: `{result.candidate}`",
    ]
    if result.rtol is not None or result.atol is not None:
        # A clean result means "nothing moved beyond this", so say what this is.
        lines[-1] += "  "
        lines.append(f"Tolerance: rtol={result.rtol or 0:g}, atol={result.atol or 0:g}")
    lines.append("")
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
                    _format_float(value_diff.allowed_abs_diff) if value_diff else "",
                    _format_float(value_diff.max_rel_diff) if value_diff else "",
                    difference.detail,
                ]
            )
        lines += _table(
            ["item", "kind", "n changed", "max abs", "allowed", "max rel", "detail"], rows
        )
    if result.diff_files:
        lines += ["", "Diff pages:", ""]
        for item, path in result.diff_files.items():
            link = Path(os.path.relpath(path, report_dir)) if report_dir else Path(path)
            lines.append(f"- [{item}]({link.as_posix()})")
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
