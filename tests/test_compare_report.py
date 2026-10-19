"""Tests for vivarium_gates_mncnh.tools.compare.report.render_report."""

import math
import re

from vivarium_gates_mncnh.tools.compare import (
    ComparisonResult,
    Difference,
    DifferenceKind,
    PackageChange,
    ValueDiff,
    render_report,
)

_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _code_blocks(report: str) -> list[str]:
    """The contents of every fenced code block in the report."""
    return re.findall(r"```[^\n]*\n(.*?)```", report, flags=re.DOTALL)


def _outside_code_blocks(report: str) -> str:
    return re.sub(r"```.*?```", "", report, flags=re.DOTALL)


def _numbers(text: str) -> list[float]:
    return [float(token) for token in _NUMBER.findall(text)]


def _mentions_number(text: str, value: float) -> bool:
    """Whether the text shows ``value`` in some common format.

    Formatting is not pinned: 0.125 may appear as 0.125, 1.25e-01 or 12.5%, so a
    match on the value or on one hundred times it (a percentage) is accepted, to
    three significant figures.
    """
    return any(
        math.isclose(number, target, rel_tol=5e-3)
        for number in _numbers(text)
        for target in (value, 100 * value)
    )


def _heading_lines(report: str) -> list[str]:
    return [line for line in report.splitlines() if re.match(r"^\s*#+\s", line)]


def test_clean_result_says_clean():
    """A result with no differences renders its title and a clean verdict."""
    result = ComparisonResult(
        title="Results",
        reference="/runs/reference/results",
        candidate="/runs/candidate/results",
        n_compared=3,
    )

    report = render_report([result])

    assert any("Results" in line for line in _heading_lines(report))
    assert "/runs/reference/results" in report
    assert "/runs/candidate/results" in report
    assert "clean" in report.lower()


def test_moved_result_lists_differences():
    """A result with differences renders a moved verdict, and each difference's item,
    kind and value magnitudes inside a fenced code block."""
    result = ComparisonResult(
        title="Results",
        reference="/runs/reference/results",
        candidate="/runs/candidate/results",
        n_compared=13,
        differences=[
            Difference(
                item="deaths",
                kind=DifferenceKind.VALUES,
                detail="values moved",
                value_diff=ValueDiff(n_changed=7, max_abs_diff=0.0625, max_rel_diff=0.375),
            ),
            Difference(item="ylls", kind=DifferenceKind.ONLY_IN_REFERENCE),
        ],
    )

    report = render_report([result])

    # The verdict: moved, with how many items moved out of how many compared.
    verdict_lines = [
        line for line in _outside_code_blocks(report).splitlines() if "moved" in line.lower()
    ]
    assert verdict_lines
    assert any({2.0, 13.0} <= set(_numbers(line)) for line in verdict_lines), verdict_lines
    assert "clean" not in _outside_code_blocks(report).lower()

    blocks = "\n".join(_code_blocks(report))
    assert blocks
    deaths_rows = [line for line in blocks.splitlines() if "deaths" in line]
    assert deaths_rows
    deaths_row = deaths_rows[0]
    assert DifferenceKind.VALUES.value in deaths_row
    assert 7.0 in _numbers(deaths_row)
    assert _mentions_number(deaths_row, 0.0625)
    assert _mentions_number(deaths_row, 0.375)
    ylls_rows = [line for line in blocks.splitlines() if "ylls" in line]
    assert ylls_rows
    assert DifferenceKind.ONLY_IN_REFERENCE.value in ylls_rows[0]


def test_environment_changes_rendered():
    """Environment changes render as a table of package, reference and candidate
    versions."""
    result = ComparisonResult(
        title="Results",
        reference="ref",
        candidate="cand",
        n_compared=1,
        environment_changes=[
            PackageChange("pandas", "2.2.1", "2.2.2"),
            PackageChange("newpkg", None, "0.1.7"),
            PackageChange("oldpkg", "3.4.5", None),
        ],
    )

    report = render_report([result])

    blocks = "\n".join(_code_blocks(report))
    pandas_rows = [line for line in blocks.splitlines() if "pandas" in line]
    assert pandas_rows
    assert "2.2.1" in pandas_rows[0] and "2.2.2" in pandas_rows[0]
    assert pandas_rows[0].index("2.2.1") < pandas_rows[0].index("2.2.2")
    newpkg_rows = [line for line in blocks.splitlines() if "newpkg" in line]
    assert newpkg_rows and "0.1.7" in newpkg_rows[0]
    oldpkg_rows = [line for line in blocks.splitlines() if "oldpkg" in line]
    assert oldpkg_rows and "3.4.5" in oldpkg_rows[0]
    # Environment changes alone are not movement.
    assert "clean" in _outside_code_blocks(report).lower()


def test_notes_rendered():
    """Notes render as plain lines and do not change the verdict."""
    note = "plots.ipynb, cell 3: plot data on one side only; plots not compared"
    result = ComparisonResult(
        title="Notebooks",
        reference="ref_dir",
        candidate="cand_dir",
        n_compared=2,
        notes=[note],
    )

    report = render_report([result])

    assert note in _outside_code_blocks(report)
    assert all(note not in block for block in _code_blocks(report))
    assert "clean" in _outside_code_blocks(report).lower()
    without_notes = render_report(
        [ComparisonResult(title="Notebooks", reference="ref_dir", candidate="cand_dir")]
    )
    assert "clean" in without_notes.lower()


def test_multiple_results_each_get_a_section():
    """Each result renders under its own heading, in order."""
    results = [
        ComparisonResult(
            title="Artifact", reference="a.hdf", candidate="b.hdf", n_compared=4
        ),
        ComparisonResult(
            title="Results",
            reference="ref/results",
            candidate="cand/results",
            n_compared=2,
            differences=[Difference(item="deaths", kind=DifferenceKind.SCHEMA)],
        ),
        ComparisonResult(
            title="Notebooks", reference="ref_nb", candidate="cand_nb", n_compared=1
        ),
    ]

    report = render_report(results)

    lines = report.splitlines()
    heading_index = {}
    for title in ["Artifact", "Results", "Notebooks"]:
        matches = [i for i, line in enumerate(lines) if re.match(rf"^\s*#+\s.*{title}", line)]
        assert len(matches) == 1, (title, matches)
        heading_index[title] = matches[0]
    assert heading_index["Artifact"] < heading_index["Results"] < heading_index["Notebooks"]

    artifact_section = "\n".join(lines[heading_index["Artifact"] : heading_index["Results"]])
    results_section = "\n".join(lines[heading_index["Results"] : heading_index["Notebooks"]])
    notebooks_section = "\n".join(lines[heading_index["Notebooks"] :])
    assert "clean" in artifact_section.lower() and "a.hdf" in artifact_section
    assert "moved" in results_section.lower() and "deaths" in results_section
    assert "clean" in notebooks_section.lower() and "ref_nb" in notebooks_section
