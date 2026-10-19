"""Tests for vivarium_gates_mncnh.tools.compare.report.render_report."""


def test_clean_result_says_clean():
    """A result with no differences renders its title and a clean verdict."""


def test_moved_result_lists_differences():
    """A result with differences renders a moved verdict, and each difference's item,
    kind and value magnitudes inside a fenced code block."""


def test_environment_changes_rendered():
    """Environment changes render as a table of package, reference and candidate
    versions."""


def test_notes_rendered():
    """Notes render as plain lines and do not change the verdict."""


def test_multiple_results_each_get_a_section():
    """Each result renders under its own heading, in order."""
