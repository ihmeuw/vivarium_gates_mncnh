"""End-to-end tests of the compare_outputs command line interface."""


def test_results_clean_exits_zero():
    """`compare_outputs results` on identical directories prints a clean report and
    exits 0."""


def test_results_moved_exits_one():
    """`compare_outputs results` on differing directories prints the moved measure and
    exits 1."""


def test_tolerance_options_passed_through():
    """--rtol/--atol suppress a small change, turning exit 1 into exit 0."""


def test_bad_input_exits_two():
    """A missing path exits 2 with a one-line message rather than a traceback."""


def test_artifacts_command():
    """`compare_outputs artifacts` exits 0 on identical artifacts and 1 on changed
    ones."""


def test_notebooks_command():
    """`compare_outputs notebooks` exits 0 on identical directories and 1 on changed
    ones."""


def test_execute_notebooks_command():
    """`compare_outputs execute-notebooks` writes executed notebooks to OUTPUT_DIR,
    exiting 0 when all run and 1 when any fails."""
