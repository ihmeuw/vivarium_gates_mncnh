"""
Fail if the repo's working tree has uncommitted changes that could affect
simulation results.

Intended to run as the first step of an automated workflow so a dirty
checkout fails fast with a useful error.
"""
import sys

from vivarium_gates_mncnh.tools.utilities import check_clean_tree


def main() -> None:
    try:
        check_clean_tree()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
