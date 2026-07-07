"""
Commit any pending changes and tag the resulting commit with ``v{model_number}``.

Intended to run as a workflow step after the simulation artifacts have been
produced. Exits non-zero on any git failure so the jobmon task fails loudly.
"""
import argparse
import sys

from vivarium_gates_mncnh.tools.utilities import (
    commit_pending_changes,
    create_and_push_tag,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_number",
        help='Model version number, e.g. "10.0" or "10.0.1". The git tag will be v{model_number}.',
    )
    args = parser.parse_args()

    try:
        commit_pending_changes(f"Model {args.model_number} workflow run")
        create_and_push_tag(args.model_number)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
