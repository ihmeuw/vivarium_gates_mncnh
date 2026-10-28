#!/usr/bin/env python
"""
Run the whole LBWSG PAF workflow, in order.

``run_paf_sim_step.py`` runs ONE phase per invocation. This script runs every
phase in order, each as its own invocation with the running interpreter, both
by hand and on the cluster: it is the single `lbwsg_pafs` step of
``model_specifications/artifact_workflow.yaml``, so the pipeline and a by-hand
run are the same code rather than two orchestrators to keep in step.

Usage
-----
Run it from the repository's environment::

    python run_paf_sim.py -a test_automation -l Ethiopia

Every other argument is passed through to ``run_paf_sim_step.py`` unchanged.
After fixing whatever made a phase fail, continue from it rather than starting
over::

    python run_paf_sim.py -a test_automation --from enn-artifact

or run that one phase on its own::

    python run_paf_sim_step.py --step enn-artifact -a test_automation
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from vivarium_gates_mncnh.constants.metadata import PAF_PHASES

RUN_PAF_SIM_STEP = Path(__file__).resolve().parent / "run_paf_sim_step.py"


def _run_step(step: str, passthrough: List[str]) -> None:
    """Run one phase with the running interpreter."""
    print()
    print("#" * 80)
    print(f"# step '{step}'")
    print("#" * 80)

    subprocess.run(
        [sys.executable, str(RUN_PAF_SIM_STEP), "--step", step, *passthrough],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run every phase of the LBWSG PAF workflow in order. Unrecognised "
            "arguments are passed through to run_paf_sim_step.py."
        )
    )
    parser.add_argument(
        "--from",
        dest="start_at",
        default=None,
        choices=list(PAF_PHASES),
        help=(
            "Start at this phase rather than the first, to continue a run after "
            "fixing whatever made a phase fail."
        ),
    )
    args, passthrough = parser.parse_known_args()

    if "--step" in passthrough:
        parser.error(
            "--step is chosen per phase by this script. To run a single phase, "
            "invoke run_paf_sim_step.py directly."
        )

    phases = list(PAF_PHASES)
    if args.start_at:
        skipped = phases.index(args.start_at)
        phases = phases[skipped:]

    print("\n" + "=" * 80)
    print(f"LBWSG PAF workflow -- {len(phases)} of {len(PAF_PHASES)} phases")
    if args.start_at:
        print(f"Starting at '{args.start_at}'; {skipped} earlier phase(s) skipped")
    print(f"Environment: {sys.prefix}")
    print("=" * 80)

    for step in phases:
        try:
            _run_step(step, passthrough)
        except subprocess.CalledProcessError as e:
            print(
                f"\nStep '{step}' failed (exit code {e.returncode}). Earlier phases "
                "have already run; once you have fixed the cause, re-run this phase "
                f"on its own with:\n"
                f"  python {RUN_PAF_SIM_STEP} --step {step} {' '.join(passthrough)}\n"
                f"Or fix the cause and re-run this script with --from {step} to "
                "continue from here.",
                file=sys.stderr,
            )
            raise SystemExit(e.returncode)

    print()
    print("#" * 80)
    print(f"# All {len(phases)} phases completed.")
    print("#" * 80)


if __name__ == "__main__":
    main()
