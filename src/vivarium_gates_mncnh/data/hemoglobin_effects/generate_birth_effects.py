"""Generate hemoglobin --> birth effects (LBWSG shifts) per draw.

Usage:
    python generate_birth_effects.py [--skip-existing] [draw_numbers...]

Drives the draw-parallelizable functions in ``hgb_birth_effect_generation.py``
from the CLI so that ``parallel_runner.py`` can fan them out across draws,
mirroring ``generate_nn_sepsis_effects.py``. ``parallel_runner.py`` appends a
single draw as the last positional arg, so this script accepts one draw per
invocation (and also supports multiple draws when run directly).

For each draw this writes (relative to this file's directory):
    - lbwsg_shifts/draw_<N>.csv         (calculate_and_save_lbwsg_shifts)
    - iv_iron_lbwsg_shifts/draw_<N>.csv (calculate_and_save_iv_iron_lbwsg_shifts)

These outputs are consumed by the neonatal sepsis step
(``hgb_nn_sepsis_effect_generation.load_lbwsg_shifts``) and by
``make_artifacts`` (see ``data/loader.py``), so this step MUST run before
``hemoglobin_effects_sepsis``.

If no draw numbers are provided, generates all draws from the SCENARIO_DRAWS
list defined in ``vivarium_gates_mncnh.constants.metadata``.

NOTE: ``calculate_iv_iron_stillbirth_effects()`` loops all draws internally
(~7 min) and is NOT draw-parallel. It is deliberately NOT called here; it is
handled as a separate single-invocation workflow step (see
``artifact_workflow.yaml``).
"""

import sys
import time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))

from vivarium_gates_mncnh.constants.metadata import SCENARIO_DRAWS

# Subdirectories (relative to _DIR) written by the birth-effect functions.
_SHIFTS_SUBDIR = "lbwsg_shifts"
_IV_IRON_SUBDIR = "iv_iron_lbwsg_shifts"


def _draw_column(draw) -> str:
    """Convert a draw number to the GBD column/file stem the birth-effect
    functions expect.

    ``get_lbwsg_shifts`` indexes RR data by draw *column name* (e.g.
    ``"draw_109"``), and the save functions build the output filename by
    string-concatenating that same value (yielding ``draw_109.csv``, which is
    exactly what the sepsis step reads back). ``parallel_runner.py`` passes the
    bare draw number (e.g. ``109``), so normalize it here.
    """
    draw = str(draw)
    return draw if draw.startswith("draw_") else f"draw_{draw}"


def run_single_draw(draw) -> None:
    """Generate and save LBWSG shifts + IV iron LBWSG shifts for one draw."""
    t0 = time.time()
    draw_col = _draw_column(draw)
    print(f"[{draw_col}] Starting...")
    sys.stdout.flush()

    try:
        from hgb_birth_effect_generation import (
            calculate_and_save_iv_iron_lbwsg_shifts,
            calculate_and_save_lbwsg_shifts,
        )

        # These functions build paths via string concatenation, so the results
        # directory must be a string ending in "/".
        results_dir = str(_DIR) + "/"
        calculate_and_save_lbwsg_shifts(results_dir, draw_col)
        calculate_and_save_iv_iron_lbwsg_shifts(results_dir, draw_col)
    except Exception as e:
        elapsed = time.time() - t0
        # Re-raise so a non-zero exit code propagates to parallel_runner, which
        # marks the draw FAILED (each draw runs in its own subprocess).
        print(f"[{draw_col}] FAILED after {elapsed:.0f}s: {e}")
        sys.stdout.flush()
        raise

    elapsed = time.time() - t0
    print(f"[{draw_col}] Complete in {elapsed:.0f}s")
    sys.stdout.flush()


def main() -> None:
    args = sys.argv[1:]

    # Parse --skip-existing flag
    skip_existing = False
    if "--skip-existing" in args:
        skip_existing = True
        args.remove("--skip-existing")

    draws = [int(d) for d in args] if args else list(SCENARIO_DRAWS)

    shifts_dir = _DIR / _SHIFTS_SUBDIR
    iv_iron_dir = _DIR / _IV_IRON_SUBDIR
    shifts_dir.mkdir(parents=True, exist_ok=True)
    iv_iron_dir.mkdir(parents=True, exist_ok=True)

    # Skip draws whose outputs already exist in BOTH directories.
    existing = set()
    if skip_existing:
        for draw in draws:
            stem = _draw_column(draw)
            if (shifts_dir / f"{stem}.csv").exists() and (
                iv_iron_dir / f"{stem}.csv"
            ).exists():
                existing.add(draw)
    draws_to_run = [d for d in draws if d not in existing]

    print(f"Draws requested: {draws}")
    print(f"Skip existing:   {skip_existing}")
    if existing:
        print(f"Already exist:   {sorted(existing)}")
    print(f"To generate:     {draws_to_run}")
    print("=" * 60)
    sys.stdout.flush()

    if not draws_to_run:
        print("All requested draws already exist. Nothing to do.")
        return

    for i, draw in enumerate(draws_to_run, 1):
        print(f"\n[{i}/{len(draws_to_run)}] draw {draw}...")
        sys.stdout.flush()
        run_single_draw(draw)

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
