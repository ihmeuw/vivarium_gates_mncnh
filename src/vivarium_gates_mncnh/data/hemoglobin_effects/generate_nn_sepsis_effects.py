"""Generate neonatal sepsis effect data (direct and indirect RRs) for scenario draws.

Usage:
    python generate_nn_sepsis_effects.py [--skip-existing] [draw_numbers...]

Options:
    --skip-existing    Skip draws that already have output files.
                       Default behavior is to regenerate and overwrite all outputs.

If no draw numbers are provided, generates all draws from the
SCENARIO_DRAWS list defined in vivarium_gates_mncnh.constants.metadata.
"""

import os
import sys
import time

# Must run from the hemoglobin_effects directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from vivarium_gates_mncnh.constants.metadata import SCENARIO_DRAWS


def run_single_draw(draw):
    """Run generation for a single draw in this process."""
    from hgb_nn_sepsis_effect_generation import calculate_direct_effect

    results_dir = os.getcwd()
    os.makedirs(f"{results_dir}/direct_sepsis_effects", exist_ok=True)
    os.makedirs(f"{results_dir}/indirect_sepsis_effects", exist_ok=True)

    t0 = time.time()
    print(f"[draw {draw}] Starting...")
    sys.stdout.flush()
    try:
        direct, _ = calculate_direct_effect(results_dir, draw)
        elapsed = time.time() - t0
        print(f"[draw {draw}] Complete in {elapsed:.0f}s  shape={direct.shape}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[draw {draw}] FAILED after {elapsed:.0f}s: {e}")
    sys.stdout.flush()


def main():
    args = sys.argv[1:]

    # Parse --skip-existing flag
    skip_existing = False
    if "--skip-existing" in args:
        skip_existing = True
        args.remove("--skip-existing")

    draws = [int(d) for d in args] if args else list(SCENARIO_DRAWS)

    results_dir = os.getcwd()
    os.makedirs(f"{results_dir}/direct_sepsis_effects", exist_ok=True)
    os.makedirs(f"{results_dir}/indirect_sepsis_effects", exist_ok=True)

    existing = set()
    if skip_existing:
        existing = {
            int(f.replace("draw_", "").replace(".csv", ""))
            for f in os.listdir(f"{results_dir}/direct_sepsis_effects")
            if f.startswith("draw_") and f.endswith(".csv")
        }
    draws_to_run = [d for d in draws if d not in existing]

    if not draws_to_run:
        print("All requested draws already exist. Nothing to do.")
        return

    print(f"Draws requested: {draws}")
    print(f"Skip existing:   {skip_existing}")
    if existing:
        print(f"Already exist:   {sorted(existing & set(draws))}")
    print(f"To generate:     {draws_to_run}")
    print(f"{'='*60}")
    sys.stdout.flush()

    for i, draw in enumerate(draws_to_run, 1):
        print(f"\n[{i}/{len(draws_to_run)}] Starting draw {draw}...")
        sys.stdout.flush()
        run_single_draw(draw)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
