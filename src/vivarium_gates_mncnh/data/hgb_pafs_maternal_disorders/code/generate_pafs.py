"""
Run maternal disorders PAF generation for all scenario draws that have neonatal sepsis RRs.

Usage:
    python generate_pafs.py [--skip-existing]

Options:
    --skip-existing    Skip (location, draw) pairs that already have output files.
                       Default behavior is to regenerate and overwrite all outputs.
"""

import os
import sys
import time
import warnings
from functools import partial
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# Force unbuffered output so progress is visible in real time
print = partial(print, flush=True)

# Must run from the hgb_pafs_maternal_disorders/code directory
os.chdir(Path(__file__).parent)

from hgb_maternal_disorder_paf_generation import (
    calculate_pafs,
    load_hemoglobin_rrs_on_maternal_disorders,
)

from vivarium_gates_mncnh.constants.metadata import SCENARIO_DRAWS

POPULATION_SIZE = 500_000
LOCATIONS = ["ethiopia", "nigeria", "pakistan"]
SEPSIS_RR_DIR = Path("../../hemoglobin_effects/direct_sepsis_effects/")
OUTPUT_DIR = Path("../outputs/")


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def progress_bar(current, total, width=30):
    frac = current / total if total else 0
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    return f"|{bar}| {current}/{total} ({frac:.0%})"


def get_available_draws():
    """Get draws that have neonatal sepsis RR files AND are in the scenario draws."""
    sepsis_draws = {int(f.stem.split("_")[1]) for f in SEPSIS_RR_DIR.glob("draw_*.csv")}
    draws = sorted(sepsis_draws & set(SCENARIO_DRAWS))
    return draws


def main():
    skip_existing = "--skip-existing" in sys.argv

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    draws = get_available_draws()
    total_tasks = len(draws) * len(LOCATIONS)

    print("=" * 60)
    print("  Maternal Disorders PAF Generation")
    print("=" * 60)
    print(f"  Locations:       {', '.join(LOCATIONS)}")
    print(f"  Draws:           {draws}")
    print(f"  Population size: {POPULATION_SIZE:,}")
    print(
        f"  Total tasks:     {total_tasks} ({len(draws)} draws x {len(LOCATIONS)} locations)"
    )
    print(f"  Skip existing:   {skip_existing}")
    print("=" * 60)

    # Check which (location, draw) pairs have already been completed
    existing = set()
    if skip_existing:
        for loc in LOCATIONS:
            loc_dir = OUTPUT_DIR / loc
            if loc_dir.is_dir():
                for f in loc_dir.glob("draw_*.csv"):
                    draw_num = int(f.stem.split("_")[-1])
                    existing.add((loc, draw_num))

    tasks = [(loc, d) for loc in LOCATIONS for d in draws if (loc, d) not in existing]
    completed_count = total_tasks - len(tasks)

    if existing:
        print(f"\n  Already completed: {completed_count}/{total_tasks}")
    print(f"  Remaining:         {len(tasks)}/{total_tasks}")
    print()

    # Load hemoglobin RRs (shared across all draws/locations)
    print("Loading hemoglobin RRs on maternal disorders...")
    hgb_rrs = load_hemoglobin_rrs_on_maternal_disorders()
    print("  Done.\n")

    run_start = time.time()
    draw_times = []

    for i, (location, draw) in enumerate(tasks):
        task_num = completed_count + i + 1
        elapsed = time.time() - run_start
        if draw_times:
            avg_time = sum(draw_times) / len(draw_times)
            eta = avg_time * (len(tasks) - i)
            eta_str = f"ETA: {fmt_time(eta)}"
        else:
            eta_str = "ETA: calculating..."

        print(
            f"  {progress_bar(task_num - 1, total_tasks)}  "
            f"Elapsed: {fmt_time(elapsed)}  {eta_str}"
        )
        print(f"  -> Location: {location}, Draw: {draw}")

        draw_start = time.time()
        try:
            paf = calculate_pafs(location, draw, POPULATION_SIZE, hgb_rrs)
            paf["draw"] = draw
            paf["location_run"] = location
            loc_dir = OUTPUT_DIR / location
            loc_dir.mkdir(parents=True, exist_ok=True)
            out_file = loc_dir / f"draw_{draw}.csv"
            paf.to_csv(out_file, index=False)
            draw_elapsed = time.time() - draw_start
            draw_times.append(draw_elapsed)
            print(f"     Done in {fmt_time(draw_elapsed)}")
        except Exception as e:
            draw_elapsed = time.time() - draw_start
            draw_times.append(draw_elapsed)
            print(f"     ERROR ({fmt_time(draw_elapsed)}): {e}", file=sys.stderr)

    total_elapsed = time.time() - run_start

    print()
    print("=" * 60)
    print(f"  {progress_bar(total_tasks, total_tasks)}")
    print(f"  Total time:    {fmt_time(total_elapsed)}")
    if draw_times:
        print(f"  Avg per task:  {fmt_time(sum(draw_times) / len(draw_times))}")
    print("=" * 60)


if __name__ == "__main__":
    main()
