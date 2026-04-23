"""Run maternal disorders PAF generation for all scenario draws that have neonatal sepsis RRs.

Usage:
    python generate_pafs.py [--skip-existing] [--locations LOC [LOC ...]]

Options:
    --skip-existing           Skip (location, draw) pairs that already have output files.
                              Default behavior is to regenerate and overwrite all outputs.
    --locations LOC [LOC ..]  Run only the specified locations (case-insensitive).
                              Defaults to all locations in metadata.LOCATIONS.
"""

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_CODE_DIR = Path(__file__).parent

sys.path.insert(0, str(_CODE_DIR))

from hgb_maternal_disorder_paf_generation import (
    calculate_pafs,
    load_hemoglobin_rrs_on_maternal_disorders,
)

from vivarium_gates_mncnh.constants.metadata import LOCATIONS as _LOCATIONS
from vivarium_gates_mncnh.constants.metadata import SCENARIO_DRAWS

POPULATION_SIZE = 500_000
ALL_LOCATIONS = [loc.lower() for loc in _LOCATIONS]
SEPSIS_RR_DIR = _CODE_DIR / "../../hemoglobin_effects/direct_sepsis_effects/"
OUTPUT_DIR = _CODE_DIR / "../outputs/"


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
    sepsis_draws = {int(f.stem.split("_")[-1]) for f in SEPSIS_RR_DIR.glob("draw_*.csv")}
    draws = sorted(sepsis_draws & set(SCENARIO_DRAWS))
    return draws


def _parse_locations(args):
    """Extract --locations values from *args*, returning (locations, remaining_args)."""
    if "--locations" not in args:
        return ALL_LOCATIONS, args
    idx = args.index("--locations")
    locs = []
    for val in args[idx + 1 :]:
        if val.startswith("--"):
            break
        locs.append(val.lower())
    if not locs:
        print("Error: --locations requires at least one location.", file=sys.stderr)
        sys.exit(1)
    unknown = set(locs) - set(ALL_LOCATIONS)
    if unknown:
        print(
            f"Error: unknown location(s): {', '.join(sorted(unknown))}. "
            f"Valid: {', '.join(ALL_LOCATIONS)}",
            file=sys.stderr,
        )
        sys.exit(1)
    remaining = args[:idx] + args[idx + 1 + len(locs) :]
    return locs, remaining


def main():
    args = sys.argv[1:]
    locations, args = _parse_locations(args)
    skip_existing = "--skip-existing" in args

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    draws = get_available_draws()
    total_tasks = len(draws) * len(locations)

    print("=" * 60)
    print("  Maternal Disorders PAF Generation")
    print("=" * 60)
    print(f"  Locations:       {', '.join(locations)}")
    print(f"  Draws:           {draws}")
    print(f"  Population size: {POPULATION_SIZE:,}")
    print(
        f"  Total tasks:     {total_tasks} ({len(draws)} draws x {len(locations)} locations)"
    )
    print(f"  Skip existing:   {skip_existing}")
    print("=" * 60)

    # Check which (location, draw) pairs have already been completed
    existing = set()
    if skip_existing:
        for loc in locations:
            loc_dir = OUTPUT_DIR / loc
            if loc_dir.is_dir():
                for f in loc_dir.glob("draw_*_maternal.csv"):
                    draw_num = int(f.stem.split("_")[1])
                    existing.add((loc, draw_num))

    tasks = [(loc, d) for loc in locations for d in draws if (loc, d) not in existing]
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
    failures = []

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
            f"Elapsed: {fmt_time(elapsed)}  {eta_str}",
            flush=True,
        )
        print(f"  -> Location: {location}, Draw: {draw}", flush=True)

        draw_start = time.time()
        try:
            maternal_paf, neonatal_paf = calculate_pafs(
                location, draw, POPULATION_SIZE, hgb_rrs
            )
            maternal_paf["draw"] = draw
            maternal_paf["location_run"] = location
            neonatal_paf["draw"] = draw
            neonatal_paf["location_run"] = location
            loc_dir = OUTPUT_DIR / location
            loc_dir.mkdir(parents=True, exist_ok=True)
            maternal_paf.to_csv(loc_dir / f"draw_{draw}_maternal.csv", index=False)
            neonatal_paf.to_csv(loc_dir / f"draw_{draw}_neonatal.csv", index=False)
            draw_elapsed = time.time() - draw_start
            draw_times.append(draw_elapsed)
            print(f"     Done in {fmt_time(draw_elapsed)}", flush=True)
        except Exception as e:
            draw_elapsed = time.time() - draw_start
            draw_times.append(draw_elapsed)
            failures.append((location, draw, str(e)))
            print(f"     ERROR ({fmt_time(draw_elapsed)}): {e}", file=sys.stderr, flush=True)

    total_elapsed = time.time() - run_start

    print(flush=True)
    print("=" * 60, flush=True)
    print(f"  {progress_bar(total_tasks, total_tasks)}", flush=True)
    print(f"  Total time:    {fmt_time(total_elapsed)}", flush=True)
    if draw_times:
        print(f"  Avg per task:  {fmt_time(sum(draw_times) / len(draw_times))}", flush=True)
    if failures:
        print(f"  FAILURES:      {len(failures)}", flush=True)
        for loc, d, err in failures:
            print(f"    {loc} draw {d}: {err}", flush=True)
    print("=" * 60, flush=True)

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
