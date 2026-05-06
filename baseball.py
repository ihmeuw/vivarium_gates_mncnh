import argparse


PLAYER_STATS = {
    "shohei_ohtani": {
        "batting_average": ".274",
        "home_runs": "225",
        "rbi": "600",
        "stolen_bases": "112",
        "ops": ".942",
    },
    "cal_raleigh": {
        "batting_average": ".232",
        "home_runs": "128",
        "rbi": "310",
        "stolen_bases": "5",
        "ops": ".800",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Print basic baseball statistics.")
    parser.add_argument("player_name", help="Player name (e.g. cal_raleigh, shohei_ohtani)")
    parser.add_argument(
        "--stat-category",
        default="batting_average",
        help="Stat category (default: batting_average)",
    )
    args = parser.parse_args()

    player = args.player_name.lower()
    category = args.stat_category.lower()

    if player not in PLAYER_STATS:
        print(f"Unknown player: {args.player_name}")
        print(f"Available players: {', '.join(PLAYER_STATS.keys())}")
        return

    stats = PLAYER_STATS[player]
    if category not in stats:
        print(f"Unknown stat category: {args.stat_category}")
        print(f"Available categories: {', '.join(stats.keys())}")
        return

    print(f"{args.player_name} - {args.stat_category}: {stats[category]}")


if __name__ == "__main__":
    main()
