#!/usr/bin/env python3
"""
build_dataset.py
Builds a local, reusable cache for a season under ./data/season_<YEAR>/
Fetches (via mini_loader cache):
  - games (regular season, all weeks)
  - lines (all season)
  - rankings (all year)
  - team PPA (season snapshot)
This drastically reduces calls in weekly runs.
"""

import os, json
from pathlib import Path
from mini_loader import jget

YEAR = int(os.getenv("YEAR", "2025"))

OUT_DIR = Path(f"data/season_{YEAR}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f)

def main():
    print(f"##[notice]Building local dataset for YEAR={YEAR}")

    # 1) Games (all regular-season weeks)
    print("##[notice]1/4 Pulling season games…")
    games = jget("/games", {"year": YEAR, "seasonType": "regular"}, ttl_hours=24)
    write_json(OUT_DIR / "games.json", games)
    print(f"##[notice]   games: {len(games)}")

    # 2) Lines (entire season)
    print("##[notice]2/4 Pulling season lines…")
    lines = jget("/lines", {"year": YEAR, "seasonType": "regular"}, ttl_hours=24)
    write_json(OUT_DIR / "lines.json", lines)
    print(f"##[notice]   books: {len(lines)} weeks worth (raw records)")

    # 3) Rankings (AP throughout season)
    print("##[notice]3/4 Pulling rankings…")
    rankings = jget("/rankings", {"year": YEAR}, ttl_hours=24)
    write_json(OUT_DIR / "rankings.json", rankings)
    print(f"##[notice]   ranking snapshots: {len(rankings)}")

    # 4) Team PPA snapshot
    print("##[notice]4/4 Pulling team PPA…")
    ppa = jget("/ppa/teams", {"year": YEAR}, ttl_hours=24)
    write_json(OUT_DIR / "ppa_teams.json", ppa)
    print(f"##[notice]   teams in PPA: {len(ppa)}")

    # meta
    write_json(OUT_DIR / "_meta.json", {"year": YEAR})
    print("##[notice]Done. Cached season assets under:", OUT_DIR)

if __name__ == "__main__":
    raise SystemExit(main())