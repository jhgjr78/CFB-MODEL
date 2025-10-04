#!/usr/bin/env python3
# .github/runner/mini_loader.py
import argparse, json, os, time, random
from pathlib import Path
import requests

# ---------- CONFIG ----------
BASE = "https://api.collegefootballdata.com"
HFA_DEFAULT = 2.0  # not used here but kept for parity
HEADERS = lambda: {"Authorization": f"Bearer {os.environ.get('CFBD_API_KEY','')}"}
RETRY_MAX = 8
RETRY_BASE = 1.5
JITTER = (0.25, 0.75)

# Files we cache
def week_dir(year): return Path(f"data/weeks/{year}")
def season_dir(year): return Path(f"data/season_{year}")

def safe_write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite with empty
    if payload is None: 
        return
    if isinstance(payload, (list, dict)) and len(payload) == 0:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

def cached(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return None
    return None

def backoff_sleep(attempt):
    base = RETRY_BASE ** attempt
    jitter = random.uniform(*JITTER)
    time.sleep(base + jitter)

def fetch_json(endpoint, params):
    url = f"{BASE}{endpoint}"
    for attempt in range(RETRY_MAX):
        try:
            r = requests.get(url, headers=HEADERS(), params=params, timeout=30)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    # Sometimes a blank body with 200 (rare) — treat as retry
                    pass
            elif r.status_code in (429, 500, 502, 503, 504):
                # Polite backoff
                backoff_sleep(attempt)
            else:
                # Other errors: short backoff, keep trying
                backoff_sleep(attempt)
        except requests.RequestException:
            backoff_sleep(attempt)
    return None

def ensure_week(year, week, mode):
    wdir = week_dir(year)
    sdir = season_dir(year)

    # paths
    games_p = wdir / f"week_{week}.games.json"
    ranks_p = wdir / f"week_{week}.rankings.json"
    ppa_p   = sdir / "ppa_teams.json"
    drives_p = wdir / f"week_{week}.drives.json"
    lines_p  = wdir / f"week_{week}.lines.json"

    # GAMES
    if not games_p.exists():
        games = fetch_json("/games", {"year": year, "week": week, "seasonType": "regular"})
        safe_write(games_p, games)
    # RANKINGS
    if not ranks_p.exists():
        rankings = fetch_json("/rankings", {"year": year, "week": week})
        safe_write(ranks_p, rankings)
    # PPA TEAMS (season-level)
    if not ppa_p.exists():
        ppa = fetch_json("/ppa/teams", {"year": year})
        safe_write(ppa_p, ppa)

    if mode.upper() == "FULL":
        # DRIVES
        if not drives_p.exists():
            drives = fetch_json("/drives", {"year": year, "week": week})
            safe_write(drives_p, drives)
        # LINES
        if not lines_p.exists():
            lines = fetch_json("/lines", {"year": year, "week": week})
            safe_write(lines_p, lines)

    # final sanity: do not overwrite existing files with empties; done by safe_write
    # Print quick summary for logs
    for p in [games_p, ranks_p, ppa_p, drives_p if mode.upper()=="FULL" else None, lines_p if mode.upper()=="FULL" else None]:
        if p and p.exists(): 
            print(f"[cache ok] {p}")
        elif p:
            print(f"[cache miss] {p}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", required=True, type=int)
    ap.add_argument("--week", required=True, type=int)
    ap.add_argument("--mode", default="FAST", choices=["FAST", "FULL"])
    args = ap.parse_args()
    if not os.environ.get("CFBD_API_KEY"):
        print("WARNING: CFBD_API_KEY not set; live calls will fail if cache is empty.")
    ensure_week(args.year, args.week, args.mode)

if __name__ == "__main__":
    main()