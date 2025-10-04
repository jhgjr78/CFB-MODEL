#!/usr/bin/env python3
"""
Build a local season dataset for CFB with polite, week-by-week pulls + caching.

Writes:
- data/raw/{YEAR}/games_w{W}.json          (one per week)
- data/raw/{YEAR}/lines_w{W}.json          (one per week)
- data/raw/{YEAR}/drives_w{W}.json         (optional, if MODE=FULL)
- data/raw/{YEAR}/ppa_teams.json           (once per season)
- data/raw/{YEAR}/stats_{team}.json        (off/def/special merged; one per team used)
- data/season_{YEAR}_games.csv             (normalized, one big CSV for modeling)

Env inputs (via workflow):
  YEAR      (e.g., 2025)            default: 2025
  WEEK_TO   (highest week to fetch) default: 6
  SCOPE     (all, top25, top10…)    default: all
  MODE      (FULL or FAST)          default: FULL
  CFBD_API_KEY (secret)

Dependencies: requests, pandas
"""

import os, json, time, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# --------------------------- Tunables ---------------------------
BASE = "https://api.collegefootballdata.com"
TIMEOUT = 40
MAX_RETRIES = 6
BACKOFF_SECS = 1.5

# --------------------------- Inputs -----------------------------
YEAR    = int(os.getenv("YEAR", "2025"))
WEEK_TO = int(os.getenv("WEEK_TO", "6"))
SCOPE   = (os.getenv("SCOPE", "all") or "all").lower()
MODE    = (os.getenv("MODE", "FULL") or "FULL").upper()
API     = os.getenv("CFBD_API_KEY", "")

HEAD = {"Authorization": f"Bearer {API}"} if API else {}

# --------------------------- Utils ------------------------------
def log(msg: str) -> None:
    print(f"##[notice]{msg}")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def load_json_if_exists(path: Path) -> Optional[Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def jget(path: str, params: Dict[str, Any]) -> Any:
    """GET with retry/backoff for 429 and transient 5xx."""
    url = f"{BASE}{path}"
    for attempt in range(1, MAX_RETRIES + 1):
        r = requests.get(url, headers=HEAD, params=params, timeout=TIMEOUT)
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            delay = float(ra) if ra else BACKOFF_SECS * attempt
            log(f"[429] {url} — sleeping {delay:.1f}s then retry...")
            time.sleep(delay)
            continue
        if 500 <= r.status_code < 600:
            delay = BACKOFF_SECS * attempt
            log(f"[{r.status_code}] {url} — sleeping {delay:.1f}s then retry...")
            time.sleep(delay)
            continue
        r.raise_for_status()
        return r.json()
    # last try already raised or we'll raise explicitly:
    r.raise_for_status()

def parse_topN(scope: str) -> Optional[int]:
    if scope == "all":
        return None
    m = re.match(r"top(\d+)$", scope)
    return int(m.group(1)) if m else None

def poll_is_ap(name: str) -> bool:
    n = (name or "").lower()
    return ("ap" in n) or ("associated press" in n)

def get_topN_ap_teams(year: int, week_to: int, topN: int) -> List[str]:
    """
    Pull rankings up through WEEK_TO and return a union of AP topN team names.
    (Using union avoids missing teams that moved in/out.)
    """
    teams = set()
    for wk in range(1, week_to + 1):
        try:
            ranks = jget("/rankings", {"year": year, "week": wk})
        except Exception:
            continue
        if not ranks:
            continue
        latest = ranks[-1]
        for poll in latest.get("polls", []):
            if poll_is_ap(poll.get("poll")):
                for r in poll.get("ranks", []):
                    try:
                        rk = int(r.get("rank"))
                    except Exception:
                        rk = None
                    if rk and rk <= topN:
                        t = r.get("school") or r.get("team")
                        if t:
                            teams.add(t)
    return sorted(teams)

# --------------------------- Main -------------------------------
def main() -> int:
    log(f"Dataset build → YEAR:{YEAR} WEEK_TO:{WEEK_TO} SCOPE:{SCOPE} MODE:{MODE}")

    root_raw = Path("data/raw") / str(YEAR)
    ensure_dir(root_raw)

    # 0) Determine scope list (optional)
    scope_list: Optional[List[str]] = None
    topN = parse_topN(SCOPE)
    if topN:
        log(f"Resolving AP Top{topN} teams up to week {WEEK_TO}…")
        scope_list = get_topN_ap_teams(YEAR, WEEK_TO, topN)
        log(f"AP Top{topN} team-count (union): {len(scope_list)}")
        # small guard to avoid empty scope causing empty season
        if not scope_list:
            log("AP scope resolution returned 0 teams; continuing with ALL.")
            scope_list = None

    # 1) One-time pulls (PPA teams)
    ppa_path = root_raw / "ppa_teams.json"
    ppa = load_json_if_exists(ppa_path)
    if ppa is None:
        log("Pulling season PPA (teams)…")
        ppa = jget("/ppa/teams", {"year": YEAR})
        save_json(ppa_path, ppa)
    else:
        log("Using cached PPA (teams).")

    # Build a quick set of candidate teams (from scope or from PPA)
    candidate_teams = set()
    if scope_list:
        candidate_teams.update(scope_list)
    else:
        for r in ppa or []:
            t = r.get("team")
            if t:
                candidate_teams.add(t)

    # 2) Week-by-week pulls (games, lines, drives)
    season_rows: List[Dict[str, Any]] = []
    for wk in range(1, WEEK_TO + 1):
        log(f"Week {wk}/{WEEK_TO} — fetching games…")

        g_path = root_raw / f"games_w{wk}.json"
        games = load_json_if_exists(g_path)
        if games is None:
            # Pull ALL games, then (optionally) filter by scope after
            try:
                games = jget("/games", {"year": YEAR, "week": wk, "seasonType": "regular"})
            except Exception as e:
                log(f"WARNING: /games week {wk} failed: {e}")
                games = []
            save_json(g_path, games)
        else:
            log("  using cached games.")

        if not games:
            log("  no games that week.")
            continue

        # Optional scope filter (homeTeam/awayTeam in scope_list)
        if scope_list:
            games = [g for g in games if (g.get("homeTeam") in scope_list or g.get("awayTeam") in scope_list)]

        # Save “lines” per week (averaged later in your weekly runner)
        ln_path = root_raw / f"lines_w{wk}.json"
        lines = load_json_if_exists(ln_path)
        if lines is None:
            try:
                lines = jget("/lines", {"year": YEAR, "week": wk, "seasonType": "regular"})
            except Exception as e:
                log(f"  WARNING: /lines week {wk} failed: {e}")
                lines = []
            save_json(ln_path, lines)
        else:
            log("  using cached lines.")

        # Drives (optional; MODE == FULL)
        if MODE == "FULL":
            d_path = root_raw / f"drives_w{wk}.json"
            drives = load_json_if_exists(d_path)
            if drives is None:
                try:
                    drives = jget("/drives", {"year": YEAR, "week": wk})
                except Exception as e:
                    log(f"  WARNING: /drives week {wk} failed: {e}")
                    drives = []
                save_json(d_path, drives)
            else:
                log("  using cached drives.")

        # Normalize a minimal row for season CSV (one per game)
        for g in games:
            season_rows.append({
                "year": YEAR,
                "week": wk,
                "seasonType": g.get("seasonType"),
                "homeTeam": g.get("homeTeam"),
                "awayTeam": g.get("awayTeam"),
                "homeConference": g.get("homeConference"),
                "awayConference": g.get("awayConference"),
                "homePoints": g.get("homePoints"),
                "awayPoints": g.get("awayPoints"),
                "neutralSite": bool(g.get("neutralSite")),
                "venue": g.get("venue")
            })

        # polite spacing between weeks to avoid burst rate limits
        time.sleep(1.2)

    # 3) Per-team season stats (off/def/special), cached per team
    stats_cache_dir = root_raw / "team_stats"
    ensure_dir(stats_cache_dir)

    for t in sorted(candidate_teams):
        spath = stats_cache_dir / f"stats_{t}.json"
        if spath.exists():
            continue
        log(f"Pulling season stats for {t}…")
        allcats: Dict[str, Any] = {}
        for cat in ("offense", "defense", "special"):
            try:
                rows = jget("/stats/season", {"year": YEAR, "team": t, "category": cat})
            except Exception as e:
                log(f"  WARNING: /stats/season {t} {cat} failed: {e}")
                rows = []
            # normalize into {stat_name: value}
            m: Dict[str, Any] = {}
            for r in rows or []:
                name = (r.get("statName") or r.get("stat_name") or "").lower()
                val = r.get("statValue") or r.get("stat_value")
                try: val = float(val)
                except Exception: pass
                m[name] = val
            allcats[cat] = m
            time.sleep(0.4)  # gentle spacing between category calls
        save_json(spath, allcats)
        time.sleep(0.6)      # spacing between teams

    # 4) Write season CSV
    out_csv = Path("data") / f"season_{YEAR}_games.csv"
    ensure_dir(out_csv.parent)
    pd.DataFrame(season_rows).to_csv(out_csv, index=False)

    log(f"Done. Weeks cached: 1..{WEEK_TO}")
    log(f"Games rows written: {len(season_rows)} → {out_csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())