#!/usr/bin/env python3
"""
Mini Loader: cache-only fetches for ONE week.
- Writes under: data/weeks/<YEAR>/week_<WEEK>.(games|rankings).json
- Also writes/refreshes a season-level file for PPA:
  data/season_<YEAR>/ppa_teams.json
This keeps API calls low and lets weekly.py read from cache first.

Inputs (env):
  CFBD_API_KEY, YEAR, WEEK, SCOPE
"""

import os, json, time
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests

BASE = "https://api.collegefootballdata.com"
TIMEOUT = 35
MAX_RETRIES = 6

YEAR = int(os.getenv("YEAR","2025"))
WEEK = int(os.getenv("WEEK","6"))
SCOPE = (os.getenv("SCOPE","top25") or "top25").lower()
API   = os.getenv("CFBD_API_KEY","")
HEAD  = {"Authorization": f"Bearer {API}"} if API else {}

def log(msg: str) -> None:
    print(f"##[notice]{msg}")

def jget(path: str, params: Dict[str, Any]=None) -> Any:
    """requests.get with polite backoff (429/5xx)."""
    url = f"{BASE}{path}"
    params = params or {}
    for i in range(1, MAX_RETRIES+1):
        r = requests.get(url, headers=HEAD, params=params, timeout=TIMEOUT)
        if r.status_code == 429:
            delay = (r.headers.get("Retry-After") and float(r.headers["Retry-After"])) or 1.5*i
            log(f"[429] {path} – sleeping {delay:.1f}s then retry…")
            time.sleep(delay); continue
        if 500 <= r.status_code < 600:
            if i < MAX_RETRIES:
                delay = 1.5*i
                log(f"[{r.status_code}] {path} – sleeping {delay:.1f}s then retry…")
                time.sleep(delay); continue
        r.raise_for_status()
        return r.json()

def safe_write(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def main() -> int:
    log(f"Mini Loader start → YEAR={YEAR} WEEK={WEEK} SCOPE={SCOPE}")

    week_dir = Path(f"data/weeks/{YEAR}")
    season_dir = Path(f"data/season_{YEAR}")
    week_games_fp    = week_dir / f"week_{WEEK}.games.json"
    week_rankings_fp = week_dir / f"week_{WEEK}.rankings.json"
    season_ppa_fp    = season_dir / "ppa_teams.json"

    # 1) Games (one call)
    try:
        games = jget("/games", {"year": YEAR, "week": WEEK, "seasonType":"regular"}) or []
        safe_write(week_games_fp, games)
        log(f"Saved games → {week_games_fp} ({len(games)} rows)")
    except Exception as e:
        log(f"WARNING: games fetch failed: {e}")
        if not week_games_fp.exists(): safe_write(week_games_fp, [])

    # 2) Rankings snapshot (one call). Useful for topN filters.
    try:
        ranks = jget("/rankings", {"year": YEAR, "week": WEEK}) or []
        safe_write(week_rankings_fp, ranks)
        log(f"Saved rankings → {week_rankings_fp} (snapshots:{len(ranks)})")
    except Exception as e:
        log(f"WARNING: rankings fetch failed: {e}")
        if not week_rankings_fp.exists(): safe_write(week_rankings_fp, [])

    # 3) Season PPA for all teams (one call; we'll filter later)
    #    We refresh if file missing or zero length.
    need_ppa = (not season_ppa_fp.exists()) or (season_ppa_fp.stat().st_size < 5)
    if need_ppa:
        try:
            ppa = jget("/ppa/teams", {"year": YEAR}) or []
            safe_write(season_ppa_fp, ppa)
            log(f"Saved PPA → {season_ppa_fp} (teams:{len(ppa)})")
        except Exception as e:
            log(f"WARNING: PPA fetch failed: {e}")
            if not season_ppa_fp.exists(): safe_write(season_ppa_fp, [])

    log("Mini Loader done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())