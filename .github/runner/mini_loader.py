#!/usr/bin/env python3
import os, json, time
from pathlib import Path
import requests

BASE = "https://api.collegefootballdata.com"
API  = os.getenv("CFBD_API_KEY", "")
HEAD = {"Authorization": f"Bearer {API}"} if API else {}
TIMEOUT = 40
MAX_RETRIES = 6

YEAR = int(os.getenv("YEAR", "2025"))
WEEK = int(os.getenv("WEEK", "6"))
SCOPE = (os.getenv("SCOPE", "all") or "all").lower()

def log(msg): print(f"::notice::{msg}")

def jget(path, params):
    url = f"{BASE}{path}"
    backoff = 1.5
    for i in range(1, MAX_RETRIES+1):
        r = requests.get(url, headers=HEAD, params=params, timeout=TIMEOUT)
        if r.status_code == 429 and i < MAX_RETRIES:
            delay = backoff * i if not r.headers.get("Retry-After") else float(r.headers["Retry-After"])
            log(f"[429] {path} – sleeping {delay:.1f}s then retry…")
            time.sleep(delay); continue
        try:
            r.raise_for_status()
            return r.json()
        except requests.HTTPError:
            if 500 <= r.status_code < 600 and i < MAX_RETRIES:
                delay = backoff * i
                log(f"[{r.status_code}] {path} – sleeping {delay:.1f}s then retry…")
                time.sleep(delay); continue
            raise

def _safe_write(path: Path, payload, describe: str):
    """Only overwrite if payload is non-empty."""
    if isinstance(payload, list):
        n = len(payload)
    elif isinstance(payload, dict):
        # treat rankings snapshots count
        if "polls" in payload or "rankings" in payload:
            n = 1
        else:
            n = len(payload)
    else:
        n = 0
    if n and not (isinstance(payload, list) and n == 0):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False))
        log(f"Saved {describe} → {path} ({n} rows/snapshots)")
    else:
        log(f"Kept old {describe} – fetched empty, not overwriting: {path}")

def main():
    log(f"Mini Loader start → YEAR={YEAR} WEEK={WEEK} SCOPE={SCOPE}")

    # Paths
    p_games    = Path(f"data/weeks/{YEAR}/week_{WEEK}.games.json")
    p_rankings = Path(f"data/weeks/{YEAR}/week_{WEEK}.rankings.json")
    p_ppa      = Path(f"data/season_{YEAR}/ppa_teams.json")

    # Games (this week)
    try:
        games = jget("/games", {"year": YEAR, "week": WEEK, "seasonType": "regular"}) or []
    except Exception:
        games = []
    _safe_write(p_games, games, "games")

    # Rankings (AP snapshots list)
    try:
        rankings = jget("/rankings", {"year": YEAR, "week": WEEK}) or []
    except Exception:
        rankings = []
    _safe_write(p_rankings, rankings, "rankings")

    # PPA (season)
    try:
        ppa = jget("/ppa/teams", {"year": YEAR}) or []
    except Exception:
        ppa = []
    _safe_write(p_ppa, ppa, "PPA")

    log("Mini Loader done.")

if __name__ == "__main__":
    main()