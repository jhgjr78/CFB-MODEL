#!/usr/bin/env python3
import os, json, time, random
from pathlib import Path
import requests

BASE = "https://api.collegefootballdata.com"
API  = os.getenv("CFBD_API_KEY", "")
HEAD = {"Authorization": f"Bearer {API}"} if API else {}
TIMEOUT = 40
MAX_RETRIES = 20            # was 6
MAX_SLEEP   = 15.0          # cap per backoff step

YEAR = int(os.getenv("YEAR", "2025"))
WEEK = int(os.getenv("WEEK", "6"))
SCOPE = (os.getenv("SCOPE", "all") or "all").lower()

def log(msg): print(f"::notice::{msg}")

def _sleep(s: float):
    s = min(MAX_SLEEP, max(0.5, s))
    time.sleep(s)

def jget(path, params):
    """Retry politely; respect Retry-After; add jitter."""
    url = f"{BASE}{path}"
    for i in range(1, MAX_RETRIES + 1):
        r = requests.get(url, headers=HEAD, params=params, timeout=TIMEOUT)
        if r.status_code == 429 and i < MAX_RETRIES:
            ra = r.headers.get("Retry-After")
            base = float(ra) if ra else 1.5 * i
            delay = base + random.uniform(0, 0.75 * base)
            log(f"[429] {path} – sleeping {delay:.1f}s then retry…")
            _sleep(delay); continue
        try:
            r.raise_for_status()
            return r.json()
        except requests.HTTPError:
            if 500 <= r.status_code < 600 and i < MAX_RETRIES:
                base = 1.5 * i
                delay = base + random.uniform(0, 0.5 * base)
                log(f"[{r.status_code}] {path} – sleeping {delay:.1f}s then retry…")
                _sleep(delay); continue
            raise

def _safe_write(path: Path, payload, describe: str):
    """Only overwrite if payload is non-empty."""
    n = 0
    if isinstance(payload, list):
        n = len(payload)
    elif isinstance(payload, dict):
        # treat “non-empty” conservatively
        n = len(payload) or (1 if ("polls" in payload or "rankings" in payload) else 0)
    if n > 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False))
        log(f"Saved {describe} → {path} ({n} rows/snapshots)")
    else:
        log(f"Kept old {describe} – fetched empty, not overwriting: {path}")

def main():
    # Stagger start a bit so we don't collide with everyone else
    jitter = random.uniform(2.0, 12.0)
    log(f"Mini Loader start → YEAR={YEAR} WEEK={WEEK} SCOPE={SCOPE} (start jitter {jitter:.1f}s)")
    _sleep(jitter)

    p_games    = Path(f"data/weeks/{YEAR}/week_{WEEK}.games.json")
    p_rankings = Path(f"data/weeks/{YEAR}/week_{WEEK}.rankings.json")
    p_ppa      = Path(f"data/season_{YEAR}/ppa_teams.json")

    try:
        games = jget("/games", {"year": YEAR, "week": WEEK, "seasonType": "regular"}) or []
    except Exception:
        games = []
    _safe_write(p_games, games, "games")

    try:
        rankings = jget("/rankings", {"year": YEAR, "week": WEEK}) or []
    except Exception:
        rankings = []
    _safe_write(p_rankings, rankings, "rankings")

    try:
        ppa = jget("/ppa/teams", {"year": YEAR}) or []
    except Exception:
        ppa = []
    _safe_write(p_ppa, ppa, "PPA")

    log("Mini Loader done.")

if __name__ == "__main__":
    main()