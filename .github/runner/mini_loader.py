#!/usr/bin/env python3
"""
mini_loader.py
Cached GET wrapper for CFBD API to avoid 429s.
- Stores responses under ./cache (keyed by path+params)
- Reuses cached data for ttl_hours (default 24h)
- Polite retry/backoff on 429/5xx
"""

import os, json, time, hashlib
from pathlib import Path
import requests

BASE = "https://api.collegefootballdata.com"
TIMEOUT = 40
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

HEADERS = {"Authorization": f"Bearer {os.getenv('CFBD_API_KEY', '')}"}

def _cache_key(path: str, params: dict) -> str:
    ident = f"{path}?{json.dumps(params or {}, sort_keys=True)}"
    return hashlib.md5(ident.encode("utf-8")).hexdigest()[:20]

def jget(path: str, params=None, ttl_hours: float = 24.0):
    """
    Cached GET with retry. Returns parsed JSON or [] on failure.
    """
    params = params or {}
    key = _cache_key(path, params)
    fp = CACHE_DIR / f"{key}.json"

    # Serve fresh-enough cache
    if fp.exists():
        age = time.time() - fp.stat().st_mtime
        if age < ttl_hours * 3600:
            try:
                with fp.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass  # fall through to refetch

    backoff = 2.0
    for attempt in range(6):
        try:
            r = requests.get(f"{BASE}{path}", headers=HEADERS, params=params, timeout=TIMEOUT)
            if r.status_code == 429:
                wait = (attempt + 1) * backoff
                print(f"##[notice][429] {path} – sleeping {wait:.1f}s then retry…")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            try:
                with fp.open("w", encoding="utf-8") as f:
                    json.dump(data, f)
            except Exception:
                pass
            return data
        except requests.RequestException as e:
            wait = (attempt + 1) * backoff
            print(f"##[notice][net] {e} – sleeping {wait:.1f}s then retry…")
            time.sleep(wait)

    print(f"##[warning] Giving up on {path} – returning []")
    return []