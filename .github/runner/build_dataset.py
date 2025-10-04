#!/usr/bin/env python3
"""
Build one local season dataset (bulk-only CFBD pulls) and write Parquet files:
data/<YEAR>/
  games.parquet
  lines.parquet
  ppa.parquet
  stats_off.parquet
  stats_def.parquet
  stats_spc.parquet
  drives_week<W>.parquet   (optional for requested weeks)

Env:
  CFBD_API_KEY (required)
  YEAR=2025
  DRIVE_WEEKS="1-6,8"  # optional: comma/range list
"""
import os, time
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import requests

BASE = "https://api.collegefootballdata.com"
HEAD = {"Authorization": f"Bearer {os.getenv('CFBD_API_KEY','')}"}
YEAR = int(os.getenv("YEAR","2025"))
DRIVE_WEEKS = os.getenv("DRIVE_WEEKS","").strip()

TIMEOUT=45
RETRIES=4

def log(msg): print(f"::notice::{msg}")

def jget(path:str, params:Dict[str,Any]|None=None)->Any:
    params=params or {}
    url=f"{BASE}{path}"
    back=1.7
    for a in range(1,RETRIES+1):
        r=requests.get(url, headers=HEAD, params=params, timeout=TIMEOUT)
        if r.status_code==429 and a<RETRIES:
            delay=float(r.headers.get("Retry-After") or back*a)
            log(f"[429] {url} — sleeping {delay:.1f}s")
            time.sleep(delay); continue
        r.raise_for_status(); return r.json()

def parse_weeks(expr:str)->List[int]:
    if not expr: return []
    out=[]
    for tok in expr.split(","):
        tok=tok.strip()
        if "-" in tok:
            s,e=tok.split("-",1)
            out += list(range(int(s), int(e)+1))
        elif tok:
            out.append(int(tok))
    return sorted(set(out))

def ensure_dir(p:Path)->None: p.mkdir(parents=True, exist_ok=True)
def to_parquet(df:pd.DataFrame, path:Path)->None:
    ensure_dir(path.parent); (df if df is not None else pd.DataFrame()).to_parquet(path, index=False)

def main():
    if not HEAD.get("Authorization"):
        raise SystemExit("CFBD_API_KEY missing")

    base_dir = Path("data")/str(YEAR)
    ensure_dir(base_dir)
    log(f"Build dataset YEAR={YEAR}")

    # 1) Games (whole season)
    log("#1 /games (season)")
    games = jget("/games", {"year": YEAR, "seasonType":"regular"}) or []
    pd.json_normalize(games).to_parquet(base_dir/"games.parquet", index=False)

    # 2) Lines (whole season)
    log("#2 /lines (season)")
    lines = jget("/lines", {"year": YEAR, "seasonType":"regular"}) or []
    recs=[]
    for g in lines:
        for ln in (g.get("lines") or []):
            rec = {k:g.get(k) for k in ["id","season","seasonType","week","homeTeam","awayTeam","neutralSite","startDate"]}
            rec.update({"provider": ln.get("provider"), "spread": ln.get("spread"), "overUnder": ln.get("overUnder")})
            recs.append(rec)
    pd.DataFrame(recs).to_parquet(base_dir/"lines.parquet", index=False)

    # 3) PPA teams (season)
    log("#3 /ppa/teams")
    ppa = jget("/ppa/teams", {"year": YEAR}) or []
    pd.json_normalize(ppa).to_parquet(base_dir/"ppa.parquet", index=False)

    # 4) Season stats — bulk 3 calls
    log("#4 /stats/season offense/defense/special (bulk)")
    def bulk_stats(cat):
        rows = jget("/stats/season", {"year": YEAR, "category": cat}) or []
        return pd.json_normalize(rows)
    bulk_stats("offense").to_parquet(base_dir/"stats_off.parquet", index=False)
    bulk_stats("defense").to_parquet(base_dir/"stats_def.parquet", index=False)
    bulk_stats("special").to_parquet(base_dir/"stats_spc.parquet", index=False)

    # 5) Drives for selected weeks (optional)
    weeks = parse_weeks(DRIVE_WEEKS)
    for w in weeks:
        log(f"#5 /drives week={w}")
        drv = jget("/drives", {"year": YEAR, "week": w}) or []
        pd.json_normalize(drv).to_parquet(base_dir/f"drives_week{w}.parquet", index=False)

    log("Done. Parquet files written under data/<YEAR>/")
    return 0

if __name__=="__main__":
    raise SystemExit(main())