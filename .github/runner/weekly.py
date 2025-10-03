#!/usr/bin/env python3
"""
CFB weekly projection (single-file)
- Safe defaults and clamps to avoid crazy spreads.
- Scope: 'all', or 'topN' (e.g., 'top25', 'top10').
- Modes: FAST (fewer API hits), FULL (adds drives for field position).
- Writes: docs/week_preds.json, docs/_games_dbg.json, docs/_rankings_dbg.json, week_preds.csv
Deps: pandas, requests
"""

import os, re, json, time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import requests

# -------------------------------
# Tunables / limits
# -------------------------------
SCALE_PER_0P10 = 3.0   # spread scaling from PPA gap (per 0.10 PPA)
HFA_DEFAULT    = 2.0   # home field advantage
NEUTRAL_HFA    = 0.5

COMP_LIMITS = {
    "fp": 6.0, "hidden": 4.0, "xpl": 10.0, "sr": 6.0, "havoc": 6.0, "recency": 6.0
}
WEIGHTS = {k: 1.0 for k in COMP_LIMITS.keys()}

BASE_EPP = 0.42       # baseline pts/play (~55 over 130 plays)
TOTAL_FLOOR, TOTAL_CEIL = 30, 95
SPREAD_FLOOR, SPREAD_CEIL = -40.0, 40.0

# HTTP
BASE = "https://api.collegefootballdata.com"
TIMEOUT = 40
MAX_RETRIES = 5

# -------------------------------
# Env inputs from workflow
# -------------------------------
YEAR  = int(os.getenv("YEAR", "2025"))
WEEK  = int(os.getenv("WEEK", "6"))
SCOPE = (os.getenv("SCOPE", "all") or "all").lower()   # 'all', 'top25', 'top10'
MODE  = (os.getenv("MODE", "FAST") or "FAST").upper()  # 'FAST' or 'FULL'
API   = os.getenv("CFBD_API_KEY", "")

HEAD = {"Authorization": f"Bearer {API}"} if API else {}

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def log(msg: str) -> None:
    print(f"::notice::{msg}")

# polite, retrying GET
def jget(path: str, params: Dict[str, Any] = None) -> Any:
    url = f"{BASE}{path}"
    params = params or {}
    backoff = 1.5
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=HEAD, params=params, timeout=TIMEOUT)
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                delay = float(ra) if ra else backoff * attempt
                log(f"[429] sleeping {delay:.1f}s… {url}")
                time.sleep(delay)
                continue
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", 0)
            if 500 <= code < 600 and attempt < MAX_RETRIES:
                delay = backoff * attempt
                log(f"[{code}] retrying in {delay:.1f}s… {url}")
                time.sleep(delay); continue
            raise
        except requests.RequestException as e:
            if attempt < MAX_RETRIES:
                delay = backoff * attempt
                log(f"[net error] {e}; retrying in {delay:.1f}s… {url}")
                time.sleep(delay); continue
            raise

def parse_topN(scope: str) -> Optional[int]:
    if scope == "all":
        return None
    m = re.match(r"top(\d+)$", scope)
    if m:
        return int(m.group(1))
    return None

# --------------------------------
# Data fetch helpers
# --------------------------------
def fetch_games_resilient(year: int, week: int) -> pd.DataFrame:
    """Try a couple shapes CFBD uses, return DataFrame of games."""
    params = {"year": year, "week": week, "seasonType": "regular"}
    try:
        g = jget("/games", params)
        df = pd.DataFrame(g or [])
        if not df.empty:
            return df
    except Exception as e:
        log(f"/games fetch failed once: {e}")

    # second attempt: sometimes not including seasonType helps
    try:
        g = jget("/games", {"year": year, "week": week})
        df = pd.DataFrame(g or [])
        return df
    except Exception as e:
        log(f"/games second attempt failed: {e}")
        return pd.DataFrame()

def team_epp(off_ppa: float, opp_def_ppa: float) -> float:
    epp = BASE_EPP + (off_ppa - opp_def_ppa)
    return clamp(epp, 0.10, 0.80)

def pace_total(stats: Dict[str, Dict[str, float]], home: str, away: str) -> float:
    def ppg(m: Dict[str, float]) -> Optional[float]:
        plays = m.get("plays")
        g = m.get("games") or m.get("gp") or m.get("gms")
        try: return float(plays)/float(g) if plays and g else None
        except Exception: return None
    p_h = ppg(stats[home]["off"]); p_a = ppg(stats[away]["off"])
    return (p_h or 65.0) + (p_a or 65.0)

def pace_scale(x: float, total_plays: float, baseline: float = 130.0, elasticity: float = 0.5) -> float:
    return x * (1.0 + elasticity * ((total_plays - baseline) / baseline))

def fp_points(drives: Dict[str, Dict[str, float]], h: str, a: str, pts_per_yd: float = 0.06) -> float:
    exp_h = 0.5*drives[h]["osfp"] + 0.5*drives[a]["dsfp"]
    exp_a = 0.5*drives[a]["osfp"] + 0.5*drives[h]["dsfp"]
    return (exp_h - exp_a) * pts_per_yd

def hidden_yards(spc_h: Dict[str, float], spc_a: Dict[str, float], pts_per_yd: float = 0.055) -> float:
    net_h = spc_h.get("netpunting", 0) or (spc_h.get("puntyards",0)-spc_h.get("opponentpuntreturnyards",0))/max(1, spc_h.get("punts",1))
    net_a = spc_a.get("netpunting", 0) or (spc_a.get("puntyards",0)-spc_a.get("opponentpuntreturnyards",0))/max(1, spc_a.get("punts",1))
    ko_h = (spc_h.get("kickreturnyards",0)-spc_h.get("opponentkickreturnyards",0))/max(1, spc_h.get("kickreturns",1))
    ko_a = (spc_a.get("kickreturnyards",0)-spc_a.get("opponentkickreturnyards",0))/max(1, spc_a.get("kickreturns",1))
    return ((net_h - net_a) + 0.5*(ko_h - ko_a)) * pts_per_yd

def success_points(off_h: Dict[str,float], def_a: Dict[str,float],
                   off_a: Dict[str,float], def_h: Dict[str,float],
                   scale_per_5pct: float = 1.5) -> float:
    def sr(m: Dict[str,float]) -> Optional[float]:
        for k,v in m.items():
            k=str(k).lower()
            if "success" in k and "%" in k:
                try: return float(v)/100.0
                except: return None
        return None
    sh_off, sa_def = sr(off_h), sr(def_a)
    sa_off, sh_def = sr(off_a), sr(def_h)
    pts = 0.0
    if sh_off is not None and sa_def is not None: pts += ((sh_off-sa_def)/0.05)*scale_per_5pct
    if sa_off is not None and sh_def is not None: pts -= ((sa_off-sh_def)/0.05)*scale_per_5pct
    return pts

def explosiveness_points(off_map: Dict[str,float], def_map: Dict[str,float],
                         h: str, a: str, scale_per_0p10: float = SCALE_PER_0P10) -> float:
    oh, da = off_map.get(h,0.0), def_map.get(a,0.0)
    oa, dh = off_map.get(a,0.0), def_map.get(h,0.0)
    return ((oh-da)-(oa-dh))/0.10 * scale_per_0p10

def havoc_points(off_h: Dict[str,float], def_h: Dict[str,float],
                 off_a: Dict[str,float], def_a: Dict[str,float], scale: float = 3.0) -> float:
    def rate(d: Dict[str,float], o: Dict[str,float]) -> float:
        tfl = (d.get("tacklesforloss",0) or d.get("tfl",0)) + (d.get("sacks",0) or 0)
        plays = d.get("plays",0) or 1
        sacks_allowed = o.get("sacksallowed",0) or 0
        return (tfl + sacks_allowed) / max(1, plays)
    try:    return (rate(def_h, off_a) - rate(def_a, off_h)) * scale
    except: return 0.0

def recency_points(team_rows: List[Dict[str,Any]], opp_rows: List[Dict[str,Any]],
                   week: int, n: int = 4, scale: float = 0.5) -> float:
    def last_pdpg(rows):
        vals = [(g.get("pointsFor"), g.get("pointsAgainst")) for g in rows if (g.get("week") or 99) < week]
        vals = vals[-n:]
        return sum((pf-pa) for pf,pa in vals)/len(vals) if vals else 0.0
    return (last_pdpg(team_rows) - last_pdpg(opp_rows)) * scale

# --------------------------------
# Main
# --------------------------------
def main() -> int:
    try:
        log(f"Inputs → YEAR:{YEAR} WEEK:{WEEK} SCOPE:{SCOPE} MODE:{MODE}")

        # 1) Games, with debug dump
        gdf = fetch_games_resilient(YEAR, WEEK)

        Path("docs").mkdir(parents=True, exist_ok=True)
        with open("docs/_games_dbg.json", "w", encoding="utf-8") as f:
            recs = gdf.to_dict("records") if not gdf.empty else []
            mini = [
                {
                    "homeTeam": r.get("homeTeam"),
                    "awayTeam": r.get("awayTeam"),
                    "week": r.get("week"),
                    "seasonType": r.get("seasonType"),
                    "startDate": r.get("startDate") or r.get("start_date")
                } for r in recs
            ]
            json.dump(mini, f, ensure_ascii=False, indent=2)

        if gdf.empty:
            log("No games returned after attempts. Writing empty outputs.")
            return write_outputs([], {"games": 0, "note": "no games"})

        # 2) Rankings filter (optional)
        topN = parse_topN(SCOPE)

        def _poll_is_ap(name: str) -> bool:
            if not name: return False
            n = name.lower()
            return ("ap" in n) or ("associated press" in n) or ("ap top" in n)

        def _topN_from_rankings(ranks_json, N: int):
            if not ranks_json: return []
            latest = ranks_json[-1]
            out=[]
            for poll in latest.get("polls", []):
                if _poll_is_ap(poll.get("poll", "")):
                    for r in poll.get("ranks", []):
                        team = r.get("school") or r.get("team")
                        try: rk = int(r.get("rank"))
                        except: rk = None
                        if team and rk and rk <= N:
                            out.append(team)
            return out

        if topN is not None:
            try:
                ranks = jget("/rankings", {"year": YEAR, "week": WEEK})
                top_list = _topN_from_rankings(ranks, topN)
                with open("docs/_rankings_dbg.json", "w", encoding="utf-8") as f:
                    json.dump({"topN": topN, "teams": top_list}, f, ensure_ascii=False, indent=2)
                if top_list:
                    before = len(gdf)
                    keep = set(top_list)
                    gdf = gdf[gdf["homeTeam"].isin(keep) | gdf["awayTeam"].isin(keep)]
                    log(f"Filtered via AP top{topN}: {before} → {len(gdf)}")
                else:
                    log("AP poll empty/unavailable; keeping ALL games (no filter).")
            except Exception as e:
                log(f"Rankings fetch failed ({e}); keeping ALL games")
        else:
            # Still produce a small rankings dbg file for transparency
            with open("docs/_rankings_dbg.json", "w", encoding="utf-8") as f:
                json.dump({"topN": None, "teams": []}, f, ensure_ascii=False, indent=2)

        if gdf.empty:
            log("Scope left zero games. Writing empty outputs.")
            return write_outputs([], {"games": 0, "note": "filtered to zero"})

        teams = sorted(set(gdf["homeTeam"]).union(gdf["awayTeam"]))

        # 3) PPA (season)
        ppa = jget("/ppa/teams", {"year": YEAR})
        off_map, def_map = {}, {}
        for row in ppa or []:
            t = row.get("team")
            off = ((row.get("offense") or {}).get("overall") or (row.get("offense") or {}).get("ppa") or 0.0) or 0.0
            deff= ((row.get("defense") or {}).get("overall") or (row.get("defense") or {}).get("ppa") or 0.0) or 0.0
            if t in teams:
                off_map[t]=float(off); def_map[t]=float(deff)

        # 4) Season stats (off/def/special)
        def team_cat(team: str, cat: str) -> Dict[str, float]:
            try:
                rows = jget("/stats/season", {"year": YEAR, "team": team, "category": cat})
                m={}
                for r in rows or []:
                    n=(r.get("statName") or r.get("stat_name") or "").lower()
                    v=r.get("statValue") or r.get("stat_value")
                    try: v=float(v)
                    except: pass
                    m[n]=v
                return m
            except Exception:
                return {}
        stats = {t: {"off": team_cat(t,"offense"),
                     "def": team_cat(t,"defense"),
                     "spc": team_cat(t,"special")} for t in teams}

        # 5) Drives → field position (optional FULL)
        drives = {t: {"osfp":25.0, "dsfp":25.0} for t in teams}
        if MODE == "FULL":
            for t in teams:
                try:
                    drv = jget("/drives", {"year": YEAR, "week": WEEK, "team": t})
                    vals=[100 - d.get("start_yards_to_goal") for d in drv if d.get("start_yards_to_goal") is not None]
                    drives[t]["osfp"]=sum(vals)/len(vals) if vals else 25.0
                except Exception: pass
            try:
                drv_all = jget("/drives", {"year": YEAR, "week": WEEK})
                per_def={t:[] for t in teams}
                for d in drv_all or []:
                    tt=d.get("defense")
                    if tt in per_def and d.get("start_yards_to_goal") is not None:
                        per_def[tt].append(100 - d["start_yards_to_goal"])
                for t,vals in per_def.items():
                    drives[t]["dsfp"]=sum(vals)/len(vals) if vals else 25.0
            except Exception: pass

        # 6) Vegas lines (avg)
        lines = jget("/lines", {"year": YEAR, "week": WEEK, "seasonType":"regular"})
        v_map={}
        for ln in lines or []:
            home,away = ln.get("homeTeam"), ln.get("awayTeam")
            for b in (ln.get("lines") or []):
                sp=b.get("spread"); to=b.get("overUnder")
                if home and away and (sp is not None or to is not None):
                    v_map.setdefault((home,away), []).append((sp,to))
        vegas={}
        for k,vals in v_map.items():
            s=[v for v,_ in vals if isinstance(v,(int,float))]
            t=[u for _,u in vals if isinstance(u,(int,float))]
            vegas[k]={"vegas_spread": round(sum(s)/len(s),1) if s else None,
                      "vegas_total":  round(sum(t)/len(t),1) if t else None}

        # 7) Recency cache
        rec_cache={}
        for t in teams:
            try:
                rec_cache[t]= jget("/games/teams", {"year": YEAR, "team": t, "seasonType":"regular"}) or []
            except Exception:
                rec_cache[t]=[]

        # 8) Compute rows
        out=[]
        for _, g in gdf.iterrows():
            h, a = g["homeTeam"], g["awayTeam"]
            hfa = NEUTRAL_HFA if bool(g.get("neutralSite")) else HFA_DEFAULT

            base = ((off_map.get(h,0.0)-def_map.get(a,0.0)) -
                    (off_map.get(a,0.0)-def_map.get(h,0.0))) / 0.10 * SCALE_PER_0P10 + hfa

            plays = pace_total(stats, h, a)

            xpl_raw = pace_scale(explosiveness_points(off_map, def_map, h, a), plays)
            sr_raw  = pace_scale(success_points(stats[h]["off"], stats[a]["def"],
                                                stats[a]["off"], stats[h]["def"]), plays)
            hv_raw  = pace_scale(havoc_points(stats[h]["off"], stats[h]["def"],
                                              stats[a]["off"], stats[a]["def"]), plays)
            fp_raw  = fp_points(drives, h, a) if MODE=="FULL" else 0.0
            hy_raw  = hidden_yards(stats[h]["spc"], stats[a]["spc"]) if MODE=="FULL" else 0.0
            rcy_raw = recency_points(rec_cache.get(h,[]), rec_cache.get(a,[]), WEEK)

            def cap(name,val): return clamp(val*WEIGHTS[name], -COMP_LIMITS[name], COMP_LIMITS[name])
            fp, hy = cap("fp",fp_raw), cap("hidden",hy_raw)
            xpl, sr = cap("xpl",xpl_raw), cap("sr",sr_raw)
            hv, rcy = cap("havoc",hv_raw), cap("recency",rcy_raw)

            adj = clamp(base + fp + hy + xpl + sr + hv + rcy, SPREAD_FLOOR, SPREAD_CEIL)

            epp_h = team_epp(off_map.get(h,0.0), def_map.get(a,0.0))
            epp_a = team_epp(off_map.get(a,0.0), def_map.get(h,0.0))
            total_pts = epp_h*(plays/2.0) + epp_a*(plays/2.0)
            total_pts += 0.5*(xpl + sr)
            total_pts = int(round(clamp(total_pts, TOTAL_FLOOR, TOTAL_CEIL)))

            home_pts = int(round((total_pts + adj)/2))
            away_pts = int(round(total_pts - home_pts))
            home_pts = max(0, home_pts); away_pts = max(0, away_pts)
            favored = h if adj >= 0 else a

            v = vegas.get((h,a), {})
            out.append({
                "home":h,"away":a,"favored":favored,
                "base_spread": round(base,1),
                "adj_spread":  round(adj,1),
                "home_pts": home_pts, "away_pts": away_pts,
                "total_pts": int(total_pts),
                "vegas_spread": v.get("vegas_spread"),
                "vegas_total":  v.get("vegas_total"),
                "plays_est": int(round(plays)),
                "fp": round(fp,2), "hidden": round(hy,2),
                "xpl": round(xpl,2), "sr": round(sr,2),
                "havoc": round(hv,2), "recency": round(rcy,2)
            })

        out = sorted(out, key=lambda r: abs(r["adj_spread"]), reverse=True)
        return write_outputs(out, {"games": len(out)})

    except Exception as e:
        log(f"ERROR: {e}")
        return write_outputs([], {"error": str(e)})

def write_outputs(rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> int:
    Path("docs").mkdir(parents=True, exist_ok=True)
    with open("docs/week_preds.json","w",encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    pd.DataFrame(rows).to_csv("week_preds.csv", index=False)
    with open("docs/_meta_dbg.json","w",encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"Wrote {len(rows)} rows → docs/week_preds.json and week_preds.csv")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())