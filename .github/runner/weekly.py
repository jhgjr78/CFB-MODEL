#!/usr/bin/env python3
"""
weekly.py
Produces docs/week_preds.json and week_preds.csv using:
- Cached season data from data/season_<YEAR>/
- Minimal live calls (stats by team; optional drives)
- Recency computed from cached games (no extra API)
"""

import os, re, json, math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

from mini_loader import jget

# -------------------------------
# #0 Inputs
# -------------------------------
YEAR  = int(os.getenv("YEAR", "2025"))
WEEK  = int(os.getenv("WEEK", "6"))
SCOPE = (os.getenv("SCOPE", "top25") or "top25").lower()  # 'all', 'top25', 'top10', etc.
MODE  = (os.getenv("MODE", "FULL") or "FULL").upper()

DATA_DIR = Path(f"data/season_{YEAR}")
DOCS = Path("docs"); DOCS.mkdir(exist_ok=True)

# -------------------------------
# #1 Tunables
# -------------------------------
SCALE_PER_0P10 = 3.0
HFA_DEFAULT    = 2.0
NEUTRAL_HFA    = 0.5

COMP_LIMITS = {"fp":6.0, "hidden":4.0, "xpl":10.0, "sr":6.0, "havoc":6.0, "recency":6.0}
WEIGHTS = {k:1.0 for k in COMP_LIMITS}
BASE_EPP = 0.42
TOTAL_FLOOR, TOTAL_CEIL = 30, 95
SPREAD_FLOOR, SPREAD_CEIL = -40.0, 40.0

# -------------------------------
# helpers
# -------------------------------
def clamp(x, lo, hi): return max(lo, min(hi, x))
def log(m): print(f"##[notice]{m}")

def parse_topN(scope: str) -> Optional[int]:
    if scope == "all": return None
    m = re.fullmatch(r"top(\d+)", scope)
    return int(m.group(1)) if m else (25 if scope == "top25" else None)

# -------------------------------
# load cached season assets (from build_dataset.py)
# -------------------------------
def load_cached(name: str):
    fp = DATA_DIR / name
    if not fp.exists():
        return None
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)

G_SEASON = load_cached("games.json") or []
L_SEASON = load_cached("lines.json") or []
R_SEASON = load_cached("rankings.json") or []
PPA      = load_cached("ppa_teams.json") or []

# -------------------------------
# small utils from season cache
# -------------------------------
def teams_in_week(games_week: List[dict]) -> List[str]:
    return sorted(set([g["homeTeam"] for g in games_week] + [g["awayTeam"] for g in games_week]))

def ap_topN_for_week(rankings: List[dict], week: int, N: int) -> List[str]:
    # find the ranking snapshot matching (or nearest before) this week
    snap = None
    for r in rankings:
        if r.get("week") == week:
            snap = r
    if not snap and rankings:
        snap = rankings[-1]  # fallback to last available snapshot

    out = []
    if not snap: return out
    for poll in snap.get("polls", []):
        nm = (poll.get("poll") or "").lower()
        if "ap" in nm or "associated press" in nm:
            for row in poll.get("ranks", []):
                try: rk = int(row.get("rank"))
                except: rk = None
                if rk and rk <= N:
                    team = row.get("school") or row.get("team")
                    if team: out.append(team)
    return out

def recency_pdpg(team: str, opp: str, week: int, n: int = 4) -> float:
    # compute point diff per game from cached season games < week
    rows = []
    for g in G_SEASON:
        if (g.get("week") or 999) >= week: 
            continue
        if g.get("homeTeam") == team:
            rows.append((g.get("homePoints"), g.get("awayPoints")))
        elif g.get("awayTeam") == team:
            rows.append((g.get("awayPoints"), g.get("homePoints")))
    rows = [(pf, pa) for (pf, pa) in rows if pf is not None and pa is not None]
    rows = rows[-n:]
    pdpg = sum((pf - pa) for pf, pa in rows) / len(rows) if rows else 0.0

    rows_o = []
    for g in G_SEASON:
        if (g.get("week") or 999) >= week: 
            continue
        if g.get("homeTeam") == opp:
            rows_o.append((g.get("homePoints"), g.get("awayPoints")))
        elif g.get("awayTeam") == opp:
            rows_o.append((g.get("awayPoints"), g.get("homePoints")))
    rows_o = [(pf, pa) for (pf, pa) in rows_o if pf is not None and pa is not None]
    rows_o = rows_o[-n:]
    opdpg = sum((pf - pa) for pf, pa in rows_o) / len(rows_o) if rows_o else 0.0

    return 0.5 * (pdpg - opdpg)

# -------------------------------
# component calculators (same as before)
# -------------------------------
def team_epp(off_ppa, opp_def_ppa): return clamp(BASE_EPP + (off_ppa - opp_def_ppa), 0.10, 0.80)

def pace_total(stats, home, away):
    def ppg(m):
        plays = m.get("plays"); g = m.get("games") or m.get("gp") or m.get("gms")
        try: return float(plays)/float(g) if plays and g else None
        except: return None
    p_h = ppg(stats[home]["off"]); p_a = ppg(stats[away]["off"])
    return (p_h or 65.0) + (p_a or 65.0)

def pace_scale(x, total_plays, baseline=130.0, elasticity=0.5):
    return x * (1.0 + elasticity * ((total_plays - baseline)/baseline))

def hidden_yards(spc_h, spc_a, pts_per_yd=0.055):
    net_h = spc_h.get("netpunting", 0) or (spc_h.get("puntyards",0)-spc_h.get("opponentpuntreturnyards",0))/max(1, spc_h.get("punts",1))
    net_a = spc_a.get("netpunting", 0) or (spc_a.get("puntyards",0)-spc_a.get("opponentpuntreturnyards",0))/max(1, spc_a.get("punts",1))
    ko_h  = (spc_h.get("kickreturnyards",0)-spc_h.get("opponentkickreturnyards",0))/max(1, spc_h.get("kickreturns",1))
    ko_a  = (spc_a.get("kickreturnyards",0)-spc_a.get("opponentkickreturnyards",0))/max(1, spc_a.get("kickreturns",1))
    return ((net_h - net_a) + 0.5*(ko_h - ko_a)) * pts_per_yd

def success_points(off_h, def_a, off_a, def_h, scale_per_5pct=1.5):
    def sr(m):
        for k,v in m.items():
            k=str(k).lower()
            if "success" in k and "%" in k:
                try: return float(v)/100.0
                except: return None
        return None
    sh_off, sa_def = sr(off_h), sr(def_a)
    sa_off, sh_def = sr(off_a), sr(def_h)
    pts=0.0
    if sh_off is not None and sa_def is not None: pts += ((sh_off - sa_def)/0.05)*scale_per_5pct
    if sa_off is not None and sh_def is not None: pts -= ((sa_off - sh_def)/0.05)*scale_per_5pct
    return pts

def explosiveness_points(off_map, def_map, h, a, scale_per_0p10=SCALE_PER_0P10):
    oh, da = off_map.get(h,0.0), def_map.get(a,0.0)
    oa, dh = off_map.get(a,0.0), def_map.get(h,0.0)
    return ((oh-da)-(oa-dh))/0.10*scale_per_0p10

def havoc_points(off_h, def_h, off_a, def_a, scale=3.0):
    def rate(d,o):
        tfl = (d.get("tacklesforloss",0) or d.get("tfl",0)) + (d.get("sacks",0) or 0)
        plays = d.get("plays",0) or 1
        sacks_allowed = o.get("sacksallowed",0) or 0
        return (tfl + sacks_allowed)/max(1,plays)
    try: return (rate(def_h, off_a) - rate(def_a, off_h))*scale
    except: return 0.0

# -------------------------------
# #2 Games for this week + scope filter (from cache, not API)
# -------------------------------
def games_for_week_from_cache(all_games: List[dict], year: int, week: int) -> List[dict]:
    return [g for g in all_games if g.get("year")==year and g.get("seasonType")=="regular" and g.get("week")==week]

def scope_filter(gdf: pd.DataFrame, scope: str) -> pd.DataFrame:
    N = parse_topN(scope)
    if N is None:
        return gdf
    ap = set(ap_topN_for_week(R_SEASON, WEEK, N))
    if not ap:
        return gdf
    return gdf[gdf["homeTeam"].isin(ap) | gdf["awayTeam"].isin(ap)]

# -------------------------------
# #3 Weekly run
# -------------------------------
def main() -> int:
    log(f"Inputs → YEAR:{YEAR} WEEK:{WEEK} SCOPE:{SCOPE} MODE:{MODE}")

    games = games_for_week_from_cache(G_SEASON, YEAR, WEEK)
    gdf = pd.DataFrame(games)
    if gdf.empty:
        log("No games for this (year, week) in season cache. Writing empty outputs.")
        return write_outputs([])

    gdf = scope_filter(gdf, SCOPE)
    if gdf.empty:
        log("Scope filter left zero games. Writing empty outputs.")
        return write_outputs([])

    teams = teams_in_week(gdf.to_dict("records"))

    # --- PPA maps (from cached PPA snapshot)
    off_map, def_map = {}, {}
    for row in PPA or []:
        t = row.get("team")
        off = ((row.get("offense") or {}).get("overall")
               or (row.get("offense") or {}).get("ppa") or 0.0) or 0.0
        deff = ((row.get("defense") or {}).get("overall")
               or (row.get("defense") or {}).get("ppa") or 0.0) or 0.0
        if t in teams:
            off_map[t] = float(off); def_map[t] = float(deff)

    # --- season stats (per-team, minimal live calls, cached by loader)
    def team_cat(team: str, cat: str) -> Dict[str, float]:
        rows = jget("/stats/season", {"year": YEAR, "team": team, "category": cat}, ttl_hours=24) or []
        m = {}
        for r in rows:
            n = (r.get("statName") or r.get("stat_name") or "").lower()
            v = r.get("statValue") or r.get("stat_value")
            try: v = float(v)
            except: pass
            m[n] = v
        return m

    stats = {t: {"off": team_cat(t,"offense"),
                 "def": team_cat(t,"defense"),
                 "spc": team_cat(t,"special")} for t in teams}

    # --- drives (optional)
    drives = {t: {"osfp":25.0,"dsfp":25.0} for t in teams}
    if MODE == "FULL":
        for t in teams:
            drv = jget("/drives", {"year": YEAR, "week": WEEK, "team": t}, ttl_hours=8)
            vals = [100 - d.get("start_yards_to_goal") for d in drv if d.get("start_yards_to_goal") is not None]
            drives[t]["osfp"] = (sum(vals)/len(vals)) if vals else 25.0
        drv_all = jget("/drives", {"year": YEAR, "week": WEEK}, ttl_hours=8)
        per_def = {t:[] for t in teams}
        for d in drv_all or []:
            tt = d.get("defense")
            if tt in per_def and d.get("start_yards_to_goal") is not None:
                per_def[tt].append(100 - d["start_yards_to_goal"])
        for t,vals in per_def.items():
            drives[t]["dsfp"] = (sum(vals)/len(vals)) if vals else 25.0

    # --- vegas lines (cached season lines -> compute averages for this week’s matchups)
    vegas_map = {}
    week_lines = [ln for ln in L_SEASON if ln.get("week")==WEEK]
    for ln in week_lines:
        h, a = ln.get("homeTeam"), ln.get("awayTeam")
        acc = []
        for b in (ln.get("lines") or []):
            sp = b.get("spread"); to = b.get("overUnder")
            if sp is not None or to is not None:
                acc.append((sp,to))
        if acc:
            s = [v for v,_ in acc if isinstance(v,(int,float))]
            t = [u for _,u in acc if isinstance(u,(int,float))]
            vegas_map[(h,a)] = {
                "vegas_spread": round(sum(s)/len(s),1) if s else None,
                "vegas_total":  round(sum(t)/len(t),1) if t else None
            }

    # --- compute rows
    out=[]
    for _, g in gdf.iterrows():
        h, a = g["homeTeam"], g["awayTeam"]
        hfa = NEUTRAL_HFA if bool(g.get("neutralSite")) else HFA_DEFAULT

        base = ((off_map.get(h,0.0) - def_map.get(a,0.0)) -
                (off_map.get(a,0.0) - def_map.get(h,0.0))) / 0.10 * SCALE_PER_0P10 + hfa

        plays = pace_total(stats, h, a)

        xpl_raw = pace_scale(explosiveness_points(off_map, def_map, h, a), plays)
        sr_raw  = pace_scale(success_points(stats[h]["off"], stats[a]["def"],
                                            stats[a]["off"], stats[h]["def"]), plays)
        hv_raw  = pace_scale(havoc_points(stats[h]["off"], stats[h]["def"],
                                          stats[a]["off"], stats[a]["def"]), plays)
        fp_raw  = 0.0 if MODE!="FULL" else (0.06 * (0.5*drives[h]["osfp"] + 0.5*drives[a]["dsfp"] - (0.5*drives[a]["osfp"] + 0.5*drives[h]["dsfp"])))
        hy_raw  = 0.0 if MODE!="FULL" else hidden_yards(stats[h]["spc"], stats[a]["spc"])
        rcy_raw = recency_pdpg(h, a, WEEK, n=4)

        def cap(name, val): return clamp(val*WEIGHTS[name], -COMP_LIMITS[name], COMP_LIMITS[name])

        fp  = cap("fp", fp_raw);   hy  = cap("hidden", hy_raw)
        xpl = cap("xpl", xpl_raw); sr  = cap("sr",   sr_raw)
        hv  = cap("havoc", hv_raw);rcy = cap("recency", rcy_raw)

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

        v = vegas_map.get((h,a), {})
        out.append({
            "home":h, "away":a, "favored":favored,
            "base_spread": round(base,1), "adj_spread": round(adj,1),
            "home_pts": home_pts, "away_pts": away_pts, "total_pts": int(total_pts),
            "plays_est": int(round(plays)),
            "fp": round(fp,2), "hidden": round(hy,2),
            "xpl": round(xpl,2), "sr": round(sr,2),
            "havoc": round(hv,2), "recency": round(rcy,2),
            "vegas_spread": v.get("vegas_spread"), "vegas_total": v.get("vegas_total")
        })

    out = sorted(out, key=lambda r: abs(r["adj_spread"]), reverse=True)
    return write_outputs(out)

def write_outputs(rows: List[Dict[str, Any]]) -> int:
    with (DOCS / "week_preds.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    pd.DataFrame(rows).to_csv("week_preds.csv", index=False)
    log(f"Wrote {len(rows)} rows → docs/week_preds.json & week_preds.csv")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())