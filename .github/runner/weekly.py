# runner/weekly.py
# CFB Model — weekly runner (clean, phone-friendly)
# All knobs live in "TUNABLES" below. No other edits needed.

import os, math, json, asyncio
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt

# ---------- TUNABLES (edit these only if you want to change behavior) ----------
TUNABLES = {
    # Spread construction
    "HFA_DEFAULT": 2.0,                 # home-field advantage (non-neutral)
    "SCALE_PER_0P10": 3.0,              # PPA gap → points (per 0.10)

    # Component caps (belt & suspenders)
    "COMP_LIMITS": {
        "fp": 6.0, "hidden": 4.0, "xpl": 10.0, "sr": 6.0, "havoc": 6.0, "recency": 6.0
    },
    "WEIGHTS": { "fp":1.0, "hidden":1.0, "xpl":1.0, "sr":1.0, "havoc":1.0, "recency":1.0 },

    # Pace / totals
    "BASE_EPP": 0.42,                   # ~55 pts over 130 plays
    "TOTAL_MIN": 30, "TOTAL_MAX": 95,   # keep totals sane
    "PACE_BASELINE": 130.0, "PACE_ELASTICITY": 0.5,

    # API hygiene
    "TIMEOUT": 40,                      # seconds
    "PARALLEL_LIMIT": 12,               # concurrency cap for calls
}

# ---------- Helpers ----------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def parse_top_n(scope: str) -> Optional[int]:
    scope = (scope or "top25").lower()
    if scope == "all": return None
    if scope.startswith("top"):
        try: return int(scope[3:])
        except: return 25
    return 25

def notice(msg: str) -> None:
    print(f"::notice::{msg}")

BASE = "https://api.collegefootballdata.com"
KEY  = os.getenv("CFBD_API_KEY")
HEAD = {"Authorization": f"Bearer {KEY}"} if KEY else {}

YEAR = int(os.getenv("YEAR", "2025"))
WEEK = int(os.getenv("WEEK", "6"))
SCOPE = os.getenv("SCOPE", "top25")
MODE  = (os.getenv("MODE", "FAST") or "FAST").upper()   # FAST or FULL
TOP_N = parse_top_n(SCOPE)

# ---------- HTTP client with retry ----------
@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(4))
async def jget(client: httpx.AsyncClient, path: str, params: Dict[str, Any] = None) -> Any:
    r = await client.get(path, headers=HEAD, params=params or {}, timeout=TUNABLES["TIMEOUT"])
    r.raise_for_status()
    return r.json()

# ---------- Component calculators ----------
def pace_total(stats: Dict[str, Dict[str, float]], h: str, a: str) -> float:
    def ppg(m):
        plays = m.get("plays"); g = m.get("games") or m.get("gp") or m.get("gms")
        try: return float(plays)/float(g) if plays and g else None
        except: return None
    p_h = ppg(stats[h]["off"]); p_a = ppg(stats[a]["off"])
    return (p_h or 65.0) + (p_a or 65.0)

def pace_scale(x: float, total_plays: float) -> float:
    return x * (1.0 + TUNABLES["PACE_ELASTICITY"] * ((total_plays - TUNABLES["PACE_BASELINE"]) / TUNABLES["PACE_BASELINE"]))

def fp_points(drives: Dict[str, Dict[str, float]], h: str, a: str, pts_per_yd: float = 0.06) -> float:
    exp_h = 0.5*drives[h]["osfp"] + 0.5*drives[a]["dsfp"]
    exp_a = 0.5*drives[a]["osfp"] + 0.5*drives[h]["dsfp"]
    return (exp_h - exp_a) * pts_per_yd

def hidden_yards(special: Dict[str, Dict[str, float]], h: str, a: str, pts_per_yd: float = 0.055) -> float:
    sh, sa = special[h], special[a]
    net_h = sh.get("netpunting",0) or (sh.get("puntyards",0)-sh.get("opponentpuntreturnyards",0))/max(1, sh.get("punts",1))
    net_a = sa.get("netpunting",0) or (sa.get("puntyards",0)-sa.get("opponentpuntreturnyards",0))/max(1, sa.get("punts",1))
    ko_h = (sh.get("kickreturnyards",0)-sh.get("opponentkickreturnyards",0))/max(1, sh.get("kickreturns",1))
    ko_a = (sa.get("kickreturnyards",0)-sa.get("opponentkickreturnyards",0))/max(1, sa.get("kickreturns",1))
    return ((net_h - net_a) + 0.5*(ko_h - ko_a)) * pts_per_yd

def success_points(off: Dict[str, float], deff: Dict[str, float]) -> Optional[float]:
    # returns success rate fraction if present (else None)
    def sr(m):
        for k, v in m.items():
            if "success" in k and "%" in k:
                try: return float(v)/100.0
                except: return None
        return None
    return sr(off), sr(deff)

def success_vs(success_h_off, success_a_def, success_a_off, success_h_def, scale_per_5pct=1.5) -> float:
    pts = 0.0
    if success_h_off is not None and success_a_def is not None:
        pts += ((success_h_off - success_a_def)/0.05) * scale_per_5pct
    if success_a_off is not None and success_h_def is not None:
        pts -= ((success_a_off - success_h_def)/0.05) * scale_per_5pct
    return pts

def explosiveness_points(off_map: Dict[str,float], def_map: Dict[str,float], h: str, a: str, scale_per_0p10=3.0) -> float:
    oh, da = off_map.get(h,0.0), def_map.get(a,0.0)
    oa, dh = off_map.get(a,0.0), def_map.get(h,0.0)
    return ((oh-da) - (oa-dh)) / 0.10 * scale_per_0p10

def havoc_points(off_stats: Dict[str, Dict[str,float]], def_stats: Dict[str, Dict[str,float]], h: str, a: str, scale=3.0) -> float:
    def rate(d, o):
        tfl = (d.get("tacklesforloss",0) or d.get("tfl",0)) + (d.get("sacks",0) or 0)
        plays = d.get("plays",0) or 1
        sacks_allowed = o.get("sacksallowed",0) or 0
        return (tfl + sacks_allowed) / max(1, plays)
    try:
        return (rate(def_stats[h], off_stats[a]) - rate(def_stats[a], off_stats[h])) * scale
    except:
        return 0.0

async def recency_points(client: httpx.AsyncClient, year: int, week: int, h: str, a: str, n=4, scale=0.5) -> float:
    async def team_pdpg(team):
        try:
            gt = await jget(client, "/games/teams", {"year": year, "team": team, "seasonType":"regular"})
            rows=[(g.get("pointsFor"), g.get("pointsAgainst")) for g in gt if g.get("week",99) < week]
            rows = rows[-n:]
            return (sum((pf-pa) for pf,pa in rows)/len(rows)) if rows else 0.0
        except:
            return 0.0
    h_pdpg, a_pdpg = await asyncio.gather(team_pdpg(h), team_pdpg(a))
    return (h_pdpg - a_pdpg) * scale

def team_epp(off_ppa: float, opp_def_ppa: float) -> float:
    epp = TUNABLES["BASE_EPP"] + (off_ppa - opp_def_ppa)
    return clamp(epp, 0.10, 0.80)

def predict_total_pts(plays: float, off_h: float, def_a: float, off_a: float, def_h: float, xpl_pts: float, sr_pts: float) -> int:
    total = team_epp(off_h, def_a)*(plays/2.0) + team_epp(off_a, def_h)*(plays/2.0)
    total += 0.5*(xpl_pts + sr_pts)
    return int(round(clamp(total, TUNABLES["TOTAL_MIN"], TUNABLES["TOTAL_MAX"])))

# ---------- Main async ----------
async def main():
    if not KEY:
        raise SystemExit("CFBD_API_KEY is missing (add it to repo Secrets).")

    limits = httpx.Limits(max_connections=TUNABLES["PARALLEL_LIMIT"], max_keepalive_connections= TUNABLES["PARALLEL_LIMIT"])
    async with httpx.AsyncClient(base_url=BASE, limits=limits) as client:
        notice(f"1/7: games…")
        games = await jget(client, "/games", {"year": YEAR, "week": WEEK, "seasonType":"regular"})
        gdf = pd.DataFrame(games)
        if gdf.empty:
            os.makedirs("docs", exist_ok=True)
            open("docs/week_preds.json","w").write("[]"); pd.DataFrame().to_csv("week_preds.csv", index=False)
            return

        # Top-N via AP poll
        if TOP_N is not None:
            notice("2/7: rankings…")
            ranks = await jget(client, "/rankings", {"year": YEAR, "week": WEEK})
            ap=set()
            for wk in ranks:
                for poll in wk.get("polls", []):
                    if (poll.get("poll") or "").startswith("AP"):
                        for r in poll.get("ranks", []):
                            try:
                                if r.get("school") and int(r.get("rank")) <= TOP_N:
                                    ap.add(r["school"])
                            except: pass
            gdf = gdf[gdf["homeTeam"].isin(ap) | gdf["awayTeam"].isin(ap)]
            if gdf.empty:
                os.makedirs("docs", exist_ok=True)
                open("docs/week_preds.json","w").write("[]"); pd.DataFrame().to_csv("week_preds.csv", index=False)
                return

        teams = sorted(set(gdf["homeTeam"]).union(gdf["awayTeam"]))

        notice("3/7: PPA…")
        ppa = await jget(client, "/ppa/teams", {"year": YEAR})
        off_map, def_map = {}, {}
        for row in ppa:
            t=row.get("team")
            off=((row.get("offense") or {}).get("overall") or (row.get("offense") or {}).get("ppa") or 0.0) or 0.0
            deff=((row.get("defense") or {}).get("overall") or (row.get("defense") or {}).get("ppa") or 0.0) or 0.0
            if t in teams: off_map[t]=float(off); def_map[t]=float(deff)

        notice("4/7: team stats (off/def/special)…")
        async def stats_one(team: str) -> Tuple[str, Dict[str, Dict[str, float]]]:
            async def one(cat):
                try:
                    rows = await jget(client, "/stats/season", {"year": YEAR, "team": team, "category": cat})
                except: rows=[]
                m={}
                for r in rows:
                    n=(r.get("statName") or r.get("stat_name") or "").lower()
                    v=r.get("statValue") or r.get("stat_value")
                    try: m[n]=float(v)
                    except: pass
                return cat, m
            cats = await asyncio.gather(one("offense"), one("defense"), one("special"))
            return team, {k:v for k,v in cats}
        stats_pairs = await asyncio.gather(*(stats_one(t) for t in teams))
        stats = {k:v for k,v in stats_pairs}

        # For convenience
        off_stats = {t:v["offense"] for t,v in stats.items()}
        def_stats = {t:v["defense"] for t,v in stats.items()}
        spc_stats = {t:v["special"] for t,v in stats.items()}

        # Field position via drives (optional)
        drives = {t:{"osfp":25.0,"dsfp":25.0} for t in teams}
        if MODE == "FULL":
            notice("5/7: drives (field position)…")
            async def osfp_one(team):
                try:
                    drv = await jget(client, "/drives", {"year": YEAR, "week": WEEK, "team": team})
                    vals=[100 - d.get("start_yards_to_goal") for d in drv if d.get("start_yards_to_goal") is not None]
                    drives[team]["osfp"] = sum(vals)/len(vals) if vals else 25.0
                except: drives[team]["osfp"] = 25.0
            await asyncio.gather(*(osfp_one(t) for t in teams))
            try:
                drv_all = await jget(client, "/drives", {"year": YEAR, "week": WEEK})
                per_def = {t:[] for t in teams}
                for d in drv_all:
                    t = d.get("defense")
                    if t in per_def and d.get("start_yards_to_goal") is not None:
                        per_def[t].append(100 - d["start_yards_to_goal"])
                for t,vals in per_def.items():
                    drives[t]["dsfp"] = (sum(vals)/len(vals)) if vals else 25.0
            except: pass

        # Vegas lines (average)
        notice("6/7: vegas…")
        lines = await jget(client, "/lines", {"year": YEAR, "week": WEEK, "seasonType":"regular"})
        v_map={}
        for ln in lines or []:
            home = ln.get("homeTeam"); away = ln.get("awayTeam")
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

        # Compute
        notice("7/7: compute spreads & totals…")
        out=[]
        for _, g in gdf.iterrows():
            h, a = g["homeTeam"], g["awayTeam"]

            hfa = TUNABLES["HFA_DEFAULT"] if not bool(g.get("neutralSite")) else 0.5
            base = ((off_map.get(h,0.0) - def_map.get(a,0.0)) -
                    (off_map.get(a,0.0) - def_map.get(h,0.0))) / 0.10 * TUNABLES["SCALE_PER_0P10"] + hfa

            plays = pace_total(stats, h, a)

            xpl_raw = pace_scale(explosiveness_points(off_map, def_map, h, a), plays)
            s_h_off, s_a_def = success_points(stats[h]["offense"], stats[a]["defense"])
            s_a_off, s_h_def = success_points(stats[a]["offense"], stats[h]["defense"])
            sr_raw  = pace_scale(success_vs(s_h_off, s_a_def, s_a_off, s_h_def), plays)
            hv_raw  = pace_scale(havoc_points(off_stats, def_stats, h, a), plays)
            fp_raw  = fp_points(drives, h, a) if MODE=="FULL" else 0.0
            hy_raw  = hidden_yards(spc_stats, h, a) if MODE=="FULL" else 0.0
            rcy_raw = await recency_points(client, YEAR, WEEK, h, a)

            # Cap components
            limits=TUNABLES["COMP_LIMITS"]; w=TUNABLES["WEIGHTS"]
            def cap(name,val): return clamp(val*w[name], -limits[name], limits[name])
            fp=cap("fp",fp_raw); hy=cap("hidden",hy_raw); xpl=cap("xpl",xpl_raw)
            sr=cap("sr",sr_raw); hv=cap("havoc",hv_raw); rcy=cap("recency",rcy_raw)

            adj = clamp(base + fp + hy + xpl + sr + hv + rcy, -40.0, 40.0)

            total_pts = predict_total_pts(plays, off_map.get(h,0.0), def_map.get(a,0.0),
                                          off_map.get(a,0.0), def_map.get(h,0.0),
                                          xpl, sr)

            home_pts = int(round((total_pts + adj)/2)); away_pts = int(round(total_pts - home_pts))
            home_pts = max(0, home_pts); away_pts = max(0, away_pts)
            fav = h if adj >= 0 else a

            v_spread = vegas.get((h,a),{}).get("vegas_spread")
            v_total  = vegas.get((h,a),{}).get("vegas_total")

            out.append({
                "home":h,"away":a,"favored":fav,
                "base_spread": round(base,1),
                "adj_spread":  round(adj,1),
                "home_pts": home_pts, "away_pts": away_pts,
                "total_pts": int(total_pts),
                "vegas_spread": v_spread, "vegas_total": v_total,
                "plays_est": int(round(plays)),
                "fp": round(fp,2), "hidden": round(hy,2),
                "xpl": round(xpl,2), "sr": round(sr,2),
                "havoc": round(hv,2), "recency": round(rcy,2),
                "mode": MODE, "scope": SCOPE
            })

        out = sorted(out, key=lambda r: abs(r["adj_spread"]), reverse=True)

        # Write outputs
        pd.DataFrame(out).to_csv("week_preds.csv", index=False)
        os.makedirs("docs", exist_ok=True)
        with open("docs/week_preds.json","w",encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())