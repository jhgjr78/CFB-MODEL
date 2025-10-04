#!/usr/bin/env python3
"""
Weekly predictions builder.
- Reads cached data from data/weeks/<year>/week_<n>.games.json and rankings.json
- Falls back to API only if cache files are missing.
- Never writes empty outputs unless there are truly no games.
Outputs:
  - docs/week_preds.json
  - week_preds.csv
"""

import os, re, json, math, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import requests

# ---------- Tunables ----------
SCALE_PER_0P10 = 3.0
HFA_DEFAULT    = 2.0
NEUTRAL_HFA    = 0.5

COMP_LIMITS = {"fp":6.0,"hidden":4.0,"xpl":10.0,"sr":6.0,"havoc":6.0,"recency":6.0}
WEIGHTS     = {k:1.0 for k in COMP_LIMITS}
BASE_EPP = 0.42
TOTAL_FLOOR, TOTAL_CEIL = 30, 95
SPREAD_FLOOR, SPREAD_CEIL = -40.0, 40.0

BASE = "https://api.collegefootballdata.com"
API  = os.getenv("CFBD_API_KEY","")
HEAD = {"Authorization": f"Bearer {API}"} if API else {}
TIMEOUT=40
MAX_RETRIES=5

YEAR  = int(os.getenv("YEAR","2025"))
WEEK  = int(os.getenv("WEEK","6"))
SCOPE = (os.getenv("SCOPE","all") or "all").lower()
MODE  = (os.getenv("MODE","FULL") or "FULL").upper()

def log(m): print(f"::notice::{m}")

def clamp(x, lo, hi): return max(lo, min(hi, x))

def jget(path, params):
    url=f"{BASE}{path}"; back=1.5
    for i in range(1, MAX_RETRIES+1):
        r=requests.get(url, headers=HEAD, params=params, timeout=TIMEOUT)
        if r.status_code==429 and i<MAX_RETRIES:
            ra = r.headers.get("Retry-After")
            delay = float(ra) if ra else back*i
            log(f"[429] {path} – sleeping {delay:.1f}s then retry…")
            time.sleep(delay); continue
        try:
            r.raise_for_status(); return r.json()
        except requests.HTTPError:
            if 500<=r.status_code<600 and i<MAX_RETRIES:
                delay=back*i; log(f"[{r.status_code}] {path} – sleeping {delay:.1f}s then retry…")
                time.sleep(delay); continue
            raise

def parse_topN(scope:str)->Optional[int]:
    if scope=="all": return None
    m=re.match(r"top(\d+)$", scope); 
    if m: return int(m.group(1))
    if scope=="top25": return 25
    return None

# ---------- Components ----------
def team_epp(off_ppa, opp_def_ppa): return clamp(BASE_EPP+(off_ppa-opp_def_ppa), 0.10, 0.80)

def pace_total(stats, h,a):
    def pg(m):
        plays=m.get("plays"); g=m.get("games") or m.get("gp") or m.get("gms")
        try: return float(plays)/float(g) if plays and g else None
        except: return None
    return (pg(stats[h]["off"]) or 65.0)+(pg(stats[a]["off"]) or 65.0)

def pace_scale(x, total_plays, baseline=130.0, elasticity=0.5):
    return x*(1.0+elasticity*((total_plays-baseline)/baseline))

def fp_points(dr, h,a, pts_per_yd=0.06):
    exp_h=0.5*dr[h]["osfp"]+0.5*dr[a]["dsfp"]
    exp_a=0.5*dr[a]["osfp"]+0.5*dr[h]["dsfp"]
    return (exp_h-exp_a)*pts_per_yd

def hidden_yards(spc_h, spc_a, pts_per_yd=0.055):
    net_h = spc_h.get("netpunting",0) or (spc_h.get("puntyards",0)-spc_h.get("opponentpuntreturnyards",0))/max(1, spc_h.get("punts",1))
    net_a = spc_a.get("netpunting",0) or (spc_a.get("puntyards",0)-spc_a.get("opponentpuntreturnyards",0))/max(1, spc_a.get("punts",1))
    ko_h=(spc_h.get("kickreturnyards",0)-spc_h.get("opponentkickreturnyards",0))/max(1, spc_h.get("kickreturns",1))
    ko_a=(spc_a.get("kickreturnyards",0)-spc_a.get("opponentkickreturnyards",0))/max(1, spc_a.get("kickreturns",1))
    return ((net_h-net_a)+0.5*(ko_h-ko_a))*pts_per_yd

def success_points(off_h, def_a, off_a, def_h, scale_per_5pct=1.5):
    def sr(m):
        for k,v in m.items():
            if "success" in str(k).lower() and "%" in str(k):
                try: return float(v)/100.0
                except: return None
        return None
    sh_off, sa_def = sr(off_h), sr(def_a)
    sa_off, sh_def = sr(off_a), sr(def_h)
    pts=0.0
    if sh_off is not None and sa_def is not None: pts+=((sh_off-sa_def)/0.05)*scale_per_5pct
    if sa_off is not None and sh_def is not None: pts-=((sa_off-sh_def)/0.05)*scale_per_5pct
    return pts

def explosiveness_points(off_map, def_map, h,a, scale_per_0p10=SCALE_PER_0P10):
    oh,da=off_map.get(h,0.0), def_map.get(a,0.0)
    oa,dh=off_map.get(a,0.0), def_map.get(h,0.0)
    return ((oh-da)-(oa-dh))/0.10*scale_per_0p10

def havoc_points(off_h, def_h, off_a, def_a, scale=3.0):
    def rate(d,o):
        tfl=(d.get("tacklesforloss",0) or d.get("tfl",0))+(d.get("sacks",0) or 0)
        plays=d.get("plays",0) or 1
        sacks_allowed=o.get("sacksallowed",0) or 0
        return (tfl+sacks_allowed)/max(1,plays)
    try: return (rate(def_h,off_a)-rate(def_a,off_h))*scale
    except: return 0.0

# ---------- Main ----------
def write_outputs(rows: List[Dict[str,Any]])->int:
    Path("docs").mkdir(parents=True, exist_ok=True)
    with open("docs/week_preds.json","w",encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    pd.DataFrame(rows).to_csv("week_preds.csv", index=False)
    log(f"Wrote {len(rows)} rows → docs/week_preds.json and week_preds.csv")
    return 0

def _load_cache(path: Path):
    if path.exists():
        try: return json.loads(path.read_text())
        except: return []
    return None  # “missing” vs “empty list”

def main()->int:
    log(f"Inputs → YEAR:{YEAR} WEEK:{WEEK} SCOPE:{SCOPE} MODE:{MODE}")

    # 1) Games (prefer cache)
    p_games    = Path(f"data/weeks/{YEAR}/week_{WEEK}.games.json")
    p_rankings = Path(f"data/weeks/{YEAR}/week_{WEEK}.rankings.json")
    p_ppa      = Path(f"data/season_{YEAR}/ppa_teams.json")

    games = _load_cache(p_games)
    if games is None:  # not cached – one fallback hit
        log(f"#1 Fetching games YEAR={YEAR} WEEK={WEEK}")
        try:
            games = jget("/games", {"year": YEAR, "week": WEEK, "seasonType":"regular"}) or []
        except Exception:
            games = []
    gdf = pd.DataFrame(games)
    if gdf.empty:
        log("#1 No games returned. Writing empty outputs.")
        return write_outputs([])

    # 2) Rankings filter (TopN via AP) — robust to empty snapshots
    def _poll_is_ap(name:str)->bool:
        if not name: return False
        n=name.lower(); return "ap" in n or "associated press" in n or "ap top" in n

    topN = parse_topN(SCOPE)
    if topN is not None:
        ranks_json = _load_cache(p_rankings)
        if ranks_json is None:
            try:
                ranks_json = jget("/rankings", {"year": YEAR, "week": WEEK}) or []
            except Exception:
                ranks_json = []
        ap=set()
        if ranks_json:
            latest = ranks_json[-1]
            for poll in latest.get("polls", []):
                if _poll_is_ap(poll.get("poll","")):
                    for r in poll.get("ranks", []):
                        team = r.get("school") or r.get("team")
                        try: rk = int(r.get("rank"))
                        except: rk = None
                        if team and rk and rk <= topN:
                            ap.add(team)
        if ap:
            gdf = gdf[gdf["homeTeam"].isin(ap) | gdf["awayTeam"].isin(ap)]
        if gdf.empty:
            log("Scope filter left zero games. Writing empty outputs.")
            return write_outputs([])

    teams = sorted(set(gdf["homeTeam"]).union(gdf["awayTeam"]))

    # 3) PPA maps (prefer cache)
    try:
        ppa = _load_cache(p_ppa)
        if ppa is None: ppa = jget("/ppa/teams", {"year": YEAR}) or []
    except Exception:
        ppa = []
    off_map, def_map = {}, {}
    for row in ppa:
        t=row.get("team")
        off=((row.get("offense") or {}).get("overall")
             or (row.get("offense") or {}).get("ppa") or 0.0) or 0.0
        deff=((row.get("defense") or {}).get("overall")
              or (row.get("defense") or {}).get("ppa") or 0.0) or 0.0
        if t in teams:
            off_map[t]=float(off); def_map[t]=float(deff)

    # 4) Team season stats (live hits; cheap per team)
    def team_cat(team: str, cat: str)->Dict[str,float]:
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
        except: return {}
    stats = {t: {"off": team_cat(t,"offense"), "def": team_cat(t,"defense"), "spc": team_cat(t,"special")}
             for t in teams}

    # 5) Drives (FULL only)
    drives = {t: {"osfp":25.0,"dsfp":25.0} for t in teams}
    if MODE == "FULL":
        for t in teams:
            try:
                drv = jget("/drives", {"year": YEAR, "week": WEEK, "team": t})
                vals=[100-d.get("start_yards_to_goal") for d in drv if d.get("start_yards_to_goal") is not None]
                drives[t]["osfp"] = sum(vals)/len(vals) if vals else 25.0
            except: pass
        try:
            drv_all = jget("/drives", {"year": YEAR, "week": WEEK})
            per_def = {t:[] for t in teams}
            for d in drv_all or []:
                tt=d.get("defense")
                if tt in per_def and d.get("start_yards_to_goal") is not None:
                    per_def[tt].append(100-d["start_yards_to_goal"])
            for t,vals in per_def.items():
                drives[t]["dsfp"] = (sum(vals)/len(vals)) if vals else 25.0
        except: pass

    # 6) Vegas (optional)
    vegas={}
    try:
        lines = jget("/lines", {"year": YEAR, "week": WEEK, "seasonType":"regular"})
        vmap={}
        for ln in lines or []:
            h,a=ln.get("homeTeam"), ln.get("awayTeam")
            for b in (ln.get("lines") or []):
                sp,tot=b.get("spread"), b.get("overUnder")
                if h and a and (sp is not None or tot is not None):
                    vmap.setdefault((h,a),[]).append((sp,tot))
        for k,vals in vmap.items():
            s=[v for v,_ in vals if isinstance(v,(int,float))]
            t=[u for _,u in vals if isinstance(u,(int,float))]
            vegas[k]={"vegas_spread": round(sum(s)/len(s),1) if s else None,
                      "vegas_total":  round(sum(t)/len(t),1) if t else None}
    except: pass

    # 7) Compute rows
    out=[]
    for _, g in gdf.iterrows():
        h,a=g["homeTeam"], g["awayTeam"]
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

        # simple recency via games/teams (cheap enough, but skip if it 429s)
        try:
            gh = jget("/games/teams", {"year": YEAR, "team": h, "seasonType":"regular"}) or []
            ga = jget("/games/teams", {"year": YEAR, "team": a, "seasonType":"regular"}) or []
            def last_pdpg(rows, n=4):
                vals=[(r.get("pointsFor"), r.get("pointsAgainst"))
                      for r in rows if (r.get("week") or 99) < WEEK][-n:]
                return sum((pf-pa) for pf,pa in vals)/len(vals) if vals else 0.0
            rcy_raw = (last_pdpg(gh)-last_pdpg(ga))*0.5
        except:
            rcy_raw = 0.0

        def cap(name,val): return clamp(val*WEIGHTS[name], -COMP_LIMITS[name], COMP_LIMITS[name])

        fp=cap("fp",fp_raw); hy=cap("hidden",hy_raw); xpl=cap("xpl",xpl_raw)
        sr=cap("sr",sr_raw); hv=cap("havoc",hv_raw); rcy=cap("recency",rcy_raw)

        adj = clamp(base + fp + hy + xpl + sr + hv + rcy, SPREAD_FLOOR, SPREAD_CEIL)

        epp_h = team_epp(off_map.get(h,0.0), def_map.get(a,0.0))
        epp_a = team_epp(off_map.get(a,0.0), def_map.get(h,0.0))
        total_pts = int(round(clamp(epp_h*(plays/2.0)+epp_a*(plays/2.0)+0.5*(xpl+sr), TOTAL_FLOOR, TOTAL_CEIL)))

        home_pts = max(0, int(round((total_pts+adj)/2)))
        away_pts = max(0, int(round(total_pts-home_pts)))
        favored  = h if adj>=0 else a

        v = vegas.get((h,a),{})
        out.append({
            "home":h,"away":a,"favored":favored,
            "base_spread":round(base,1),"adj_spread":round(adj,1),
            "home_pts":home_pts,"away_pts":away_pts,"total_pts":int(total_pts),
            "plays_est":int(round(plays)),
            "fp":round(fp,2),"hidden":round(hy,2),"xpl":round(xpl,2),"sr":round(sr,2),
            "havoc":round(hv,2),"recency":round(rcy,2),
            "vegas_spread":v.get("vegas_spread"),"vegas_total":v.get("vegas_total")
        })

    out = sorted(out, key=lambda r: abs(r["adj_spread"]), reverse=True)
    return write_outputs(out)

if __name__ == "__main__":
    raise SystemExit(main())