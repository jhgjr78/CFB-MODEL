#!/usr/bin/env python3
"""
Weekly predictions (cache-first, API-fallback & low-quota).
Reads the week cache written by mini_loader.py. If missing,
falls back to API with polite backoff and writes week cache.

Outputs:
  - docs/week_preds.json    (for your website)
  - week_preds.csv          (artifact)
Also drops light debug files in docs/ if something is empty.

Env:
  CFBD_API_KEY, YEAR, WEEK, SCOPE ('all'|'top25'|'top10'), MODE ('FULL'|'FAST')
"""

import os, re, json, time, math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import requests
import pandas as pd

# ---------- Tunables ----------
SCALE_PER_0P10 = 3.0
HFA_DEFAULT    = 2.0
NEUTRAL_HFA    = 0.5
COMP_LIMITS = {"fp":6.0, "hidden":4.0, "xpl":10.0, "sr":6.0, "havoc":6.0, "recency":6.0}
WEIGHTS = {k:1.0 for k in COMP_LIMITS}
BASE_EPP = 0.42
TOTAL_FLOOR, TOTAL_CEIL = 30, 95
SPREAD_FLOOR, SPREAD_CEIL = -40.0, 40.0

# ---------- Env & HTTP ----------
BASE = "https://api.collegefootballdata.com"
TIMEOUT = 35
MAX_RETRIES = 6

YEAR  = int(os.getenv("YEAR","2025"))
WEEK  = int(os.getenv("WEEK","6"))
SCOPE = (os.getenv("SCOPE","top25") or "top25").lower()
MODE  = (os.getenv("MODE","FULL") or "FULL").upper()
API   = os.getenv("CFBD_API_KEY","")
HEAD  = {"Authorization": f"Bearer {API}"} if API else {}

def log(msg: str) -> None:
    print(f"##[notice]{msg}")

def jget(path: str, params: Dict[str,Any]=None) -> Any:
    """API GET with backoff; use sparingly (cache-first)."""
    url = f"{BASE}{path}"; params = params or {}
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

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def parse_topN(scope: str) -> Optional[int]:
    if scope == "all": return None
    m = re.match(r"top(\d+)$", scope)
    if m: return int(m.group(1))
    if scope == "top25": return 25
    return None

# ---------- Paths ----------
week_dir   = Path(f"data/weeks/{YEAR}")
week_games_fp    = week_dir / f"week_{WEEK}.games.json"
week_rankings_fp = week_dir / f"week_{WEEK}.rankings.json"
season_dir = Path(f"data/season_{YEAR}")
ppa_fp     = season_dir / "ppa_teams.json"

# ---------- Light file I/O ----------
def read_json(p: Path, default):
    try:
        if p.exists() and p.stat().st_size > 1:
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

def write_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

# ---------- Components ----------
def team_epp(off_ppa: float, opp_def_ppa: float) -> float:
    return clamp(BASE_EPP + (off_ppa - opp_def_ppa), 0.10, 0.80)

def pace_total(stats, home, away):
    def ppg(m):
        plays = m.get("plays"); g = m.get("games") or m.get("gp") or m.get("gms")
        try: return float(plays)/float(g) if plays and g else None
        except: return None
    p_h = ppg(stats[home]["off"]); p_a = ppg(stats[away]["off"])
    return (p_h or 65.0) + (p_a or 65.0)

def pace_scale(x, total_plays, baseline=130.0, elasticity=0.5):
    return x*(1.0 + elasticity*((total_plays-baseline)/baseline))

def fp_points(dr, h, a, pts_per_yd=0.06):
    exp_h = 0.5*dr[h]["osfp"] + 0.5*dr[a]["dsfp"]
    exp_a = 0.5*dr[a]["osfp"] + 0.5*dr[h]["dsfp"]
    return (exp_h - exp_a)*pts_per_yd

def hidden_yards(spc_h, spc_a, pts_per_yd=0.055):
    net_h = spc_h.get("netpunting",0) or (spc_h.get("puntyards",0)-spc_h.get("opponentpuntreturnyards",0))/max(1, spc_h.get("punts",1))
    net_a = spc_a.get("netpunting",0) or (spc_a.get("puntyards",0)-spc_a.get("opponentpuntreturnyards",0))/max(1, spc_a.get("punts",1))
    ko_h  = (spc_h.get("kickreturnyards",0)-spc_h.get("opponentkickreturnyards",0))/max(1, spc_h.get("kickreturns",1))
    ko_a  = (spc_a.get("kickreturnyards",0)-spc_a.get("opponentkickreturnyards",0))/max(1, spc_a.get("kickreturns",1))
    return ((net_h-net_a)+0.5*(ko_h-ko_a))*pts_per_yd

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
    if sh_off is not None and sa_def is not None: pts += ((sh_off-sa_def)/0.05)*scale_per_5pct
    if sa_off is not None and sh_def is not None: pts -= ((sa_off-sh_def)/0.05)*scale_per_5pct
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
    try: return (rate(def_h,off_a) - rate(def_a,off_h))*scale
    except: return 0.0

# recency from cached past weeks (no API calls)
def recency_from_cache(all_weeks_games: Dict[int, List[Dict[str,Any]]], week: int, h: str, a: str, n=4, scale=0.5):
    def last_pdpg(team):
        vals=[]
        for w in range(max(1,week-12), week):  # lookback window
            for g in all_weeks_games.get(w, []):
                if g.get("homeTeam")==team or g.get("awayTeam")==team:
                    pf = g.get("homePoints") if g.get("homeTeam")==team else g.get("awayPoints")
                    pa = g.get("awayPoints") if g.get("homeTeam")==team else g.get("homePoints")
                    if pf is not None and pa is not None:
                        vals.append(pf-pa)
        vals = vals[-n:]
        return sum(vals)/len(vals) if vals else 0.0
    return (last_pdpg(h) - last_pdpg(a))*scale

# ---------- Stats helpers with small per-team cache ----------
def load_team_stats(year: int, team: str, category: str) -> Dict[str,Any]:
    p = season_dir / "stats" / f"{team}_{category}.json"
    obj = read_json(p, None)
    if obj is not None: return obj
    try:
        rows = jget("/stats/season", {"year": year, "team": team, "category": category}) or []
        m={}
        for r in rows:
            name = (r.get("statName") or r.get("stat_name") or "").lower()
            val  = r.get("statValue") or r.get("stat_value")
            try: val = float(val)
            except: pass
            m[name]=val
        write_json(p, m)
        return m
    except Exception:
        return {}

# ---------- Main ----------
def main() -> int:
    log(f"Inputs → YEAR:{YEAR} WEEK:{WEEK} SCOPE:{SCOPE} MODE:{MODE}")

    # 1) Load caches or live
    games = read_json(week_games_fp, [])
    if not games:
        log("Cache miss: games. Pulling live once.")
        try:
            games = jget("/games", {"year": YEAR, "week": WEEK, "seasonType":"regular"}) or []
            write_json(week_games_fp, games)
        except Exception as e:
            log(f"ERROR pulling games: {e}")
            games = []

    if not games:
        # write empty outputs and bail early
        Path("docs").mkdir(parents=True, exist_ok=True)
        write_json(Path("docs/_games_dbg.json"), games)
        write_json(Path("docs/week_preds.json"), [])
        pd.DataFrame().to_csv("week_preds.csv", index=False)
        log("No games found. Wrote empty outputs.")
        return 0

    ranks = read_json(week_rankings_fp, [])
    if not ranks:
        try:
            ranks = jget("/rankings", {"year": YEAR, "week": WEEK}) or []
            write_json(week_rankings_fp, ranks)
        except Exception:
            ranks = []

    # 2) Scope filter (topN via cached rankings)
    topN = parse_topN(SCOPE)
    df = pd.DataFrame(games)
    if topN is not None:
        def _poll_is_ap(name: str) -> bool:
            n=(name or "").lower()
            return ("ap" in n) or ("associated press" in n) or ("ap top" in n)
        ap=set()
        latest = ranks[-1] if ranks else {}
        for poll in latest.get("polls", []):
            if _poll_is_ap(poll.get("poll","")):
                for r in poll.get("ranks", []):
                    try:
                        rk=int(r.get("rank"))
                    except:
                        rk=None
                    t=r.get("school") or r.get("team")
                    if t and rk and rk<=topN: ap.add(t)
        if ap:
            df = df[df["homeTeam"].isin(ap) | df["awayTeam"].isin(ap)]
    if df.empty:
        Path("docs").mkdir(parents=True, exist_ok=True)
        write_json(Path("docs/week_preds.json"), [])
        pd.DataFrame().to_csv("week_preds.csv", index=False)
        log("Scope resulted in 0 games. Wrote empty outputs.")
        return 0

    teams = sorted(set(df["homeTeam"]).union(df["awayTeam"]))

    # 3) PPA (cache season file or pull once)
    ppa = read_json(ppa_fp, [])
    if not ppa:
        try:
            log("Cache miss: PPA. Pulling live once.")
            ppa = jget("/ppa/teams", {"year": YEAR}) or []
            write_json(ppa_fp, ppa)
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

    # 4) Season stats (cache per team/category)
    stats = {t: {
        "off": load_team_stats(YEAR, t, "offense"),
        "def": load_team_stats(YEAR, t, "defense"),
        "spc": load_team_stats(YEAR, t, "special"),
    } for t in teams}

    # 5) Drives (field position) – FAST mode skips
    drives = {t: {"osfp":25.0, "dsfp":25.0} for t in teams}
    if MODE == "FULL":
        # Offense-side per team
        for t in teams:
            try:
                drv = jget("/drives", {"year": YEAR, "week": WEEK, "team": t}) or []
                vals=[100 - d.get("start_yards_to_goal") for d in drv if d.get("start_yards_to_goal") is not None]
                drives[t]["osfp"] = (sum(vals)/len(vals)) if vals else 25.0
            except Exception:
                pass
        # Defense aggregation
        try:
            all_drv = jget("/drives", {"year": YEAR, "week": WEEK}) or []
            per_def = {t:[] for t in teams}
            for d in all_drv:
                tt = d.get("defense")
                if tt in per_def and d.get("start_yards_to_goal") is not None:
                    per_def[tt].append(100 - d["start_yards_to_goal"])
            for t,vals in per_def.items():
                drives[t]["dsfp"] = (sum(vals)/len(vals)) if vals else 25.0
        except Exception:
            pass

    # 6) Vegas lines (avg)
    vegas_raw=[]
    try:
        vegas_raw = jget("/lines", {"year": YEAR, "week": WEEK, "seasonType":"regular"}) or []
    except Exception:
        vegas_raw=[]
    vmap={}
    for ln in vegas_raw:
        h,a = ln.get("homeTeam"), ln.get("awayTeam")
        vals=ln.get("lines") or []
        ss=[x.get("spread") for x in vals if isinstance(x.get("spread"), (int,float))]
        tt=[x.get("overUnder") for x in vals if isinstance(x.get("overUnder"), (int,float))]
        if h and a and (ss or tt):
            vmap[(h,a)]={"vegas_spread": round(sum(ss)/len(ss),1) if ss else None,
                         "vegas_total":  round(sum(tt)/len(tt),1) if tt else None}

    # 7) Recency from cached prior weeks
    all_weeks_games: Dict[int, List[Dict[str,Any]]] = {}
    for w in range(1, WEEK):
        fp = Path(f"data/weeks/{YEAR}/week_{w}.games.json")
        all_weeks_games[w] = read_json(fp, [])

    # 8) Compute
    out=[]
    for _, g in df.iterrows():
        h, a = g["homeTeam"], g["awayTeam"]
        hfa = NEUTRAL_HFA if bool(g.get("neutralSite")) else HFA_DEFAULT

        base = ((off_map.get(h,0.0) - def_map.get(a,0.0))
               -(off_map.get(a,0.0) - def_map.get(h,0.0))) / 0.10 * SCALE_PER_0P10 + hfa

        plays = pace_total(stats, h, a)

        xpl_raw = pace_scale(explosiveness_points(off_map, def_map, h, a), plays)
        sr_raw  = pace_scale(success_points(stats[h]["off"], stats[a]["def"], stats[a]["off"], stats[h]["def"]), plays)
        hv_raw  = pace_scale(havoc_points(stats[h]["off"], stats[h]["def"], stats[a]["off"], stats[a]["def"]), plays)
        fp_raw  = fp_points(drives, h, a) if MODE=="FULL" else 0.0
        hy_raw  = hidden_yards(stats[h]["spc"], stats[a]["spc"]) if MODE=="FULL" else 0.0
        rcy_raw = recency_from_cache(all_weeks_games, WEEK, h, a)

        def cap(name,val): return clamp(val*WEIGHTS[name], -COMP_LIMITS[name], COMP_LIMITS[name])
        fp = cap("fp", fp_raw); hy=cap("hidden",hy_raw); xpl=cap("xpl",xpl_raw)
        sr = cap("sr",sr_raw);  hv=cap("havoc",hv_raw);  rcy=cap("recency",rcy_raw)

        adj = clamp(base + fp + hy + xpl + sr + hv + rcy, SPREAD_FLOOR, SPREAD_CEIL)

        epp_h = team_epp(off_map.get(h,0.0), def_map.get(a,0.0))
        epp_a = team_epp(off_map.get(a,0.0), def_map.get(h,0.0))
        total_pts = epp_h*(plays/2.0) + epp_a*(plays/2.0)
        total_pts += 0.5*(xpl + sr)
        total_pts = int(round(clamp(total_pts, TOTAL_FLOOR, TOTAL_CEIL)))

        home_pts = int(round((total_pts + adj)/2))
        away_pts = max(0, total_pts - home_pts)
        home_pts = max(0, home_pts)
        favored = h if adj >= 0 else a

        v = vmap.get((h,a), {})
        out.append({
            "home": h, "away": a, "favored": favored,
            "base_spread": round(base,1),
            "adj_spread": round(adj,1),
            "home_pts": home_pts, "away_pts": away_pts,
            "total_pts": int(total_pts),
            "plays_est": int(round(plays)),
            "fp": round(fp,2), "hidden": round(hy,2),
            "xpl": round(xpl,2), "sr": round(sr,2),
            "havoc": round(hv,2), "recency": round(rcy,2),
            "vegas_spread": v.get("vegas_spread"),
            "vegas_total": v.get("vegas_total")
        })

    out = sorted(out, key=lambda r: abs(r["adj_spread"]), reverse=True)

    # write outputs
    Path("docs").mkdir(parents=True, exist_ok=True)
    write_json(Path("docs/week_preds.json"), out)
    pd.DataFrame(out).to_csv("week_preds.csv", index=False)

    # tiny debug drops
    if not out:
        write_json(Path("docs/_games_dbg.json"), games)
        write_json(Path("docs/_rankings_dbg.json"), ranks)

    log(f"Wrote {len(out)} rows → docs/week_preds.json & week_preds.csv")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())