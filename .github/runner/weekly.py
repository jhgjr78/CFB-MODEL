#!/usr/bin/env python3
"""
Weekly projections — reads local Parquet dataset built by build_dataset.py.
NO live API calls here.

Env:
  YEAR=2025
  WEEK=6
  SCOPE=all|top25|top10   (we keep 'all' until we add local poll)
  MODE=FULL|FAST          (FULL uses drives_week<W>.parquet if present)

Writes:
  docs/week_preds.json
  week_preds.csv
"""
import os, json, re
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

# ---------- load weights (no code edits) ----------
def load_weights():
    wpath = Path("weights.json")
    if wpath.exists():
        return json.loads(wpath.read_text())
    return {}

W = load_weights()
SCALE_PER_0P10 = W.get("SCALE_PER_0P10", 3.0)
HFA_DEFAULT    = W.get("HFA_DEFAULT", 2.0)
NEUTRAL_HFA    = W.get("NEUTRAL_HFA", 0.5)
COMP_LIMITS    = W.get("COMP_LIMITS", {"fp":6.0,"hidden":4.0,"xpl":10.0,"sr":6.0,"havoc":6.0,"recency":6.0})
WEIGHTS        = W.get("WEIGHTS",     {"fp":1.0,"hidden":1.0,"xpl":1.0,"sr":1.0,"havoc":1.0,"recency":1.0})
BASE_EPP       = W.get("BASE_EPP", 0.42)
TOTAL_MIN      = W.get("TOTAL_MIN", 30)
TOTAL_MAX      = W.get("TOTAL_MAX", 95)
SPREAD_MIN     = W.get("SPREAD_MIN", -40.0)
SPREAD_MAX     = W.get("SPREAD_MAX", 40.0)

# ---------- inputs ----------
YEAR  = int(os.getenv("YEAR","2025"))
WEEK  = int(os.getenv("WEEK","6"))
SCOPE = (os.getenv("SCOPE","all") or "all").lower()
MODE  = (os.getenv("MODE","FULL") or "FULL").upper()

def log(m): print(f"::notice::{m}")
def clamp(x,lo,hi): return max(lo,min(hi,x))
def parse_topN(scope:str)->Optional[int]:
    if scope=="all": return None
    m=re.match(r"top(\d+)$", scope)
    if m: return int(m.group(1))
    if scope=="top25": return 25
    return None

# ---------- helpers ----------
DATA_DIR = Path("data")/str(YEAR)
def need(path:Path, label:str)->pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_parquet(path)

def sr_from_map(m:Dict[str,float])->Optional[float]:
    for k,v in m.items():
        k=str(k).lower()
        if "success" in k and "%" in k:
            try: return float(v)/100.0
            except: return None
    return None

def main():
    log(f"Inputs → YEAR:{YEAR} WEEK:{WEEK} SCOPE:{SCOPE} MODE:{MODE}")

    # A) load dataset
    df_games = need(DATA_DIR/"games.parquet", "games")
    df_lines = need(DATA_DIR/"lines.parquet", "lines")
    df_ppa   = need(DATA_DIR/"ppa.parquet",   "ppa")
    df_off   = need(DATA_DIR/"stats_off.parquet", "stats offense")
    df_def   = need(DATA_DIR/"stats_def.parquet", "stats defense")
    df_spc   = need(DATA_DIR/"stats_spc.parquet", "stats special")

    # Normalize team key columns
    team_col = "team" if "team" in df_off.columns else ("school" if "school" in df_off.columns else None)
    if team_col is None:
        raise SystemExit("Could not find team key in stats_off.parquet")

    # B) discover current week matchups (prefer official games; fallback to lines)
    g = df_games[(df_games["seasonType"]=="regular") & (df_games["week"]==WEEK)]
    if g.empty:
        ln = df_lines[df_lines["week"]==WEEK].dropna(subset=["homeTeam","awayTeam"]).drop_duplicates(subset=["homeTeam","awayTeam"])
        g = ln[["homeTeam","awayTeam"]].copy(); g["neutralSite"]=False
    else:
        g = g[["homeTeam","awayTeam","neutralSite"]].copy()

    if g.empty:
        log("No games for requested week (even via lines).")
        return write_outputs([])

    # Optional AP topN → skip for now (no local poll yet), keep all
    if parse_topN(SCOPE) is not None:
        log("SCOPE topN requested; no local poll → keeping all games until polls are added.")

    # C) build maps from bulk tables
    def rows_to_map(df)->Dict[str, Dict[str,float]]:
        d={}
        for _,r in df.iterrows():
            team = r.get("team") or r.get("school")
            if not team: continue
            name = (r.get("statName") or r.get("stat_name") or "").lower()
            val  = r.get("statValue") or r.get("stat_value")
            try: val=float(val)
            except: pass
            d.setdefault(team,{})[name]=val
        return d

    off_stats = rows_to_map(df_off)
    def_stats = rows_to_map(df_def)
    spc_stats = rows_to_map(df_spc)

    off_ppa = {}
    def_ppa = {}
    for _,r in df_ppa.iterrows():
        t = r.get("team") or r.get("school")
        if not t: continue
        o = (r.get("offense") or {})
        d = (r.get("defense") or {})
        off_ppa[t] = float((o.get("overall") or o.get("ppa") or 0.0) or 0.0)
        def_ppa[t] = float((d.get("overall") or d.get("ppa") or 0.0) or 0.0)

    teams = sorted(set(g["homeTeam"]).union(g["awayTeam"]))

    # D) recency from season games (weeks < current)
    rec_by_team={}
    hist = df_games[(df_games["seasonType"]=="regular") & (df_games["week"]<WEEK)]
    for _,row in hist.iterrows():
        ht,at=row.get("homeTeam"),row.get("awayTeam")
        hp,ap=row.get("homePoints"),row.get("awayPoints")
        wk=row.get("week")
        if not (ht and at and isinstance(hp,int) and isinstance(ap,int) and isinstance(wk,int)): continue
        rec_by_team.setdefault(ht,[]).append({"wk":wk,"pf":hp,"pa":ap})
        rec_by_team.setdefault(at,[]).append({"wk":wk,"pf":ap,"pa":hp})

    def recency(h,a,n=4,scale=0.5)->float:
        def pdpg(rows):
            rows=sorted(rows,key=lambda r:r["wk"])[-n:]
            return sum(r["pf"]-r["pa"] for r in rows)/len(rows) if rows else 0.0
        return (pdpg(rec_by_team.get(h,[])) - pdpg(rec_by_team.get(a,[]))) * scale

    # E) drives (if present and FULL)
    fp_map = {t:{"osfp":25.0,"dsfp":25.0} for t in teams}
    drv_path = DATA_DIR/f"drives_week{WEEK}.parquet"
    if MODE=="FULL" and drv_path.exists():
        d = pd.read_parquet(drv_path)
        by_off,by_def={},{}
        for _,r in d.iterrows():
            ytg=r.get("start_yards_to_goal"); 
            if ytg is None: continue
            start = 100 - ytg
            off=r.get("offense"); de=r.get("defense")
            if off: by_off.setdefault(off,[]).append(start)
            if de:  by_def.setdefault(de, []).append(start)
        for t in teams:
            if by_off.get(t): fp_map[t]["osfp"]=sum(by_off[t])/len(by_off[t])
            if by_def.get(t): fp_map[t]["dsfp"]=sum(by_def[t])/len(by_def[t])

    # F) component functions
    def team_epp(o_ppa, d_ppa): return clamp(BASE_EPP + (o_ppa - d_ppa), 0.10, 0.80)
    def pace_total(h,a):
        def ppg(m):
            plays=m.get("plays"); gms=m.get("games") or m.get("gp") or m.get("gms")
            try: return float(plays)/float(gms) if plays and gms else None
            except: return None
        return (ppg(off_stats.get(h,{})) or 65.0) + (ppg(off_stats.get(a,{})) or 65.0)
    def pace_scale(x, total, base=130.0, eps=0.5): return x*(1.0 + eps*((total-base)/base))
    def fp_points(h,a,ppyard=0.06):
        exp_h = 0.5*fp_map[h]["osfp"] + 0.5*fp_map[a]["dsfp"]
        exp_a = 0.5*fp_map[a]["osfp"] + 0.5*fp_map[h]["dsfp"]
        return (exp_h-exp_a)*ppyard
    def hidden_yards(h,a,ppyard=0.055):
        sh,sa=spc_stats.get(h,{}), spc_stats.get(a,{})
        net_h = sh.get("netpunting",0) or (sh.get("puntyards",0)-sh.get("opponentpuntreturnyards",0))/max(1, sh.get("punts",1))
        net_a = sa.get("netpunting",0) or (sa.get("puntyards",0)-sa.get("opponentpuntreturnyards",0))/max(1, sa.get("punts",1))
        ko_h = (sh.get("kickreturnyards",0)-sh.get("opponentkickreturnyards",0))/max(1, sh.get("kickreturns",1))
        ko_a = (sa.get("kickreturnyards",0)-sa.get("opponentkickreturnyards",0))/max(1, sa.get("kickreturns",1))
        return ((net_h-net_a) + 0.5*(ko_h-ko_a))*ppyard
    def success_points(h,a, per5=1.5):
        sh_off = sr_from_map(off_stats.get(h,{}))
        sa_def = sr_from_map(def_stats.get(a,{}))
        sa_off = sr_from_map(off_stats.get(a,{}))
        sh_def = sr_from_map(def_stats.get(h,{}))
        pts=0.0
        if sh_off is not None and sa_def is not None: pts += ((sh_off-sa_def)/0.05)*per5
        if sa_off is not None and sh_def is not None: pts -= ((sa_off-sh_def)/0.05)*per5
        return pts
    def explosiveness_points(h,a, scale=SCALE_PER_0P10):
        oh,da=off_ppa.get(h,0.0), def_ppa.get(a,0.0)
        oa,dh=off_ppa.get(a,0.0), def_ppa.get(h,0.0)
        return ((oh-da)-(oa-dh))/0.10*scale
    def havoc_points(h,a, scale=3.0):
        dh,da=def_stats.get(h,{}), def_stats.get(a,{})
        oh,oa=off_stats.get(h,{}), off_stats.get(a,{})
        def rate(d,o):
            tfl=(d.get("tfl",0) or d.get("tacklesforloss",0)) + (d.get("sacks",0) or 0)
            plays=d.get("plays",0) or 1; sa=o.get("sacksallowed",0) or 0
            return (tfl + sa)/max(1,plays)
        try: return (rate(dh,oa) - rate(da,oh))*scale
        except: return 0.0

    # G) vegas averages (for reference)
    v = (df_lines.groupby(["homeTeam","awayTeam"])
          .agg(vegas_spread=("spread","mean"), vegas_total=("overUnder","mean"))
          .reset_index())

    # H) compute rows
    out=[]
    for _,row in g.iterrows():
        h,a=row["homeTeam"], row["awayTeam"]
        hfa = NEUTRAL_HFA if bool(row.get("neutralSite")) else HFA_DEFAULT
        base = ((off_ppa.get(h,0.0)-def_ppa.get(a,0.0)) - (off_ppa.get(a,0.0)-def_ppa.get(h,0.0))) / 0.10 * SCALE_PER_0P10 + hfa
        plays = pace_total(h,a)

        xpl_raw = pace_scale(explosiveness_points(h,a), plays)
        sr_raw  = pace_scale(success_points(h,a), plays)
        hv_raw  = pace_scale(havoc_points(h,a), plays)
        fp_raw  = fp_points(h,a) if MODE=="FULL" else 0.0
        hy_raw  = hidden_yards(h,a) if MODE=="FULL" else 0.0
        rcy_raw = recency(h,a)

        def cap(n,v): return clamp(v*WEIGHTS.get(n,1.0), -COMP_LIMITS.get(n,6.0), COMP_LIMITS.get(n,6.0))
        fp,hy,xpl,sr,hv,rcy = cap("fp",fp_raw), cap("hidden",hy_raw), cap("xpl",xpl_raw), cap("sr",sr_raw), cap("havoc",hv_raw), cap("recency",rcy_raw)
        adj = clamp(base + fp + hy + xpl + sr + hv + rcy, SPREAD_MIN, SPREAD_MAX)

        epp_h = clamp(BASE_EPP + (off_ppa.get(h,0.0)-def_ppa.get(a,0.0)), 0.10, 0.80)
        epp_a = clamp(BASE_EPP + (off_ppa.get(a,0.0)-def_ppa.get(h,0.0)), 0.10, 0.80)
        total = epp_h*(plays/2.0) + epp_a*(plays/2.0) + 0.5*(xpl + sr)
        total = int(round(clamp(total, TOTAL_MIN, TOTAL_MAX)))

        hp = int(round((total + adj)/2)); ap = int(round(total - hp))
        hp=max(0,hp); ap=max(0,ap)
        fav = h if adj>=0 else a

        row_v = v[(v["homeTeam"]==h) & (v["awayTeam"]==a)]
        vs = float(row_v["vegas_spread"].mean()) if not row_v.empty else None
        vt = float(row_v["vegas_total"].mean()) if not row_v.empty else None

        out.append({
            "home":h,"away":a,"favored":fav,
            "base_spread":round(base,1),"adj_spread":round(adj,1),
            "home_pts":hp,"away_pts":ap,"total_pts":int(total),
            "plays_est":int(round(plays)),
            "fp":round(fp,2),"hidden":round(hy,2),"xpl":round(xpl,2),
            "sr":round(sr,2),"havoc":round(hv,2),"recency":round(rcy,2),
            "vegas_spread":vs,"vegas_total":vt
        })

    out = sorted(out, key=lambda r: abs(r["adj_spread"]), reverse=True)
    return write_outputs(out)

def write_outputs(rows:List[Dict[str,Any]])->int:
    Path("docs").mkdir(parents=True, exist_ok=True)
    with open("docs/week_preds.json","w",encoding="utf-8") as f: json.dump(rows, f, ensure_ascii=False)
    pd.DataFrame(rows).to_csv("week_preds.csv", index=False)
    log(f"Wrote {len(rows)} rows → docs/week_preds.json & week_preds.csv")
    return 0

if __name__=="__main__":
    raise SystemExit(main())