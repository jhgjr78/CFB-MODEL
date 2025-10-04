#!/usr/bin/env python3
# .github/runner/weekly.py
import argparse, json, os
from pathlib import Path
import math
import pandas as pd
import requests

# ---------- Defaults & guardrails ----------
HFA_DEFAULT = 2.0
NEUTRAL_HFA = 0.5
SCALE_PER_0P10 = 3.0
CAPS = dict(fp=6, hidden=4, xpl=10, sr=6, havoc=6, recency=6)
SPREAD_CLAMP = (-40, 40)
TOTAL_CLAMP = (30, 95)
PACE_BASELINE = 130
RECENCY_N = 4
RECENCY_SCALE = 0.5

CFBD = "https://api.collegefootballdata.com"
HEADERS = lambda: {"Authorization": f"Bearer {os.environ.get('CFBD_API_KEY','')}"}

def jload(p: Path):
    if p.exists():
        with open(p, "r", encoding="utf-8") as f: 
            return json.load(f)
    return None

def fetch_json(endpoint, params):
    # fallback if cache missing
    try:
        r = requests.get(CFBD+endpoint, headers=HEADERS(), params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def avg_line_for_matchup(lines, home, away):
    if not lines: return (None, None)
    vals_spread, vals_total = [], []
    for b in lines:
        try:
            if (b.get("homeTeam")==home and b.get("awayTeam")==away) or (b.get("homeTeam")==away and b.get("awayTeam")==home):
                for m in b.get("lines", []) or []:
                    s = m.get("spread")
                    t = m.get("overUnder")
                    if isinstance(s, (int,float)): vals_spread.append(float(s))
                    if isinstance(t, (int,float)): vals_total.append(float(t))
        except Exception:
            continue
    return (sum(vals_spread)/len(vals_spread) if vals_spread else None,
            sum(vals_total)/len(vals_total) if vals_total else None)

def comp_delta(a,b, key, default=0.0):
    try:
        return float(a.get(key, default)) - float(b.get(key, default))
    except Exception:
        return 0.0

def cap(v, lo, hi):
    return max(lo, min(hi, v))

def compute_row(game, team_metrics, lines, drives, scope_set, mode):
    home = game.get("home_team")
    away = game.get("away_team")
    neutral = bool(game.get("neutral_site"))
    if scope_set and (home not in scope_set and away not in scope_set):
        return None

    home_m = team_metrics.get(home, {})
    away_m = team_metrics.get(away, {})

    # Off/Def PPA (fallback 0)
    h_off = float(home_m.get("off_ppa", 0.0))
    h_def = float(home_m.get("def_ppa", 0.0))
    a_off = float(away_m.get("off_ppa", 0.0))
    a_def = float(away_m.get("def_ppa", 0.0))

    # Base spread: (home off vs away def) - (away off vs home def)
    ppa_gap = (h_off - a_def) - (a_off - h_def)
    base_spread = (ppa_gap / 0.10) * SCALE_PER_0P10
    hfa = NEUTRAL_HFA if neutral else HFA_DEFAULT
    base_spread += hfa

    # Components (use deltas if available)
    # Success rate / explosiveness / havoc proxies (safe if missing)
    sr = comp_delta(home_m, away_m, "sr_delta", 0.0)  # if provided; else 0
    xpl = comp_delta(home_m, away_m, "xpl_delta", 0.0)
    havoc = comp_delta(home_m, away_m, "havoc_delta", 0.0)

    # Pace & plays estimate (fallback = baseline)
    pace_adj = comp_delta(home_m, away_m, "pace_delta", 0.0)
    plays_est = PACE_BASELINE + 10.0 * pace_adj  # gentle tilt
    plays_est = max(110.0, min(155.0, plays_est))

    # Field position & hidden yards (FULL only, if drives cached)
    fp = 0.0
    hidden = 0.0
    if mode == "FULL" and drives:
        # crude: average start yardline differential by team
        def avg_start(team):
            starts = []
            for d in drives:
                try:
                    if d.get("offense") == team and isinstance(d.get("start_yardline"), (int,float)):
                        starts.append(float(d["start_yardline"]))
                except Exception:
                    pass
            return sum(starts)/len(starts) if starts else None
        hs = avg_start(home); as_ = avg_start(away)
        if hs is not None and as_ is not None:
            fp = (hs - as_) * 0.05  # scale into points
        # hidden yards proxy via average starting field + punt/kick deltas if present
        hidden = 0.5 * fp

    # Recency (simple): last N games point margin proxy via PPA deltas if we have any
    recency = RECENCY_SCALE * cap((ppa_gap / 0.10) * 1.0, -CAPS["recency"], CAPS["recency"])

    # Clamp components
    fp = cap(fp, -CAPS["fp"], CAPS["fp"])
    hidden = cap(hidden, -CAPS["hidden"], CAPS["hidden"])
    xpl = cap(xpl, -CAPS["xpl"], CAPS["xpl"])
    sr = cap(sr, -CAPS["sr"], CAPS["sr"])
    havoc = cap(havoc, -CAPS["havoc"], CAPS["havoc"])

    # Adjusted spread
    adj_spread = base_spread + fp + hidden + xpl + sr + havoc + recency
    adj_spread = cap(adj_spread, *SPREAD_CLAMP)

    # Totals: estimate EPP ~ blended offensive vs defensive PPA
    epp_base = ((h_off - a_def) + (a_off - h_def)) / 2.0
    total_pts = (epp_base * plays_est) + 0.5 * (xpl + sr)
    total_pts = cap(total_pts, *TOTAL_CLAMP)

    # Split into home/away projected points (simple)
    home_pts = (total_pts / 2.0) + (adj_spread / 2.0)
    away_pts = total_pts - home_pts

    vegas_spread, vegas_total = avg_line_for_matchup(lines, home, away)

    favored = home if adj_spread >= 0 else away
    return dict(
        home=home, away=away, favored=favored,
        base_spread=round(base_spread,2),
        adj_spread=round(adj_spread,2),
        home_pts=round(home_pts,1),
        away_pts=round(away_pts,1),
        total_pts=round(total_pts,1),
        plays_est=round(plays_est,1),
        fp=round(fp,2), hidden=round(hidden,2),
        xpl=round(xpl,2), sr=round(sr,2), havoc=round(havoc,2),
        recency=round(recency,2),
        vegas_spread=vegas_spread, vegas_total=vegas_total
    )

def build_team_metrics(ppa_teams):
    """
    Normalize CFBD /ppa/teams response to a minimal dict per team:
    { team: {off_ppa, def_ppa, sr_delta, xpl_delta, havoc_delta, pace_delta} }
    If the richer metrics aren't available, they default to 0 deltas.
    """
    out = {}
    if isinstance(ppa_teams, list):
        for t in ppa_teams:
            team = t.get("team")
            off = t.get("offense", {}) or {}
            de = t.get("defense", {}) or {}
            out[team] = dict(
                off_ppa=float(off.get("ppa", 0.0) or 0.0),
                def_ppa=float(de.get("ppa", 0.0) or 0.0),
                # optional extras if present
                sr_delta=float(off.get("successRate", 0.0) or 0.0) - float(de.get("successRate", 0.0) or 0.0),
                xpl_delta=float(off.get("explosiveness", 0.0) or 0.0) - float(de.get("explosiveness", 0.0) or 0.0),
                havoc_delta=float(off.get("havoc", 0.0) or 0.0) - float(de.get("havoc", 0.0) or 0.0),
                pace_delta=float(off.get("playsPerGame", 0.0) or 0.0) / 70.0 - float(de.get("playsPerGame", 0.0) or 0.0)/70.0,
            )
    return out

def choose_scope_set(rankings, scope):
    if scope == "all": return None
    nums = 25 if scope=="top25" else 10
    teams = set()
    # rankings format can be [{season, seasonType, week, polls:[{poll, ranks:[{rank, school}]}]}]
    try:
        latest = rankings[-1]
        for poll in latest.get("polls", []):
            # use AP or CFP if available; else first poll
            if poll.get("poll", "").lower() in ("ap", "cfp", "coaches"):
                for r in poll.get("ranks", [])[:nums]:
                    teams.add(r.get("school"))
    except Exception:
        pass
    return teams

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", required=True, type=int)
    ap.add_argument("--week", required=True, type=int)
    ap.add_argument("--scope", default="all", choices=["all","top25","top10"])
    ap.add_argument("--mode", default="FAST", choices=["FAST","FULL"])
    args = ap.parse_args()

    # Load caches (prefer)
    games_p = Path(f"data/weeks/{args.year}/week_{args.week}.games.json")
    ranks_p = Path(f"data/weeks/{args.year}/week_{args.week}.rankings.json")
    ppa_p   = Path(f"data/season_{args.year}/ppa_teams.json")
    drives_p = Path(f"data/weeks/{args.year}/week_{args.week}.drives.json")
    lines_p  = Path(f"data/weeks/{args.year}/week_{args.week}.lines.json")

    games = jload(games_p) or fetch_json("/games", {"year": args.year, "week": args.week, "seasonType":"regular"}) or []
    rankings = jload(ranks_p) or fetch_json("/rankings", {"year": args.year, "week": args.week}) or []
    ppa_teams = jload(ppa_p) or fetch_json("/ppa/teams", {"year": args.year}) or []
    drives = jload(drives_p) if args.mode=="FULL" else []
    lines = jload(lines_p) or []

    team_metrics = build_team_metrics(ppa_teams)
    scope_set = choose_scope_set(rankings, args.scope)

    rows = []
    for g in games or []:
        r = compute_row(g, team_metrics, lines, drives, scope_set, args.mode)
        if r:
            rows.append(r)

    # Write outputs
    Path("docs").mkdir(exist_ok=True)
    with open("docs/week_preds.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)

    df = pd.DataFrame(rows, columns=[
        "home","away","favored","base_spread","adj_spread","home_pts","away_pts",
        "total_pts","plays_est","fp","hidden","xpl","sr","havoc","recency",
        "vegas_spread","vegas_total"
    ])
    df.to_csv("week_preds.csv", index=False)
    print(f"Wrote {len(rows)} predictions.")

if __name__ == "__main__":
    main()