# -------------------------------
# #3 Weekly run  (resilient to empty/missing season cache)
# -------------------------------
def main() -> int:
    log(f"Inputs → YEAR:{YEAR} WEEK:{WEEK} SCOPE:{SCOPE} MODE:{MODE}")

    # Prefer cached games; if none, fetch this week live (and the loader will cache it)
    games = games_for_week_from_cache(G_SEASON, YEAR, WEEK)
    if not games:
        log(f"#1 No cached games for week {WEEK}; fetching live…")
        live = jget("/games", {"year": YEAR, "week": WEEK, "seasonType": "regular"}, ttl_hours=3) or []
        games = [g for g in live if g.get("year")==YEAR and g.get("seasonType")=="regular" and g.get("week")==WEEK]

    gdf = pd.DataFrame(games)
    if gdf.empty:
        log("#1 No games returned after fallback. Writing empty outputs.")
        return write_outputs([])

    # Scope filter: only apply Top-N if we actually have AP rankings for the week
    filtered = gdf
    N = parse_topN(SCOPE)
    if N is not None:
        ap = set(ap_topN_for_week(R_SEASON, WEEK, N))
        if ap:
            filtered = gdf[gdf["homeTeam"].isin(ap) | gdf["awayTeam"].isin(ap)]
            log(f"#2 Scope=top{N}: kept {len(filtered)} of {len(gdf)} games")
        else:
            log(f"#2 No rankings in cache → ignoring Top-{N} filter (keeping all {len(gdf)} games)")
    gdf = filtered
    if gdf.empty:
        log("#2 Scope filter left zero games. Writing empty outputs.")
        return write_outputs([])

    teams = teams_in_week(gdf.to_dict("records"))

    # --- PPA maps. Use season snapshot when available; otherwise fetch per-team (small # of calls).
    off_map, def_map = {}, {}
    if PPA:
        for row in PPA:
            t = row.get("team")
            off = ((row.get("offense") or {}).get("overall")
                   or (row.get("offense") or {}).get("ppa") or 0.0) or 0.0
            deff = ((row.get("defense") or {}).get("overall")
                    or (row.get("defense") or {}).get("ppa") or 0.0) or 0.0
            if t in teams:
                off_map[t] = float(off); def_map[t] = float(deff)

    # Per-team fallback for any missing PPA
    missing_for = [t for t in teams if t not in off_map or t not in def_map]
    if missing_for:
        log(f"#3 PPA snapshot missing {len(missing_for)} teams → fetching per-team (cached)…")
        for t in missing_for:
            rows = jget("/ppa/teams", {"year": YEAR, "team": t}, ttl_hours=6) or []
            # API returns a list with one object for the team
            for r in rows:
                if (r.get("team") or "").lower() == t.lower():
                    off = ((r.get("offense") or {}).get("overall")
                           or (r.get("offense") or {}).get("ppa") or 0.0) or 0.0
                    deff = ((r.get("defense") or {}).get("overall")
                            or (r.get("defense") or {}).get("ppa") or 0.0) or 0.0
                    off_map[t] = float(off); def_map[t] = float(deff)

    # --- season stats (per-team, minimal live calls; cached by loader)
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

    # --- vegas lines from cached season lines; if missing, we’re fine (fields stay None)
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

    # --- compute rows (unchanged)
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