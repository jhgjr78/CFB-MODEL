# CFB-Model

## One-time setup
1) Add repo secret: `CFBD_API_KEY` (your CollegeFootballData API token).
2) Enable GitHub Pages (Settings → Pages → deploy from `main` / `docs`).

## Weekly flow
1) **Build Dataset** (Actions → *Build Dataset*)
   - Inputs: `year=2025`, `drive_weeks=1-6` (or the week you want)
   - Produces Parquet files under `data/2025/`.

2) **Weekly Runner** (Actions → *Weekly Runner*)
   - Inputs: `year=2025`, `week=6`, `scope=all`, `mode=FULL`
   - Produces:
     - `docs/week_preds.json` (the dashboard reads this)
     - `week_preds.csv` (downloadable artifact)

3) Open the Pages site (the URL GitHub shows in Settings → Pages).

## Tuning the model
- Edit `weights.json` to adjust caps/weights/HFA without touching code.
- Re-run **Weekly Runner**; it reads local Parquet (no API cost).

## Notes
- If `/games` is empty for a future week, the model discovers matchups from `lines.parquet`.
- For phone-only use: all editing can be done in GitHub Mobile; the dashboard is mobile-friendly.