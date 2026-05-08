# Deploying to GitHub Pages

This site is fully static and ships with a graceful fallback for the parts of the
dashboard that normally hit a local Flask backend (`tileserver.py`). Everything
under this folder can be pushed to a GitHub repo and served via Pages without
any server-side runtime.

## TL;DR

```bash
# 1. one-time, in this folder
git init -b main
git add .gitignore .nojekyll
git add .                              # picks up everything else
git commit -m "Initial commit"
git remote add origin git@github.com:<you>/<repo>.git
git push -u origin main

# 2. on github.com:
#    Settings -> Pages -> Source: "Deploy from a branch"
#                       Branch:   main / (root)
```

The site will be live at `https://<you>.github.io/<repo>/`.

## What was set up for GitHub Pages

- `.nojekyll` — disables Jekyll so files starting with `_` are served verbatim.
- `.gitignore` — keeps editor metadata, Python venvs, the 405 MB raw GeoJSON,
  and `martin.zip` out of the repo. The two `.pmtiles` tilesets stay in the
  repo as ordinary files: GitHub Pages does **not** serve Git LFS objects
  through the Pages CDN, so anything the browser needs to fetch must be
  committed directly. The largest committed file (`vacancy_predictions.pmtiles`,
  ~46 MB) sits comfortably under GitHub's 100 MB per-file hard limit.

## How the static dashboard works

`dashboard.html` is fully static. It does not call `tileserver.py` or any
backend — every panel is derived client-side from files committed alongside
the HTML.

| Feature | Source |
|---|---|
| Map basemap | CARTO raster tiles (CDN) |
| Parcel layer + popups | `vacancy_predictions.pmtiles` (same-origin range requests) |
| SHAP risk drivers | `dashboard_shap.json` |
| Ward choropleth | `ward_stats.json` + `ward_boundaries.geojson` |
| Sidebar summary cards | Aggregated from `ward_stats.json` (citywide totals) |
| Ward filter list + fly-to | `ward_stats.json` for stats, bounds computed from `ward_boundaries.geojson` |
| Census tract filter | Derived from `vacancy_predictions_flagged.geojson` (lazy-loaded on first tract search) |
| Parcel search | Filters `vacancy_predictions_flagged.geojson` by parcel number / address (flagged subset only, ~3.9 MB, lazy-loaded) |

Search is intentionally limited to the flagged subset. The full ~440K-parcel
index would be too heavy to ship as static JSON; flagged parcels are what a
user of this tool would realistically search for.

The sidebar is decoupled from PMTiles: if `vacancy_predictions.pmtiles` fails
to load, the map area shows an inline error but the metrics, ward list, and
chart still render.

## Hosting the large files

The two PMTiles tilesets are the heavy pieces:

- `vacancy_predictions.pmtiles` (~46 MB) — full citywide.
- `vacancy_flagged.pmtiles` (~2.2 MB) — top 1 % flagged subset.

Both are tracked via Git LFS by default. If you'd rather host them on S3,
Cloudflare R2, or any static bucket, set the `PMTILES_URL` constant at the top
of the `<script>` block in `dashboard.html` to the absolute URL — the
PMTiles JS reader streams ranges over HTTP.

## Files to keep

The site at minimum needs:

- `dashboard.html`, `Vacancy Risk Landing Page.html`, `PhillyStat360 v2.html`
- `assets/`, `fonts/`, `colors_and_type.css`
- `ward_stats.json`, `ward_boundaries.geojson`, `dashboard_shap.json`
- `vacancy_predictions.pmtiles`, `vacancy_flagged.pmtiles`

`tileserver.py`, `load_db.py`, the raw `*.geojson` exports, and `martin.zip`
are only needed if you're regenerating the data locally — they can be dropped
from a deployment if repo size matters.
