# Waterlogging heatmap web app

FastAPI backend + Leaflet frontend. Same method as `waterlogging_heatmap_v2.ipynb`:
Copernicus DEM, ESA WorldCover, SoilGrids, OSM (roads, buildings, drains), Sentinel-2
-> terrain features + SCS-CN runoff - drainage -> heatmap per rainfall scenario,
robust-hotspot map, optional evaluation against reported flood points.

## Run

    pip install -r requirements.txt
    uvicorn app.main:app --reload
    # open http://127.0.0.1:8000

Needs internet access to S3 (DEM, WorldCover), Overpass, SoilGrids and Earth Search.

## What the app gives you

- **Heatmaps** per rainfall scenario plus a robust-hotspot map (share of 30 perturbed runs).
- **Top hotspots**: clusters of robust-hotspot cells ranked by total heat, numbered on the map, with area,
  average heat, and the building area and road length within about 30 m. Click one in the list to zoom to it.
  Building and road figures only cover what OpenStreetMap has mapped, and the heat is a relative index
  built on assumed drain capacity, so treat the ranking as "where to look first", not as a flood depth.
- **Downloads** for GIS: one GeoTIFF per scenario (millimetres of excess water), the robust map (0 to 1),
  hotspot outlines (GeoJSON) and a hotspot list (CSV). All in EPSG:4326. The last 8 runs stay downloadable
  until the server restarts.
- **Recent places**: every cached place is listed in the sidebar for one-click reuse; the x deletes its data.
- **Forecast rainfall**: "Use forecast peak" asks Open-Meteo for the next 3 days and sets the rainfall to the
  worst 3-hour total with a 3 h duration. It is a single point forecast, not a flood warning.

Cold runs download elevation and land cover together, then soil, the OSM layers and Sentinel-2 together
(the three OSM queries stay sequential because public Overpass servers throttle parallel requests).

| Endpoint | Purpose |
|---|---|
| `POST /api/run`, `GET /api/job/{id}` | Start an analysis, poll progress and result |
| `GET /api/export/{export_id}/{file}` | `heat_<i>.tif`, `robust.tif`, `hotspots.geojson`, `hotspots.csv` |
| `GET /api/places`, `DELETE /api/places/{area}` | List or delete cached places. No login, so do not expose the server publicly |
| `GET /api/forecast?lat=&lon=` | Peak 3 h and 24 h rainfall for the next 3 days |

## Caching

Three levels, so repeat analyses of a place are fast:

1. **Result cache** (memory, last 6 runs). The exact same request returns instantly.
2. **Terrain features** (memory + disk). The slow priority-flood step runs once per place and is
   reused for any rainfall, drain or storm settings, even after a restart.
3. **Downloaded layers** (disk, `./data`, override with `WL_DATA_DIR`). One file per layer per place:
   elevation, land cover, soil, buildings, roads, drains and the Sentinel-2 image.

Rules: a layer is cached as soon as it downloads, so if only Overpass fails, the next run fetches just
the OSM layers. Failures are never cached, and neither are terrain features or results built from a
fallback layer. "No data exists" (no SoilGrids value, no scene under the cloud limit) is remembered
for `WL_NEG_TTL_DAYS` (7). Sentinel-2 images expire after `WL_S2_TTL_DAYS` (14). Elevation, land cover,
soil and OSM layers do not expire: delete `./data` (or one place's folder) to refresh them. The folder is
trimmed to `WL_CACHE_MAX_MB` (500), removing the least recently used places first. Coordinates are
rounded to 4 decimals (about 11 m) and radius to 2 decimals so nearby clicks share an entry.

## Sentinel-2 parameters

The satellite image is a background layer only; it does not change the heatmap.

| Request field | Default | Meaning |
|---|---|---|
| `s2_enabled` | `true` | Set `false` to skip the download entirely |
| `s2_max_cloud` | `10` | Cloud tolerance, 0 to 100 %. Measured over the whole ~110 km scene, not your area |
| `s2_lookback_days` | `365` | How far back to search, 7 to 1095 days |
| `s2_prefer` | `"clearest"` | `"clearest"` = least cloudy scene under the limit, `"latest"` = newest scene under the limit |

A higher tolerance with `"latest"` gives a more recent but cloudier image. The sidebar shows the date and
cloud cover of the scene that was used.

## Retries

Each data source has its own retry limit (extra attempts after the first), so an outage fails fast:

| Source | Default | Environment override |
|---|---|---|
| Elevation (DEM) | 1 | `WL_RETRIES_DEM` |
| Land cover | 1 | `WL_RETRIES_LANDCOVER` |
| Soil | 0 | `WL_RETRIES_SOIL` |
| Buildings, roads, drains (Overpass) | 2 (each retry uses the next mirror) | `WL_RETRIES_BUILDINGS`, `_ROADS`, `_DRAINS` |
| Sentinel-2 | 0 | `WL_RETRIES_SENTINEL2` |

`WL_RETRY_DELAY` (default 1 s) is the pause between attempts. GDAL's own HTTP retry is capped at 1.

## Differences from the notebook

- Overpass: tries up to three mirror servers, one attempt each (the notebook run timed out on every OSM layer).
- Any missing optional layer shows as a warning in the sidebar instead of a silent fallback.
  With no buildings, roads or drains the heatmap is terrain + land cover only.
- Sentinel-2 search uses a configurable look-back window and cloud limit rather than hard-coded dates.
- Rainfall maps share one colour scale, as in the notebook. Radius is capped at 5 km (the
  priority-flood is pure Python, so a 5 km run takes several seconds).
- Flood CSV: `lat,lon[,label]`; label 1 = flooded, 0 = dry. Without 0s, random land cells are used as negatives.

Drain capacity (10 mm/h) is an assumption. Use a local design value.
