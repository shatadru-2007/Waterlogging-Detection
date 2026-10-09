import base64
import csv
import hashlib
import io
import json
import os
import threading
import uuid
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from rasterio.features import shapes
from rasterio.io import MemoryFile

from . import data as datamod
from .cache import LRU
from .model import FEATURES_VERSION, W0, Model

app = FastAPI(title="Waterlogging heatmap")
STATIC = os.path.join(os.path.dirname(__file__), "static")
# Terrain features are the slow part (also saved to disk, see compute). Finished results are kept in memory.
_F_CACHE, _R_CACHE, _X_CACHE = LRU(8), LRU(6), LRU(8)   # features, finished results, raw arrays for export


class FloodPoint(BaseModel):
    lat: float
    lon: float
    label: int = 1


class RunReq(BaseModel):
    lat: float = Field(ge=-85, le=85)
    lon: float = Field(ge=-180, le=180)
    radius_km: float = Field(1, gt=0, le=5)
    rain_cm: list[float] = Field([5, 10, 20], min_length=1, max_length=6)
    duration_h: float = Field(3, gt=0, le=72)
    drain_mm_h: float = Field(10, ge=0, le=200)   # ASSUMED drain capacity: replace with a local design value
    road_drain_frac: float = Field(.5, ge=0, le=1)
    flood_points: list[FloodPoint] = []
    use_calibrated: bool = False
    # Sentinel-2 background image (display only, it does not change the heatmap)
    s2_enabled: bool = True
    s2_max_cloud: int = Field(10, ge=0, le=100)             # scene-level cloud cover limit, %
    s2_lookback_days: int = Field(365, ge=7, le=1095)
    s2_prefer: Literal["clearest", "latest"] = "clearest"   # among scenes under the limit


def png(arr):
    buf = io.BytesIO()
    mpimg.imsave(buf, arr, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def colour(x, vmax, cmap, alpha=.8):
    import matplotlib as mpl
    rgba = mpl.colormaps[cmap](np.clip(np.nan_to_num(x) / vmax, 0, 1))
    rgba[..., 3] = np.where(np.isnan(x), 0, alpha)
    return png(rgba)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


JOBS = {}


def compute(q, prog):
    # Snap so near-identical requests share one cache entry, then see if this exact run was already done.
    lat, lon, R = datamod.snap(q.lat, q.lon, q.radius_km)
    q = q.model_copy(update=dict(lat=lat, lon=lon, radius_km=R))
    rkey = hashlib.sha1(q.model_dump_json().encode()).hexdigest()
    done = _R_CACHE.get(rkey)
    if done is not None and _X_CACHE.get(rkey) is not None:   # export arrays must still be available
        prog(95, "Using cached result")
        return {**done, "cache": {**done["cache"], "result": True}}

    s2p = datamod.S2Params(q.s2_enabled, q.s2_max_cloud, q.s2_lookback_days, q.s2_prefer)
    d = datamod.load(lat, lon, R, prog, s2p)

    # Terrain features: memory -> disk -> compute. Never cached when buildings/roads fell back to empty.
    area = datamod.CACHE.area(lat, lon, R)
    fname, fkey, F, fsrc = f"features_v{FEATURES_VERSION}", (area, FEATURES_VERSION), None, None
    if d.terrain_ok:
        F = _F_CACHE.get(fkey)
        fsrc = "memory" if F is not None else None
        if F is None:
            hit = datamod.CACHE.load(area, fname)
            if hit is not None and all(k in hit for k in W0):
                F, fsrc = {k: hit[k] for k in W0}, "disk"
    prog(72, "Preparing terrain layers" if F is not None else "Computing terrain layers (slowest step)")
    m = Model(d, lat, lon, R, q.duration_h, q.drain_mm_h, q.road_drain_frac, F)
    if d.terrain_ok:
        if fsrc != "memory":
            _F_CACHE.put(fkey, m.F)
        if fsrc is None:
            datamod.CACHE.save(area, fname, **m.F)

    ev, w = None, W0
    if q.flood_points:
        prog(82, "Checking against reported points")
        try:
            ev = m.evaluate([p.model_dump() for p in q.flood_points])
        except ImportError:
            ev = dict(lines=["Install scikit-learn to evaluate against reported points."], W_CAL=None, pts=[])
        if q.use_calibrated and ev["W_CAL"]:
            w = ev["W_CAL"]

    prog(86, "Running rainfall scenarios")
    heats = m.scenario_maps(q.rain_cm, w)
    vmax = float(np.nanmax(heats)) if np.isfinite(heats).any() else 0.0
    vmax = vmax if vmax > 0 else 1.0
    prog(91, "Finding robust hotspots")
    robust = m.robust(max(q.rain_cm) * 10, w)
    prog(94, "Ranking hotspots")
    items, exposure, lab = m.hotspots(robust, heats[int(np.argmax(q.rain_cm))])
    _X_CACHE.put(rkey, dict(T=d.T, rain=list(q.rain_cm), heats=heats, robust=robust, items=items, lab=lab))
    prog(96, "Drawing maps")
    west, east, south, north = m.bounds
    result = {
        "bounds": [[south, west], [north, east]],
        "vmax": vmax,
        "scenarios": [{"rain_cm": p, "image": colour(h, vmax, "turbo")} for p, h in zip(q.rain_cm, heats)],
        "robust": {"rain_cm": max(q.rain_cm), "image": colour(robust, 1, "magma", .85),
                   "pct_ge80": float(100 * np.nanmean(robust >= .8))},
        "export_id": rkey,
        "hotspots": items,
        "exposure": exposure,
        "satellite": png(d.rgb.transpose(1, 2, 0)) if d.rgb.max() else None,
        "satellite_info": d.s2 and {**d.s2, "max_cloud": q.s2_max_cloud, "prefer": q.s2_prefer},
        "cache": {"layers": d.cached, "total": d.total_layers, "features": fsrc, "result": False},
        "stats": {"grid": list(d.dem.shape), "soil_group": "ABCD"[d.hsg], "land_cells": int(m.land.sum()),
                  "drains_km_per_km2": round(m.drain_km2, 2), "major_roads_km_per_km2": round(m.major_km2, 2),
                  "underpass_cells": int(d.uw.sum()), "weights": w},
        "warnings": d.warnings + (["Mapped drain density is below 0.5 km/km2, so results lean on the road-drain fraction. Use municipal drain data if you can."]
                                  if m.drain_km2 < .5 else []),
        "evaluation": ev and {"lines": ev["lines"], "points": ev["pts"], "calibrated": ev["W_CAL"]},
    }
    if not d.warnings:   # a run with missing layers is never frozen into the result cache
        _R_CACHE.put(rkey, result)
    return result


def _work(job, q):
    def prog(pct, msg):
        job.update(pct=pct, stage=msg)
    try:
        result = compute(q, prog)
        job.update(result=result, pct=100, stage="Done", status="done")
    except Exception as e:  # noqa: BLE001
        msg = (f"Could not download the elevation or land-cover data ({type(e).__name__}: {str(e)[:160]})."
               if job["pct"] < 30 else f"The analysis failed ({type(e).__name__}: {str(e)[:160]}).")
        job.update(status="error", error=msg)


@app.post("/api/run")
def run(q: RunReq):
    """Start an analysis; poll /api/job/{id} for progress and the result."""
    for k in list(JOBS)[:-20]:
        JOBS.pop(k, None)
    jid = uuid.uuid4().hex
    JOBS[jid] = job = dict(status="running", pct=0, stage="Starting", result=None, error=None)
    threading.Thread(target=_work, args=(job, q), daemon=True).start()
    return {"job_id": jid}


@app.get("/api/job/{jid}")
def job_status(jid: str):
    job = JOBS.get(jid)
    if job is None:
        raise HTTPException(404, "Unknown job. Run the analysis again.")
    return job if job["status"] == "done" else {k: job[k] for k in ("status", "pct", "stage", "error")}


# ---------- exports (GIS-ready files for the most recent runs) ----------
def _tif(arr, T):
    a = np.where(np.isfinite(arr), arr, -9999).astype("float32")
    with MemoryFile() as mf:
        with mf.open(driver="GTiff", height=a.shape[0], width=a.shape[1], count=1, dtype="float32", crs="EPSG:4326",
                     transform=T, nodata=-9999, compress="deflate") as dst:
            dst.write(a, 1)
        return mf.read()


@app.get("/api/export/{rid}/{name}")
def export(rid: str, name: str):
    """heat_<i>.tif (mm), robust.tif (0-1), hotspots.geojson, hotspots.csv for a finished run."""
    x = _X_CACHE.get(rid)
    if x is None:
        raise HTTPException(404, "This result is no longer stored. Run the analysis again.")
    send = lambda body, mime, fn: Response(body, media_type=mime, headers={"Content-Disposition": f'attachment; filename="{fn}"'})
    if name == "robust.tif":
        return send(_tif(x["robust"], x["T"]), "image/tiff", "robust_hotspots.tif")
    if name.startswith("heat_") and name.endswith(".tif") and name[5:-4].isdigit() and int(name[5:-4]) < len(x["heats"]):
        i = int(name[5:-4])
        return send(_tif(x["heats"][i], x["T"]), "image/tiff", f"heat_{x['rain'][i]:g}cm_mm.tif")
    if name == "hotspots.csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["rank", "lat", "lon", "area_m2", "mean_heat_mm", "buildings_m2_nearby", "roads_m_nearby"])
        for it in x["items"]:
            w.writerow([it["rank"], f"{it['lat']:.6f}", f"{it['lon']:.6f}", it["area_m2"], it["mean_mm"], it["bld_m2"], it["road_m"]])
        return send(buf.getvalue(), "text/csv", "hotspots.csv")
    if name == "hotspots.geojson":
        by_id = {it["id"]: it for it in x["items"]}
        feats = [{"type": "Feature", "geometry": g, "properties": {k: v for k, v in by_id[int(val)].items() if k != "id"}}
                 for g, val in shapes(x["lab"].astype("int32"), mask=np.isin(x["lab"], list(by_id)), transform=x["T"])]
        return send(json.dumps({"type": "FeatureCollection", "features": feats}), "application/geo+json", "hotspots.geojson")
    raise HTTPException(404, "Unknown export.")


# ---------- places already analysed (disk cache) ----------
@app.get("/api/places")
def places():
    return datamod.CACHE.places()[:15]


@app.delete("/api/places/{area}")
def forget_place(area: str):
    if not datamod.CACHE.forget(area):
        raise HTTPException(404, "Unknown place.")
    for c in (_F_CACHE, _R_CACHE, _X_CACHE):
        c.clear()
    return {"ok": True}


# ---------- rainfall forecast ----------
@app.get("/api/forecast")
def forecast(lat: float = Query(ge=-85, le=85), lon: float = Query(ge=-180, le=180)):
    """Peak 3 h and 24 h rainfall in the next 3 days (Open-Meteo), to use as a scenario."""
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=15, params=dict(
            latitude=lat, longitude=lon, hourly="precipitation", forecast_days=3, timezone="auto"))
        r.raise_for_status()
        mm = np.array([v or 0 for v in r.json()["hourly"]["precipitation"]], float)
        if not len(mm):
            raise ValueError("empty forecast")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Forecast unavailable ({type(e).__name__}).")
    peak = lambda k: float(np.convolve(mm, np.ones(k), "valid").max()) if len(mm) >= k else float(mm.sum())
    return dict(peak_3h_cm=round(peak(3) / 10, 1), peak_24h_cm=round(peak(24) / 10, 1),
                total_cm=round(float(mm.sum()) / 10, 1), hours=int(len(mm)), source="Open-Meteo")
