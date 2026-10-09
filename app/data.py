"""Data download for one area (port of the notebook's 'data download' cell).

Required layers (DEM, land cover) raise on failure. Optional layers fall back to an empty/default
value and add a human-readable warning, so the UI can tell the user what is missing.

Caching: every layer that downloaded successfully is saved on its own (see cache.py), so a failed
Overpass call costs only that layer on the next run and nothing already downloaded is fetched twice.
A failure is never cached. The one exception is "no data exists" (no SoilGrids value, no Sentinel-2
scene under the cloud limit), which is remembered for a few days so it is not re-queried every run.

Speed: elevation and land cover download together, then soil, the OSM layers and Sentinel-2 download
together. The three OSM layers stay one after another on purpose (public Overpass servers rate-limit
parallel requests from one address).

Retries: each data source has its own small retry limit (DEFAULT_RETRIES, overridable with
WL_RETRIES_<NAME>) so an outage fails fast instead of stalling the run.
"""
import datetime as dt
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import rasterio
import requests
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

from .cache import DiskCache

DATA_DIR = os.environ.get("WL_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
CACHE = DiskCache(DATA_DIR, float(os.environ.get("WL_CACHE_MAX_MB", 500)))
S2_TTL_DAYS = float(os.environ.get("WL_S2_TTL_DAYS", 14))     # new scenes keep arriving, so satellite images expire
NEG_TTL_DAYS = float(os.environ.get("WL_NEG_TTL_DAYS", 7))    # how long "no data exists" is remembered
RETRY_DELAY = float(os.environ.get("WL_RETRY_DELAY", 1))      # seconds between attempts

# Extra attempts after the first one, per data source. 0 = a single try.
# Overpass: each retry moves on to the next mirror, so 2 = every mirror once.
DEFAULT_RETRIES = {"dem": 1, "landcover": 1, "soil": 0, "buildings": 2, "roads": 2, "drains": 2, "sentinel2": 0}

OVERPASS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
STAC = "https://earth-search.aws.element84.com/v1/search"
MAJOR = ("motorway", "trunk", "primary", "secondary", "tertiary")
_lock = threading.Lock()


class NoData(Exception):
    """The source answered, but has nothing for this area (not a network failure)."""


def retries(name):
    try:
        return max(0, int(os.environ.get(f"WL_RETRIES_{name.upper()}", DEFAULT_RETRIES[name])))
    except ValueError:
        return DEFAULT_RETRIES[name]


def _retry(fn, name):
    n = retries(name)
    for attempt in range(n + 1):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if attempt == n:
                raise
            time.sleep(RETRY_DELAY)


def _gdal(name):
    """GDAL's own HTTP retry (for 429/5xx) is capped at 1 and is off when the source has 0 retries."""
    return rasterio.Env(GDAL_HTTP_MAX_RETRY=str(min(retries(name), 1)), GDAL_HTTP_RETRY_DELAY=str(RETRY_DELAY),
                        GDAL_HTTP_TIMEOUT="60", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")


@dataclass(frozen=True)
class S2Params:
    enabled: bool = True
    max_cloud: int = 10            # scene-level cloud cover limit, percent
    lookback_days: int = 365
    prefer: str = "clearest"       # "clearest": least cloudy scene under the limit; "latest": newest scene under the limit

    def cache_name(self):
        return f"s2_c{self.max_cloud}_d{self.lookback_days}_{self.prefer}"


@dataclass
class Data:
    dem: np.ndarray
    T: Affine
    lc: np.ndarray
    rgb: np.ndarray
    bld: np.ndarray
    roads: np.ndarray
    major: np.ndarray
    uw: np.ndarray
    drn: np.ndarray
    hsg: int
    warnings: list = field(default_factory=list)
    terrain_ok: bool = True      # buildings AND roads are real data, so terrain features may be cached
    s2: dict = None              # {id, date, cloud} of the Sentinel-2 scene used, if any
    cached: list = field(default_factory=list)   # layers served from the disk cache
    total_layers: int = 0


def snap(lat, lon, radius_km):
    """Round the request so nearby clicks share one cache entry (4 decimals is ~11 m; the DEM cell is 30 m)."""
    return round(lat, 4), round(lon, 4), round(max(radius_km, 0.05), 2)


def load(lat, lon, radius_km, progress=None, s2=None):
    with _lock:  # one download at a time
        return _load(lat, lon, radius_km, progress or (lambda pct, msg: None), s2 or S2Params())


def _load(lat, lon, radius_km, progress, s2p):
    dl = radius_km / 111.32
    dn = dl / math.cos(math.radians(lat))
    west, south, east, north = lon - dn, lat - dl, lon + dn, lat + dl
    area = CACHE.area(lat, lon, radius_km)
    cached, warnings, failed = [], [], set()
    _p, _pl = [0], threading.Lock()
    raw_progress = progress

    def progress(pct, msg):  # downloads run in parallel, so keep the bar from jumping backwards
        with _pl:
            _p[0] = max(_p[0], pct)
            raw_progress(_p[0], msg)

    ns = lambda v, a, b: a if v >= 0 else b

    def tiles(step, fmt):
        return sorted({fmt(math.floor(y / step) * step, math.floor(x / step) * step)
                       for y in (south, north) for x in (west, east)})

    def mosaic(urls, name):
        """Read and merge remote GeoTIFFs, retrying up to retries(name) times."""
        def read():
            src = []
            try:
                with _gdal(name):
                    for u in urls:
                        src.append(rasterio.open(u))
                    return merge(src, bounds=(west, south, east, north))
            finally:
                for x in src:
                    x.close()
        return _retry(read, name)

    def regrid(a, t, dt_, shape, rs):
        o = np.zeros(shape, a.dtype)
        reproject(a, o, src_transform=t, src_crs="EPSG:4326", dst_transform=dt_, dst_crs="EPSG:4326", resampling=rs)
        return o

    def osm(sel, name):
        """Overpass query; every retry goes to the next mirror instead of hammering the same one."""
        q = f"[out:json][timeout:60];way{sel}({south},{west},{north},{east});out geom;"
        n, last = retries(name), None
        for attempt in range(n + 1):
            try:
                r = requests.post(OVERPASS[attempt % len(OVERPASS)], data=q, timeout=75)
                r.raise_for_status()
                return r.json()["elements"]
            except Exception as e:  # noqa: BLE001
                last = e
                if attempt < n:
                    time.sleep(RETRY_DELAY)
        raise RuntimeError(f"Overpass failed after {n + 1} attempt(s) ({type(last).__name__})")

    def burn(els, poly, shape, T):
        g = [e["geometry"] for e in els if len(e.get("geometry", [])) > (3 if poly else 1)]
        sh = [{"type": "Polygon", "coordinates": [[(p["lon"], p["lat"]) for p in e]]} if poly else
              {"type": "LineString", "coordinates": [(p["lon"], p["lat"]) for p in e]} for e in g]
        return rasterize(sh, shape, transform=T, all_touched=not poly).astype(bool) if sh else np.zeros(shape, bool)

    def optional(name, fn, default, pct, file, pack, unpack, ttl=None):
        """Cache hit -> use it. Otherwise download; success is cached, failure becomes a warning."""
        hit = CACHE.load(area, file, ttl)
        if hit is not None:
            progress(pct, f"Loaded {name} from cache")
            cached.append(name)
            return unpack(hit)
        miss = CACHE.load(area, file + ".none", NEG_TTL_DAYS)
        if miss is not None:
            progress(pct, f"Loaded {name} from cache")
            cached.append(name)
            warnings.append(str(miss["msg"]))
            return default
        progress(pct, f"Downloading {name}")
        try:
            val = fn()
        except NoData as e:
            warnings.append(str(e))
            CACHE.save(area, file + ".none", msg=np.array(str(e)))
            return default
        except Exception as e:  # noqa: BLE001
            failed.add(name)
            warnings.append(f"{name.capitalize()} unavailable ({type(e).__name__}: {str(e)[:90]}); using a fallback")
            return default
        CACHE.save(area, file, **pack(val))
        return val

    # --- required layers (cached separately, so a failed land-cover download keeps the DEM) ---
    dfmt = lambda a, b: f"{ns(a,'N','S')}{abs(a):02d}_00_{ns(b,'E','W')}{abs(b):03d}_00"
    wfmt = lambda a, b: f"{ns(a,'N','S')}{abs(a):02d}{ns(b,'E','W')}{abs(b):03d}"
    dem_urls = [f"https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{n}_DEM/Copernicus_DSM_COG_10_{n}_DEM.tif"
                for n in tiles(1, dfmt)]
    lc_urls = [f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{n}_Map.tif"
               for n in tiles(3, wfmt)]
    dem_hit, lc_hit = CACHE.load(area, "dem"), CACHE.load(area, "landcover")
    ex = ThreadPoolExecutor(2)
    try:
        if dem_hit is None or lc_hit is None:
            progress(2, "Downloading elevation and land cover")
        f_dem = ex.submit(mosaic, dem_urls, "dem") if dem_hit is None else None
        f_lc = ex.submit(mosaic, lc_urls, "landcover") if lc_hit is None else None
        dem_new = f_dem.result() if f_dem else None
        lc_new = f_lc.result() if f_lc else None
    except Exception:
        ex.shutdown(wait=False, cancel_futures=True)   # fail fast: don't make the user wait for the other download
        raise
    ex.shutdown()

    if dem_hit is not None:
        cached.append("elevation")
        dem, T = dem_hit["dem"], Affine(*dem_hit["t"])
    else:
        dem, T = dem_new
        dem = dem[0].astype("float64")
        CACHE.save(area, "dem", dem=dem, t=np.array(T)[:6])
    shp = dem.shape

    if lc_hit is not None:
        cached.append("land cover")
        lc = lc_hit["lc"]
    else:
        lc0, lt = lc_new
        lc = regrid(lc0[0], lt, T, shp, Resampling.mode)
        CACHE.save(area, "landcover", lc=lc)
    progress(25, "Elevation and land cover ready" + (" (cache)" if dem_hit is not None and lc_hit is not None else ""))

    # --- optional layers ---
    def soil():  # texture -> hydrologic soil group (0=A..3=D); SoilGrids is masked over built-up pixels, so probe nearby
        k = radius_km / 111.32 / 2
        kx = k / math.cos(math.radians(lat))

        def query(dy, dx):
            r = requests.get("https://rest.isric.org/soilgrids/v2.0/properties/query", timeout=60,
                             params={"lon": lon + dx, "lat": lat + dy, "property": ["clay", "sand"],
                                     "depth": "0-5cm", "value": "mean"})
            r.raise_for_status()
            return r.json()

        for dy, dx in [(0, 0), (k, 0), (-k, 0), (0, kx), (0, -kx)]:
            j = _retry(lambda: query(dy, dx), "soil")
            v = {l["name"]: l["depths"][0]["values"]["mean"] for l in j["properties"]["layers"]}
            if None not in v.values():
                c, s = v["clay"] / 10, v["sand"] / 10
                return 0 if s > 70 and c < 15 else 1 if c < 20 else 2 if c < 35 else 3
        raise NoData("Soil: SoilGrids has no data near this area (probably built-up), so soil group C is assumed.")

    def road_masks():
        els = osm('["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified"]', "roads")
        tg = lambda e, k: e.get("tags", {}).get(k, "")
        ok = [e for e in els if tg(e, "bridge") in ("", "no")]
        uwl = [e for e in ok if tg(e, "tunnel") not in ("", "no") or tg(e, "layer").startswith("-")]
        return (burn(ok, False, shp, T), burn([e for e in ok if tg(e, "highway") in MAJOR], False, shp, T),
                burn(uwl, False, shp, T))

    def s2():  # Sentinel-2 true colour chosen by the cloud limit, look-back window and preference
        end = dt.date.today()
        start = end - dt.timedelta(days=s2p.lookback_days)
        sortby = ({"field": "properties.eo:cloud_cover", "direction": "asc"} if s2p.prefer == "clearest"
                  else {"field": "properties.datetime", "direction": "desc"})

        def search():
            r = requests.post(STAC, timeout=60, json={
                "collections": ["sentinel-2-l2a"], "bbox": [west, south, east, north], "limit": 1,
                "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
                "query": {"eo:cloud_cover": {"lte": s2p.max_cloud}}, "sortby": [sortby]})
            r.raise_for_status()
            return r.json().get("features") or []

        feats = _retry(search, "sentinel2")
        if not feats:
            raise NoData(f"Sentinel-2: no scene with cloud cover up to {s2p.max_cloud}% in the last "
                         f"{s2p.lookback_days} days. Raise the cloud tolerance or the look-back window.")
        f = feats[0]

        def read():
            with _gdal("sentinel2"), rasterio.open(f["assets"]["visual"]["href"]) as s:
                o = np.zeros((3, shp[0] * 3, shp[1] * 3), "uint8")
                reproject(rasterio.band(s, [1, 2, 3]), o, dst_transform=T * Affine.scale(1 / 3),
                          dst_crs="EPSG:4326", resampling=Resampling.bilinear)
            return o

        rgb = _retry(read, "sentinel2")
        p = f.get("properties", {})
        cc = p.get("eo:cloud_cover")
        return rgb, dict(id=str(f.get("id", "")), date=str(p.get("datetime", ""))[:10],
                         cloud=float(cc) if cc is not None else None)

    z0 = np.zeros(shp, bool)

    def get_soil():
        return optional("soil", soil, 2, 32, "soil", lambda v: dict(hsg=np.array(v)), lambda h: int(h["hsg"]))

    def get_osm():   # one after another: public Overpass servers rate-limit parallel requests
        bld = optional("buildings", lambda: burn(osm('["building"]', "buildings"), True, shp, T), z0, 38,
                       "buildings", lambda v: dict(bld=v), lambda h: h["bld"])
        roads = optional("roads", road_masks, (z0, z0, z0), 47, "roads",
                         lambda v: dict(roads=v[0], major=v[1], uw=v[2]),
                         lambda h: (h["roads"], h["major"], h["uw"]))
        drn = optional("drains", lambda: burn(osm('["waterway"~"drain|ditch|canal|stream|river"]', "drains"), False, shp, T),
                       z0, 56, "drains", lambda v: dict(drn=v), lambda h: h["drn"])
        return bld, roads, drn

    def get_s2():
        return optional(
            "Sentinel-2 image", s2, (np.zeros((3, 1, 1), "uint8"), None), 63, s2p.cache_name(),
            lambda v: dict(rgb=v[0], id=np.array(v[1]["id"]), date=np.array(v[1]["date"]),
                           cloud=np.array(-1.0 if v[1]["cloud"] is None else v[1]["cloud"])),
            lambda h: (h["rgb"], dict(id=str(h["id"]), date=str(h["date"]),
                                      cloud=float(h["cloud"]) if float(h["cloud"]) >= 0 else None)),
            ttl=S2_TTL_DAYS)

    with ThreadPoolExecutor(3) as ex:
        f_soil, f_osm = ex.submit(get_soil), ex.submit(get_osm)
        f_s2 = ex.submit(get_s2) if s2p.enabled else None
        hsg = f_soil.result()
        bld, (roads, major, uw), drn = f_osm.result()
        rgb, s2info = f_s2.result() if f_s2 else (np.zeros((3, 1, 1), "uint8"), None)

    progress(70, "Data ready")
    return Data(dem, T, lc, rgb, bld, roads, major, uw, drn, int(hsg), warnings,
                terrain_ok=not ({"buildings", "roads"} & failed), s2=s2info, cached=cached,
                total_layers=6 + int(s2p.enabled))
