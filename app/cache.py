"""Caches shared by the data loader and the API.

DiskCache  - one folder per analysed place, one compressed .npz per data layer. Files are written
             atomically (temp file + rename) so a crash never leaves a half-written layer, and the
             oldest-used places are deleted when the folder grows past a size cap.
LRU        - small thread-safe in-memory cache for objects that are expensive to rebuild.
"""
import os
import re
import shutil
import threading
import time
from collections import OrderedDict

import numpy as np

CACHE_VERSION = "v4"   # bump when the stored layout changes; older folders are simply ignored
_AREA_RE = re.compile(rf"^{CACHE_VERSION}_(-?\d+\.\d{{4}})_(-?\d+\.\d{{4}})_(\d+\.\d{{2}})$")
_BASE_LAYERS = {"dem", "landcover", "soil", "buildings", "roads", "drains"}


class DiskCache:
    def __init__(self, root, max_mb=500):
        self.root = root
        self.max_bytes = int(max_mb * 1024 * 1024)
        self._lock = threading.Lock()

    @staticmethod
    def area(lat, lon, radius_km):
        """Folder name for a place. Callers pass already-snapped coordinates (see data.snap)."""
        return f"{CACHE_VERSION}_{lat:.4f}_{lon:.4f}_{radius_km:.2f}"

    def _path(self, area, name):
        return os.path.join(self.root, area, name + ".npz")

    def load(self, area, name, max_age_days=None):
        """Return {array name: array} or None on a miss (absent, expired or unreadable)."""
        p = self._path(area, name)
        try:
            if max_age_days is not None and time.time() - os.path.getmtime(p) > max_age_days * 86400:
                return None
            with np.load(p, allow_pickle=False) as z:
                out = {k: z[k] for k in z.files}
        except FileNotFoundError:
            return None
        except Exception:  # noqa: BLE001 - corrupt file: drop it and download again
            self._remove(p)
            return None
        self._touch(area)
        return out

    def save(self, area, name, **arrays):
        """Best effort: a full or read-only disk must never fail an analysis."""
        p = self._path(area, name)
        tmp = f"{p}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **arrays)
            os.replace(tmp, p)
            self._touch(area)
            self._prune(keep=area)
        except Exception:  # noqa: BLE001
            self._remove(tmp)

    def places(self):
        """Analysed places on disk, most recently used first."""
        out = []
        try:
            entries = list(os.scandir(self.root))
        except OSError:
            return out
        for e in entries:
            m = _AREA_RE.match(e.name)
            if not (m and e.is_dir()):
                continue
            try:
                files = [f for f in os.scandir(e.path) if f.is_file() and f.name.endswith(".npz")]
                names = {f.name[:-4] for f in files}
                try:
                    used = os.path.getmtime(os.path.join(e.path, ".used"))
                except OSError:
                    used = e.stat().st_mtime
                out.append(dict(
                    area=e.name, lat=float(m[1]), lon=float(m[2]), radius_km=float(m[3]), used=used,
                    mb=round(sum(f.stat().st_size for f in files) / 1e6, 2),
                    layers=len(_BASE_LAYERS & names) + int(any(n.startswith("s2_") and not n.endswith(".none") for n in names)),
                    terrain=any(n.startswith("features") for n in names)))
            except OSError:
                continue
        return sorted(out, key=lambda p: -p["used"])

    def forget(self, area):
        """Delete one place. The name is validated so it can never point outside the cache folder."""
        if not _AREA_RE.match(area):
            return False
        p = os.path.join(self.root, area)
        existed = os.path.isdir(p)
        shutil.rmtree(p, ignore_errors=True)
        return existed

    # --- internals ---
    def _touch(self, area):
        """`.used` records when a place was last used; pruning deletes the least recently used first."""
        try:
            m = os.path.join(self.root, area, ".used")
            with open(m, "a"):
                pass
            os.utime(m)
        except OSError:
            pass

    @staticmethod
    def _remove(p):
        try:
            os.remove(p)
        except OSError:
            pass

    def _prune(self, keep):
        with self._lock:
            try:
                places, total = [], 0
                for e in os.scandir(self.root):
                    if not e.is_dir():
                        continue
                    size = sum(f.stat().st_size for f in os.scandir(e.path) if f.is_file())
                    try:
                        used = os.path.getmtime(os.path.join(e.path, ".used"))
                    except OSError:
                        used = e.stat().st_mtime
                    places.append((used, e.name, size))
                    total += size
                for _, name, size in sorted(places):
                    if total <= self.max_bytes:
                        break
                    if name != keep:
                        shutil.rmtree(os.path.join(self.root, name), ignore_errors=True)
                        total -= size
            except OSError:
                pass


class LRU:
    def __init__(self, size):
        self.size, self._d, self._lock = size, OrderedDict(), threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        return None

    def clear(self):
        with self._lock:
            self._d.clear()

    def put(self, key, value):
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.size:
                self._d.popitem(last=False)
