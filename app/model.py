"""Waterlogging susceptibility x SCS-CN runoff (port of the notebook's algorithm cells)."""
import heapq
import math

import numpy as np
from scipy import ndimage as ndi

NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
FEATURES_VERSION = 1                                      # bump when features() changes, so cached terrain layers are rebuilt
ROAD_BURN_M = .3                                          # roads carry water: lowered by this much in the DEM
W0 = dict(twi=.25, depth=.25, low=.2, flat=.15, acc=.15)  # default layer weights
# ESA WorldCover class -> curve number per hydrologic soil group (A..D)
CN = {10: (30, 55, 70, 77), 20: (48, 67, 77, 83), 30: (39, 61, 74, 80), 40: (67, 78, 85, 89), 50: (89, 92, 94, 95),
      60: (77, 86, 91, 94), 80: (100,) * 4, 90: (98,) * 4, 95: (98,) * 4}


def pfill(z, eps=1e-5):
    """Priority-flood depression fill; also returns the processing order (low -> high)."""
    h, w = z.shape
    f = z.copy()
    seen = np.zeros(z.shape, bool)
    pq, order = [], []
    for i in range(h):
        for j in range(w):
            if i in (0, h - 1) or j in (0, w - 1):
                seen[i, j] = True
                heapq.heappush(pq, (float(z[i, j]), i, j))
    while pq:
        v, i, j = heapq.heappop(pq)
        order.append((i, j))
        for di, dj in NB:
            a, b = i + di, j + dj
            if 0 <= a < h and 0 <= b < w and not seen[a, b]:
                seen[a, b] = True
                f[a, b] = max(z[a, b], v + eps)
                heapq.heappush(pq, (float(f[a, b]), a, b))
    return f, order


def d8_acc(f, order):
    h, w = f.shape
    acc = np.ones(f.shape)
    for i, j in reversed(order):
        best, t = 0, None
        for di, dj in NB:
            a, b = i + di, j + dj
            if 0 <= a < h and 0 <= b < w:
                s = (f[i, j] - f[a, b]) / math.hypot(di, dj)
                if s > best:
                    best, t = s, (a, b)
        if t:
            acc[t] += acc[i, j]
    return acc


def nrm(x, m=1e-6):
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo) / (max(hi, lo + m) - lo), 0, 1)


def features(dem, bld, roads, T, lat):
    """Expensive terrain layers, computed once per area."""
    dy = abs(T.e) * 111320
    dx = T.a * 111320 * math.cos(math.radians(lat))
    z = dem + 3 * bld - ROAD_BURN_M * roads              # buildings block water, roads carry it
    f, order = pfill(z)
    acc = d8_acc(f, order)
    gy, gx = np.gradient(ndi.gaussian_filter(dem, 1), dy, dx)
    tanb = np.maximum(np.hypot(gx, gy), 1e-3)
    return dict(twi=nrm(np.log(acc * dx / tanb)), depth=nrm(f - z, 1.0), low=nrm(ndi.uniform_filter(z, 21) - z, 1.0),
                flat=1 - nrm(tanb), acc=nrm(np.log1p(acc)))


class Model:
    def __init__(self, d, lat, lon, radius_km, duration_h, drain_mm_h, road_drain_frac, F=None):
        self.d, self.lat, self.lon, self.R = d, lat, lon, radius_km
        self.duration_h, self.drain = duration_h, drain_mm_h
        T = d.T
        self.F = F if F is not None else features(d.dem, d.bld, d.roads, T, lat)
        # drain influence 0-1: mapped drains plus major roads acting as partial drains
        self.cover = np.clip(5 * ndi.uniform_filter(np.maximum(d.drn, road_drain_frac * d.major).astype(float), 5), 0, 1)
        self.uwd = ndi.binary_dilation(d.uw)
        self.CN0 = np.full(d.dem.shape, 80.)
        for k, v in CN.items():
            self.CN0[d.lc == k] = v[d.hsg]
        ii, jj = np.indices(d.dem.shape)
        km = np.hypot((T.c + (jj + .5) * T.a - lon) * 111.32 * math.cos(math.radians(lat)),
                      (T.f + (ii + .5) * T.e - lat) * 111.32)
        self.circle = km <= radius_km
        self.land = self.circle & (d.lc != 80)
        self.bounds = (T.c, T.c + d.dem.shape[1] * T.a, T.f + d.dem.shape[0] * T.e, T.f)  # west, east, south, north
        area, self.ckm = (2 * radius_km) ** 2, abs(T.e) * 111.32
        self.drain_km2 = float(d.drn.sum() * self.ckm / area)
        self.major_km2 = float(d.major.sum() * self.ckm / area)

    def susceptibility(self, w=W0):
        s = ndi.gaussian_filter(sum(w[k] * self.F[k] for k in w) / sum(w.values()), 1)
        return np.where(self.uwd, np.maximum(s, .9), s)   # underpasses are forced hotspots

    def runoff(self, P_mm, cn_shift=0):                   # SCS-CN
        S = 25400 / np.clip(self.CN0 + cn_shift, 30, 100) - 254
        return np.where(P_mm > .2 * S, (P_mm - .2 * S) ** 2 / (P_mm + .8 * S), 0)

    def heat(self, P_mm, w=W0, cn_shift=0, cap=None, drains=True):
        q = self.runoff(P_mm, cn_shift)
        if drains:
            q = np.maximum(q - (self.drain if cap is None else cap) * self.duration_h * self.cover, 0)
        return self.susceptibility(w) * q

    def scenario_maps(self, rain_cm, w=W0):
        out = []
        for p in rain_cm:
            h = self.heat(p * 10, w)
            h[~self.land] = np.nan
            out.append(h)
        return out

    def robust(self, P_mm, w=W0, n=30, seed=0):
        """Share of n perturbed runs (weights, CN, drain capacity, drains on/off) where a cell is in the top 10%."""
        rng = np.random.default_rng(seed)
        hot = np.zeros(self.d.dem.shape)
        for _ in range(n):
            ww = dict(zip(w, rng.dirichlet(np.array(list(w.values())) * 30 + 1e-3)))
            h = self.heat(P_mm, ww, rng.uniform(-5, 5), self.drain * rng.choice([0, .5, 1, 2]), rng.random() > .2)
            hot += (h >= np.percentile(h[self.land], 90)) & (h > 0)
        return np.where(self.land, hot / n, np.nan)

    def hotspots(self, robust, heat, thr=.8, top=10, min_cells=2):
        """Rank clusters of robust-hotspot cells (share of runs >= thr) by heat summed over the cluster.

        Buildings and roads are counted within one cell (~30 m) of a cluster, because buildings are raised
        in the terrain model and so are rarely inside a low spot themselves.
        Returns (items, summary, labels); labels is the cluster raster used for the GeoJSON export.
        """
        T, d = self.d.T, self.d
        dy = abs(T.e) * 111320
        dx = T.a * 111320 * math.cos(math.radians(self.lat))
        cell, clen = dx * dy, (dx + dy) / 2
        mask = np.nan_to_num(robust) >= thr
        lab, n = ndi.label(mask, structure=np.ones((3, 3)))
        h = np.nan_to_num(heat)
        near = lambda m: ndi.binary_dilation(m, structure=np.ones((3, 3))) & self.land   # one cell in every direction, diagonals included
        bt, rt = float(d.bld[self.land].sum()), float(d.roads[self.land].sum())
        reg = near(mask)
        summary = dict(
            hotspot_ha=round(float(mask.sum() * cell / 1e4), 2),
            bld_m2=int(d.bld[reg].sum() * cell), bld_pct=round(100 * float(d.bld[reg].sum()) / bt, 1) if bt else None,
            road_km=round(float(d.roads[reg].sum() * clen / 1e3), 2), road_pct=round(100 * float(d.roads[reg].sum()) / rt, 1) if rt else None)
        if n == 0:
            return [], summary, lab
        idx = np.arange(1, n + 1)
        cnt = ndi.sum(mask, lab, idx)
        sev = ndi.sum(h, lab, idx)
        mean = ndi.mean(h, lab, idx)
        com = ndi.center_of_mass(mask, lab, idx)
        items = []
        for i in [i for i in np.argsort(-sev) if cnt[i] >= min_cells][:top]:
            r, c = com[i]
            region = near(lab == idx[i])
            items.append(dict(
                rank=len(items) + 1, id=int(idx[i]), lat=float(T.f + (r + .5) * T.e), lon=float(T.c + (c + .5) * T.a),
                area_m2=int(cnt[i] * cell), mean_mm=round(float(mean[i]), 1),
                bld_m2=int(d.bld[region].sum() * cell), road_m=int(d.roads[region].sum() * clen)))
        return items, summary, lab

    def evaluate(self, points):
        """Score the model against reported waterlogging points [{lat, lon, label}]. Returns report lines + calibrated weights."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold

        T, d, lines = self.d.T, self.d, []
        la = np.array([p["lat"] for p in points], float)
        lo = np.array([p["lon"] for p in points], float)
        y = np.array([p.get("label", 1) for p in points], int)
        r = np.floor((la - T.f) / T.e).astype(int)
        c = np.floor((lo - T.c) / T.a).astype(int)
        ok = (r >= 0) & (r < d.dem.shape[0]) & (c >= 0) & (c < d.dem.shape[1])
        r, c, y = r[ok], c[ok], y[ok]
        ok = self.land[r, c]
        r, c, y = r[ok], c[ok], y[ok]
        npos = int((y == 1).sum())
        if npos == 0:
            return dict(lines=["No reported points fall on land inside the analysed area."], W_CAL=None, pts=[])
        if (y == 0).sum() == 0:                          # no negatives given: random land cells away from positives
            pm = np.zeros(d.dem.shape, bool)
            pm[r, c] = True
            rr, cc = np.nonzero(self.land & (ndi.distance_transform_edt(~pm) > 10))
            k = np.random.default_rng(0).choice(len(rr), min(3 * npos, len(rr)), replace=False)
            r, c, y = np.r_[r, rr[k]], np.r_[c, cc[k]], np.r_[y, np.zeros(len(k), int)]
            lines.append("No negatives supplied, so random land cells were used (unverified): AUC may be optimistic or pessimistic.")
        cm = abs(T.e) * 111320
        edt = lambda m: ndi.distance_transform_edt(~m) * cm if m.any() else np.full(m.shape, 1e3)
        layers = [self.F[k] for k in W0] + [np.log1p(edt(d.drn)), np.log1p(edt(d.major)), np.log1p(edt(d.uw)), self.CN0]
        X = np.stack([L[r, c] for L in layers], 1)
        g = (r // 16) * 1000 + c // 16                   # ~500 m spatial blocks
        s0 = self.susceptibility()
        pct = np.mean(s0[self.land][:, None] < s0[r[y == 1], c[y == 1]][None])
        lines.append(f"{npos} flood points, {len(y) - npos} negatives, {len(set(g))} spatial blocks")
        if len(set(y)) < 2:
            return dict(lines=lines, W_CAL=None, pts=[])
        lines.append(f"Algorithm (fixed weights): AUC {roc_auc_score(y, s0[r, c]):.3f}; flood points sit at mean percentile {pct:.2f}")
        models = {"Logistic (re-weighted layers)": (slice(0, 5), LogisticRegression(class_weight="balanced")),
                  "Random forest (all features)": (slice(None), RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced_subsample", random_state=0))}
        if len(set(g)) >= 3:
            for name, (cols, m) in models.items():
                oof = np.full(len(y), np.nan)
                for tr, te in GroupKFold(min(5, len(set(g)))).split(X, y, g):
                    if len(set(y[tr])) > 1:
                        oof[te] = m.fit(X[tr][:, cols], y[tr]).predict_proba(X[te][:, cols])[:, 1]
                k = ~np.isnan(oof)
                if k.sum() and len(set(y[k])) > 1:
                    lines.append(f"{name}: spatial-CV AUC {roc_auc_score(y[k], oof[k]):.3f}")
        else:
            lines.append("Too few spatial blocks for spatial cross-validation; add more points.")
        co = np.clip(LogisticRegression(class_weight="balanced").fit(X[:, :5], y).coef_[0], 0, None)
        w_cal = None
        if co.sum() > 0:
            w_cal = dict(zip(W0, (co / co.sum()).tolist()))
            lines.append("Calibrated weights: " + ", ".join(f"{k} {v:.2f}" for k, v in w_cal.items()))
        pos = y == 1
        pts = [[float(T.f + (rr + .5) * T.e), float(T.c + (cc + .5) * T.a)] for rr, cc in zip(r[pos], c[pos])]
        return dict(lines=lines, W_CAL=w_cal, pts=pts)
