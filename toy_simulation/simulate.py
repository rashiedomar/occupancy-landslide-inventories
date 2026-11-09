"""
Synthetic landscape + landslides + biased landslide inventories.

The point of this file: create a world where we KNOW the true landslide
susceptibility (psi) and the true landslides (z), then generate the kind of
incomplete, road-biased inventories that real researchers actually get.

Grid cell = 100 m, default grid 200 x 200 cells = 20 km x 20 km.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


def _smooth_field(rng, shape, sigma):
    f = ndimage.gaussian_filter(rng.standard_normal(shape), sigma=sigma, mode="reflect")
    return (f - f.mean()) / (f.std() + 1e-9)


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _least_cost_path(cost, start, end):
    """Least-cost path on a 4-connected grid (Dijkstra). Returns list of flat indices."""
    H, W = cost.shape
    idx = np.arange(H * W).reshape(H, W)
    rows, cols, w = [], [], []
    # right & down neighbours (graph is undirected)
    a, b = idx[:, :-1].ravel(), idx[:, 1:].ravel()
    c = 0.5 * (cost[:, :-1].ravel() + cost[:, 1:].ravel())
    rows += [a, b]; cols += [b, a]; w += [c, c]
    a, b = idx[:-1, :].ravel(), idx[1:, :].ravel()
    c = 0.5 * (cost[:-1, :].ravel() + cost[1:, :].ravel())
    rows += [a, b]; cols += [b, a]; w += [c, c]
    G = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(H * W, H * W)).tocsr()
    s = start[0] * W + start[1]
    e = end[0] * W + end[1]
    _, pred = dijkstra(G, indices=s, return_predecessors=True)
    path, cur = [], e
    while cur != s and cur >= 0:
        path.append(cur)
        cur = pred[cur]
    path.append(s)
    return path


@dataclass
class World:
    shape: tuple
    terrain: dict          # name -> 2D array (standardised)
    access: dict           # name -> 2D array (standardised, except dist_km kept raw too)
    dist_road_km: np.ndarray
    roads: np.ndarray      # bool 2D
    towns: list
    psi: np.ndarray        # true susceptibility P(z=1)
    z: np.ndarray          # true landslide occurrence (0/1)
    p: np.ndarray          # true detection prob, shape (K, H, W)
    cover: np.ndarray      # inventory coverage mask, shape (K, H, W)
    y: np.ndarray          # observed inventories, shape (K, H, W)
    inventory_names: list
    road_causes_landslides: bool
    meta: dict = field(default_factory=dict)


def make_world(seed=0, H=200, W=200, n_towns=6, road_effect=False, target_rate=0.08):
    rng = np.random.default_rng(seed)
    shape = (H, W)

    # ---------- terrain ----------
    elev = 0.7 * _smooth_field(rng, shape, 25) + 0.3 * _smooth_field(rng, shape, 7)
    elev = (elev - elev.min()) / (elev.max() - elev.min()) * 1800 + 200      # metres
    gy, gx = np.gradient(elev, 100.0)                                         # 100 m cells
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    curv = ndimage.laplace(ndimage.gaussian_filter(elev, 2))                  # >0 convex, <0 hollow
    weak_rock = _smooth_field(rng, shape, 15)
    rain = 0.6 * _smooth_field(rng, shape, 35) + 0.4 * _z(elev)               # orographic rain
    ndvi = -0.5 * _z(elev) + 0.5 * _smooth_field(rng, shape, 10)

    terrain = {
        "slope": _z(slope),
        "curvature": _z(curv),
        "weak_rock": weak_rock,
        "rain": _z(rain),
        "ndvi": _z(ndvi),
        "elev": _z(elev),
    }

    # ---------- towns & roads (roads follow gentle valleys -> confounding) ----------
    suit = -terrain["slope"] - 0.8 * terrain["elev"]
    margin = 10
    cand = np.argwhere(suit > np.quantile(suit, 0.85))
    cand = cand[(cand[:, 0] > margin) & (cand[:, 0] < H - margin) &
                (cand[:, 1] > margin) & (cand[:, 1] < W - margin)]
    towns = []
    while len(towns) < n_towns:
        t = cand[rng.integers(len(cand))]
        if all(np.hypot(*(t - np.array(o))) > 35 for o in towns):
            towns.append(tuple(t))
    cost = 1.0 + np.exp(terrain["slope"]) * 3.0                                # steep = expensive
    roads = np.zeros(shape, bool)
    order = list(range(n_towns))
    pairs = [(order[i], order[i + 1]) for i in range(n_towns - 1)] + [(0, n_towns - 1)]
    for a, b in pairs:
        for k in _least_cost_path(cost, towns[a], towns[b]):
            roads.flat[k] = True
    roads = ndimage.binary_dilation(roads, iterations=0) if False else roads
    dist_road_km = ndimage.distance_transform_edt(~roads) * 0.1

    pop = np.zeros(shape)
    yy, xx = np.mgrid[0:H, 0:W]
    for (ty, tx) in towns:
        pop += np.exp(-((yy - ty) ** 2 + (xx - tx) ** 2) / (2 * 12.0 ** 2))
    cloud = 0.6 * _smooth_field(rng, shape, 30) + 0.4 * terrain["elev"]        # mountains cloudier

    access = {
        "dist_road": _z(np.log1p(dist_road_km)),
        "pop": _z(np.log1p(pop * 50)),
        "cloud": _z(cloud),
    }

    # ---------- TRUE susceptibility ----------
    s, r = terrain["slope"], terrain["rain"]
    logit = (1.6 * s - 0.35 * s ** 2 + 0.8 * terrain["weak_rock"] + 0.7 * r
             - 0.6 * terrain["ndvi"] - 0.5 * terrain["curvature"] + 0.5 * s * r)
    if road_effect:
        # road cuts genuinely destabilise slopes within ~150 m, more so on steep ground
        logit = logit + 1.5 * np.exp(-dist_road_km / 0.15) * _sigmoid(2 * s)
    # pick intercept so mean psi ~ target_rate
    lo, hi = -15, 5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _sigmoid(logit + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    psi = _sigmoid(logit + mid)
    z = (rng.random(shape) < psi).astype(np.int8)

    # ---------- biased inventories (the "observers") ----------
    d = dist_road_km
    names = ["news_catalog", "agency_field_survey", "satellite_event_map"]
    p = np.stack([
        # 1. news / media reports: mostly near roads & people
        _sigmoid(-0.3 - 2.0 * d + 0.9 * access["pop"]),
        # 2. agency field survey along road corridors
        _sigmoid(1.5 - 3.0 * d),
        # 3. post-event satellite mapping: road-independent, but blocked by cloud
        _sigmoid(1.6 - 1.8 * access["cloud"]),
    ])
    cover = np.ones_like(p, dtype=bool)
    # satellite mapping only done for a few event footprints (rectangles)
    sat = np.zeros(shape, bool)
    for _ in range(3):
        h, w = rng.integers(40, 80), rng.integers(40, 80)
        y0, x0 = rng.integers(0, H - h), rng.integers(0, W - w)
        sat[y0:y0 + h, x0:x0 + w] = True
    cover[2] = sat
    y = ((rng.random(p.shape) < p) & cover & (z[None] == 1)).astype(np.int8)

    return World(shape, terrain, access, dist_road_km, roads, towns, psi, z, p, cover, y,
                 names, road_effect,
                 meta=dict(seed=seed, true_rate=float(z.mean()),
                           detected_any=float(y.max(0).mean()),
                           frac_landslides_recorded=float(y.max(0)[z == 1].mean())))


if __name__ == "__main__":
    w = make_world(0)
    print(w.meta)
    print("road cells:", w.roads.sum(), " towns:", w.towns)
    for k, n in enumerate(w.inventory_names):
        print(f"{n:22s} records={w.y[k].sum():5d}  mean p (where z=1 & covered)="
              f"{w.p[k][(w.z == 1) & w.cover[k]].mean():.2f}  coverage={w.cover[k].mean():.2f}")
