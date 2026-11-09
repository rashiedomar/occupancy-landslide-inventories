"""Figures for the toy experiment. Run after run_experiments.py."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).parent / "results"
FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)

SHOW = [("Naive (terrain + access)", "Standard practice\n(terrain + road features)"),
        ("Naive (terrain only)", "Standard, road\nfeatures dropped"),
        ("Occupancy, 3 inventories (OURS)", "Occupancy model\n(ours, 3 inventories)"),
        ("Oracle (trained on true landslides)", "Oracle\n(sees hidden truth)")]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def rank01(a):
    """Convert to percentile rank so maps from models with different scales are comparable."""
    r = np.empty_like(a, dtype=float)
    r[np.argsort(a)] = np.linspace(0, 1, len(a))
    return r


def fig_world_and_maps(scen):
    d = np.load(R / f"seed0_{scen}.npz")
    H, W = d["psi"].shape
    te = d["test_idx"].reshape(H, W)
    c0 = np.argmax(te[0])                    # first test column
    Wt = W - c0

    fig, ax = plt.subplots(2, 4, figsize=(12, 6.4), constrained_layout=True)
    a = ax[0, 0]
    a.imshow(d["psi"], cmap="magma", vmin=0, vmax=np.quantile(d["psi"], 0.99))
    ry, rx = np.nonzero(d["roads"]); a.scatter(rx, ry, s=0.3, c="cyan")
    a.axvline(c0 - 0.5, color="w", ls="--", lw=1)
    a.set_title("TRUE susceptibility (hidden)\ncyan = roads, right of dashed = test")
    a = ax[0, 1]
    zy, zx = np.nonzero(d["z"]); a.scatter(zx, zy, s=0.2, c="0.6", label="all true landslides")
    oy, ox = np.nonzero(d["y_any"]); a.scatter(ox, oy, s=1.2, c="crimson", label="recorded in an inventory")
    a.set_xlim(0, W); a.set_ylim(H, 0); a.set_aspect("equal")
    a.set_title(f"What researchers see\n{d['y_any'].sum()} of {d['z'].sum()} landslides recorded")
    a.legend(loc="lower left", fontsize=7, markerscale=6, framealpha=0.9)
    a = ax[0, 2]
    a.imshow(d["dist"], cmap="viridis"); a.set_title("Distance to road (km)")
    a = ax[0, 3]
    cov = d["cover"]; a.imshow(cov[2], cmap="Greys", vmin=0, vmax=1.5)
    a.set_title("Satellite event-mapping footprint\n(the only road-independent inventory)")
    for a in ax[0]: a.set_xticks([]); a.set_yticks([])

    truth = rank01(d["psi"][te])
    panels = [("TRUE (percentile)", truth)] + [(lab, rank01(d[k])) for k, lab in SHOW[:3]]
    for a, (lab, v) in zip(ax[1], panels):
        a.imshow(v.reshape(H, Wt), cmap="magma", vmin=0, vmax=1)
        ry, rx = np.nonzero(d["roads"][:, c0:]); a.scatter(rx, ry, s=0.3, c="cyan")
        if lab != "TRUE (percentile)":
            rho = np.corrcoef(v, truth)[0, 1]
            lab = lab.replace("\n", " ") + f"\nrank corr. with truth = {rho:.2f}"
        a.set_title(lab, fontsize=8.5); a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Toy world, seed 0 — scenario {scen.replace('_', ' ')}.  Bottom row: susceptibility maps on the held-out test area",
                 fontsize=10)
    fig.savefig(FIG / f"maps_{scen}.png", dpi=150)
    plt.close(fig)


def fig_summary():
    res = json.loads((R / "raw_results.json").read_text())
    names = [("Naive (terrain + access)", "Standard\n(+road feats)"),
             ("Naive (terrain only)", "Standard\n(roads dropped)"),
             ("Occupancy, 1 merged inventory", "Occupancy\n1 merged inv."),
             ("Occupancy, 2 road-biased inventories", "Occupancy\n2 road-biased"),
             ("Occupancy, 3 inventories (OURS)", "Occupancy\n3 inv. (ours)"),
             ("Oracle (trained on true landslides)", "Oracle")]
    mets = [("auc_true_landslides", "AUC vs TRUE landslides\n(0.5 = random)", 0.5),
            ("remote_near_ratio", "Remote-vs-near-road ranking\nrelative to truth (1 = unbiased)", 1.0)]
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5), constrained_layout=True, sharex=True)
    cols = ["#9aa0a6", "#9aa0a6", "#c6dbef", "#6baed6", "#08519c", "#2ca25f"]
    for j, (scen, runs) in enumerate(res.items()):
        for i, (met, lab, ref) in enumerate(mets):
            a = ax[i, j]
            vals = [np.array([r[k][met] for r in runs]) for k, _ in names]
            x = np.arange(len(names))
            a.bar(x, [v.mean() for v in vals], color=cols, width=0.7)
            for xi, v in zip(x, vals):
                a.scatter(np.full(len(v), xi) + np.linspace(-0.15, 0.15, len(v)), v, s=9, c="k", zorder=3)
            a.axhline(ref, color="k", lw=0.8, ls=":")
            a.set_ylabel(lab)
            if met == "remote_near_ratio":
                a.set_yscale("log")
            if i == 0:
                a.set_title(scen.replace("_", " "), fontsize=10)
            a.set_xticks(x); a.set_xticklabels([n for _, n in names], fontsize=7.5)
    fig.suptitle(f"Held-out test area, {len(runs)} random worlds (dots) — bar = mean", fontsize=10)
    fig.savefig(FIG / "summary_bars.png", dpi=150)
    plt.close(fig)


def fig_distance_profile(scen):
    d = np.load(R / f"seed0_{scen}.npz")
    te = d["test_idx"]
    dist, psi = d["dist"].ravel()[te], d["psi"].ravel()[te]
    bins = np.quantile(dist, np.linspace(0, 1, 9))
    idx = np.clip(np.digitize(dist, bins[1:-1]), 0, 7)
    mids = np.array([dist[idx == b].mean() for b in range(8)])
    fig, a = plt.subplots(figsize=(6, 4), constrained_layout=True)
    def prof(v):
        v = v / v.mean()
        return np.array([v[idx == b].mean() for b in range(8)])
    a.plot(mids, prof(psi), "k-o", lw=2.5, label="TRUE")
    for (k, lab), c in zip(SHOW, ["#9aa0a6", "#b07aa1", "#08519c", "#2ca25f"]):
        a.plot(mids, prof(d[k]), "-o", c=c, label=lab.replace("\n", " "), ms=4)
    a.set_xlabel("distance to nearest road (km), test area")
    a.set_ylabel("mean susceptibility\n(each model scaled to its own mean)")
    a.set_title("Does the map under-rate remote slopes?")
    a.legend(fontsize=7)
    fig.savefig(FIG / f"distance_profile_{scen}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for scen in ["A_reporting_bias_only", "B_roads_also_cause_landslides"]:
        fig_world_and_maps(scen)
        fig_distance_profile(scen)
    fig_summary()
    print("figures written to", FIG)
