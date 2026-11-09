"""Figures for the Iburi real-terrain experiment. Run after run_iburi.py."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_iburi import OUT, block_split, load_grid, simulate_inventories

FIG = Path(__file__).parent / "figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def rank01(a):
    r = np.empty(len(a)); r[np.argsort(a)] = np.linspace(0, 1, len(a)); return r


def fig_maps(scen="A"):
    d, m, terrain, dist = load_grid()
    z = d["z"]
    rng = np.random.default_rng(0)                     # same order as run_one(seed=0)
    y, _, cover, _ = simulate_inventories(z, m, terrain, dist, d["built_density"], rng)
    split = block_split(m, rng)
    P = np.load(OUT / f"seed0_{scen}.npz")
    te_grid = np.zeros_like(m); te_grid[m] = P["test"]
    H, W = m.shape
    roads = dist["public"] < 0.06

    fig, ax = plt.subplots(1, 6, figsize=(16, 5.2), constrained_layout=True)
    def base(a, title):
        bg = np.where(m, 0.93, 1.0)
        a.imshow(bg, cmap="gray", vmin=0, vmax=1)
        a.set_title(title, fontsize=8.5); a.set_xticks([]); a.set_yticks([])
    a = ax[0]; base(a, f"REAL landslides (full inventory)\n{int(z[m].sum())} cells; gray = public roads")
    a.imshow(np.ma.masked_where(~roads, roads), cmap="Greys", vmin=0, vmax=1, alpha=0.5)
    a.imshow(np.ma.masked_where(z == 0, z), cmap="autumn", vmin=0, vmax=1)
    a = ax[1]; ya = y.max(0)
    base(a, f"Simulated biased records (seed 0)\n{int(ya.sum())} cells = {ya[m & (z == 1)].mean():.0%} of real")
    a.imshow(np.ma.masked_where(~roads, roads), cmap="Greys", vmin=0, vmax=1, alpha=0.5)
    a.imshow(np.ma.masked_where(ya == 0, ya), cmap="autumn", vmin=0, vmax=1)
    a.contour(cover[2], levels=[0.5], colors="royalblue", linewidths=0.8)
    a.text(2, H - 4, "blue = satellite footprints", fontsize=7, color="royalblue")

    show = [("TRUE landslides in test blocks", None),
            ("Naive (terrain + access)", "Standard practice"),
            ("Naive (terrain only)", "Standard, roads dropped"),
            ("Occupancy, 3 inventories (OURS)", "Occupancy model (ours)")]
    zt = z[te_grid].astype(float)
    for a, (k, lab) in zip(ax[2:], show):
        img = np.full((H, W), np.nan)
        if k.startswith("TRUE"):
            img[te_grid] = zt; title = "TRUE landslides\n(test blocks only)"; cmap = "magma"
        else:
            img[te_grid] = rank01(P[k]); cmap = "magma"
            from sklearn.metrics import roc_auc_score
            title = f"{lab}\nAUC vs real landslides = {roc_auc_score(zt, P[k]):.2f}"
        base(a, title)
        a.imshow(img, cmap=cmap, vmin=0, vmax=1)
    fig.suptitle(f"2018 Iburi earthquake area, 120 m cells — scenario {scen}. "
                 "Maps 3–6: predicted risk percentile on held-out test blocks", fontsize=10)
    fig.savefig(FIG / f"iburi_maps_{scen}.png", dpi=150)
    plt.close(fig)


def fig_summary():
    res = json.loads((OUT / "raw_results.json").read_text())
    names = [("Naive (terrain + access)", "Standard\n(+road feats)"),
             ("Naive (terrain only)", "Standard\n(roads dropped)"),
             ("Occupancy, 1 merged inventory", "Occupancy\n1 merged inv."),
             ("Occupancy, 2 road-biased inventories", "Occupancy\n2 road-biased"),
             ("Occupancy, 3 inventories (OURS)", "Occupancy\n3 inv. (ours)"),
             ("Oracle (trained on full real inventory)", "Oracle")]
    mets = [("auc_true_landslides", "AUC vs REAL landslides", None),
            ("remote_near_ratio", "Remote-vs-near-road ranking\nrelative to truth (1 = unbiased)", 1.0)]
    titles = {"A": "A: susceptibility sees terrain only", "B": "B: susceptibility also sees roads"}
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
            if ref is not None:
                a.axhline(ref, color="k", lw=0.8, ls=":")
            if met == "auc_true_landslides":
                a.set_ylim(0.5, 0.9)
            else:
                a.set_yscale("log")
            a.set_ylabel(lab)
            if i == 0:
                a.set_title(titles[scen], fontsize=10)
            a.set_xticks(x); a.set_xticklabels([n for _, n in names], fontsize=7.5)
    fig.suptitle(f"Real Iburi terrain & landslides, simulated biased records — {len(runs)} random "
                 "draws (dots), bar = mean", fontsize=10)
    fig.savefig(FIG / "iburi_summary_bars.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for s in ["A", "B"]:
        if (OUT / f"seed0_{s}.npz").exists():
            fig_maps(s)
    fig_summary()
    print("figures in", FIG)
