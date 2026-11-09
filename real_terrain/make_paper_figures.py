"""Figures of the revised manuscript, written to paper/latex/figures/. Run after rev_summary.py."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "toy_simulation"))
sys.path.insert(0, str(HERE))
from models import MonotoneDetection  # noqa: E402
from run_iburi import block_split, load_grid, simulate_inventories  # noqa: E402
from run_iburi_v3 import ACCESS, SIGNS  # noqa: E402
from summarize_claims import collect  # noqa: E402

FIG = HERE.parent / "paper/latex/figures"
R = HERE / "results_rev"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlesize": 9, "axes.titleweight": "bold", "legend.frameon": False, "savefig.dpi": 300})
MODELS = {"standard": ("Standard practice", "#eb6834", "s"), "mlp": ("Neutralised MLP", "#1baf7a", "^"),
          "glm": ("Neutralised GLM", "#eda100", "v"), "gam": ("Neutralised GAM", "#e87ba4", "P"),
          "ours": ("Occupancy model", "#2a78d6", "o"), "oracle": ("Oracle", INK, "D")}


def rank01(a):
    r = np.empty(len(a)); r[np.argsort(a)] = np.linspace(0, 1, len(a)); return r


def fig_maps():
    d, m, terrain, dist = load_grid(HERE.parent / "data/processed/iburi_grid.npz")
    z = d["z"]
    y, _, cover, _ = simulate_inventories(z, m, terrain, dist, d["built_density"], np.random.default_rng(0))
    P = np.load(HERE / "results_v3/seed0_A.npz")
    te = np.zeros_like(m); te[m] = P["test"]
    roads = dist["public"] < 0.06
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 6.2), constrained_layout=True)
    ax = ax.ravel()

    def base(a, title):
        a.imshow(np.where(m, 0.93, 1.0), cmap="gray", vmin=0, vmax=1)
        a.set_title(title, fontsize=8, fontweight="normal", color=INK); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)

    def over(a, mask, color):
        a.imshow(np.ma.masked_where(~mask, np.ones_like(mask, float)), cmap=matplotlib.colors.ListedColormap([color]), vmin=0, vmax=1)

    ya = y.max(0).astype(bool)
    base(ax[0], f"(a) Reference inventory\n{int(z[m].sum()):,} cells; roads in grey")
    over(ax[0], roads, "#a9a8a2"); over(ax[0], z.astype(bool) & m, "#d4361f")
    base(ax[1], f"(b) Simulated records\n{int(ya.sum()):,} cells, {ya[m & (z == 1)].mean():.0%} of the reference")
    over(ax[1], roads, "#a9a8a2"); over(ax[1], ya, "#d4361f")
    ax[1].contour(cover[2], levels=[0.5], colors="#2a78d6", linewidths=1.0)
    base(ax[2], "(c) Reference in test blocks")
    over(ax[2], te, "#c9c8c2"); over(ax[2], te & (z == 1), "#d4361f")
    show = [("Standard (terrain + access)", "(d) Standard practice"), ("Standard (terrain only)", "(e) Standard, roads dropped"),
            ("Occupancy + monotone + anchor (FULL)", "(f) Occupancy model")]
    zt = z[te].astype(float)
    for a, (k, lab) in zip(ax[3:], show):
        img = np.full(m.shape, np.nan); img[te] = rank01(P[k])
        base(a, f"{lab}\nAUC {roc_auc_score(zt, P[k]):.2f}")
        im = a.imshow(img, cmap="viridis", vmin=0, vmax=1)
        print(lab, round(roc_auc_score(zt, P[k]), 3))
    fig.colorbar(im, ax=ax[3:], shrink=0.75, label="Predicted risk percentile (test blocks)", ticks=[0, 0.5, 1])
    fig.savefig(FIG / "fig_iburi_maps.png"); plt.close(fig)


def fig_headline():
    sets = collect()
    hb = {}
    for f in sorted((HERE / "results_haiti").rglob("raw_results_A*.json")):
        hb.update(json.loads(f.read_text()))
    hb = [v for _, v in sorted(hb.items(), key=lambda kv: int(kv[0]))]
    H = {"standard": "Standard (terrain + access)", "mlp": "Steger-MLP (access = mean)", "glm": "Steger-GLM (access neutralised)",
         "ours": "OURS full", "oracle": "Oracle"}
    data = {}
    for name, key, gk in [("Iburi A", "Iburi-A", "iburi_A"), ("Iburi C", "Iburi-C", "iburi_C"), ("Gorkha A", "Gorkha-A", "gorkha_A"), ("Gorkha C", "Gorkha-C", "gorkha_C")]:
        s = sets[key]
        data[name] = {"standard": s["standard"], "mlp": s["steger"], "glm": s["steger_glm"], "ours": s["ours"], "oracle": s["oracle"],
                      "gam": json.loads((R / f"gam_{gk}.json").read_text())}
    data["Haiti A"] = {k: [r[v] for r in hb] for k, v in H.items()}
    data["Haiti A"]["gam"] = json.loads((R / "gam_haiti_A.json").read_text())
    mets = [("auc_true_landslides", "AUC, all test cells"), ("auc_remote_quarter", "AUC, most remote quarter"),
            ("abs_log_ratio", "Road bias |log R| (0 = unbiased)")]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 4.3), constrained_layout=True, sharey=True)
    names = list(data)
    for a, (met, lab) in zip(ax, mets):
        for i, n in enumerate(names):
            for j, (k, (label, c, mk)) in enumerate(MODELS.items()):
                v = np.array([r[met] for r in data[n][k]])
                y = i + (j - 2.5) * 0.135
                a.scatter(v, np.full(len(v), y), s=7, color=c, alpha=0.35, linewidths=0, zorder=2)
                a.scatter(v.mean(), y, s=30, color=c, marker=mk, edgecolors="white", linewidths=0.5, zorder=3,
                          label=label if i == 0 else None)
        a.set_yticks(range(len(names))); a.set_yticklabels(names); a.invert_yaxis() if a is ax[0] else None
        a.set_xlabel(lab); a.grid(axis="x", color=GRID, lw=0.6); a.set_axisbelow(True)
        for i in range(len(names) - 1):
            a.axhline(i + 0.5, color=GRID, lw=0.6)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=6, fontsize=7.5, handletextpad=0.2, columnspacing=0.9)
    fig.savefig(FIG / "fig_headline.png"); plt.close(fig)


def fig_detection():
    P = np.load(HERE / "results_v3/seed0_A.npz")
    states = torch.load(HERE / "results_v3/seed0_A_detection_heads.pt")
    heads = []
    for st in states:
        h = MonotoneDetection(SIGNS, 3); h.load_state_dict(st); heads.append(h)
    names = ["News catalogue", "Agency survey", "Satellite map"]; cols = ["#eb6834", "#2a78d6", "#1baf7a"]; ls = ["-", "--", ":"]
    labels = {"log_d_major": "Distance to major road", "log_d_public": "Distance to public road", "log_d_track": "Distance to track",
              "built": "Built-up density", "cloud": "Simulated cloud cover"}
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.4), constrained_layout=True)
    ax = ax.ravel(); g = torch.linspace(-2.5, 2.5, 100)
    for j, f in enumerate(ACCESS):
        a = ax[j]
        X = torch.zeros(100, len(ACCESS)); X[:, j] = g
        with torch.no_grad():
            comp = np.mean([h.component(X)[:, :, j].numpy() for h in heads], 0)
        for k in range(3):
            a.plot(g, comp[:, k] - comp[50, k], c=cols[k], ls=ls[k], lw=1.6, label=names[k])
        a.axhline(0, c=MUTED, lw=0.5); a.set_title(labels[f], fontweight="normal"); a.grid(color=GRID, lw=0.5)
        if j in (3, 4):
            a.set_xlabel("Standardised value")
        if j in (0, 3):
            a.set_ylabel("Change in detection log-odds")
    ax[0].legend(fontsize=7)
    a = ax[5]; ph, pt, cov = P["p_hat"], P["p_true"], P["cover"].astype(bool); rng = np.random.default_rng(0)
    for k, mk in zip(range(3), ["o", "s", "^"]):
        idx = rng.choice(np.flatnonzero(cov[:, k]), 800, replace=False)
        a.scatter(pt[idx, k], ph[idx, k], s=3, c=cols[k], marker=mk, alpha=0.45, linewidths=0)
    a.plot([0, 1], [0, 1], c=INK, lw=0.7, ls="--"); a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.set_xlabel("True detection probability"); a.set_ylabel("Learned"); a.set_title("Learned vs true", fontweight="normal")
    fig.savefig(FIG / "fig_detection.png"); plt.close(fig)


OFFSETS = {"truth_gap": {"gorkha_A": (-0.0018, 0.0025, "right"), "iburi_W3": (0.0018, 0.0015, "left"), "iburi_W1a": (-0.0018, -0.001, "right"),
                         "haiti_A": (0.0022, -0.0015, "left"), "gorkha_L": (-0.0018, 0.0025, "right"), "gorkha_C": (0.0022, -0.001, "left"),
                         "gorkha_H1": (0.0018, -0.006, "left")},
           "records_gap": {"iburi_W3": (-0.0018, 0.0025, "right"), "iburi_W1a": (0.0018, 0.001, "left"), "haiti_A": (-0.0022, -0.0015, "right"),
                           "gorkha_L": (-0.0018, -0.001, "right"), "iburi_L": (0.0018, -0.001, "left"), "gorkha_C": (0.0022, -0.001, "left"),
                           "gorkha_H1": (0.0018, -0.006, "left"), "iburi_W1": (-0.0018, 0.0025, "right")}}


def fig_rule():
    rows = json.loads((R / "summary.json").read_text())["rule"]
    lab = {"iburi_L": "I-add", "iburi_N": "I-rand", "iburi_W1": "I-real", "iburi_W2": "I-real+roads", "iburi_W3": "I-no-epi", "iburi_W1a": "I-real-add",
           "gorkha_H1": "G-hidden", "gorkha_L": "G-add", "gorkha_N": "G-rand", "iburi_A": "Iburi A", "iburi_C": "Iburi C", "gorkha_A": "Gorkha A",
           "gorkha_C": "Gorkha C", "haiti_A": "Haiti A"}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6), constrained_layout=True, sharey=True)
    for a, key, title in [(ax[0], "truth_gap", "(a) Gap on the reference inventory"), (ax[1], "records_gap", "(b) Gap on the biased records only")]:
        a.axvspan(0.015, 0.03, color="#efeeea", zorder=0)
        a.axhline(0, c=MUTED, lw=0.7); a.grid(color=GRID, lw=0.5); a.set_axisbelow(True)
        for r in rows:
            real = "_" not in r["world"] or r["world"] in ("iburi_A", "iburi_C", "gorkha_A", "gorkha_C", "haiti_A")
            oos = r["world"] == "haiti_A"
            a.scatter(r[key], r["diff3"], s=42 if real else 24, marker="o" if real else "s", linewidths=0.8,
                      color="#2a78d6" if real else "white", edgecolors="#2a78d6", zorder=3)
            dx, dy, ha = OFFSETS[key].get(r["world"], (0.0018, 0.0025, "left"))
            a.text(r[key] + dx, r["diff3"] + dy, lab[r["world"]], fontsize=6.5, color=INK, ha=ha, fontweight="bold" if oos else "normal")
        a.set_xlim(-0.024, 0.072); a.set_ylim(-0.066, 0.078)
        a.set_xlabel("Interaction gap (AUC, GBM minus GAM)"); a.set_title(title, fontweight="normal")
    ax[0].set_ylabel("AUC, occupancy minus best neutralisation")
    fig.savefig(FIG / "fig_rule.png"); plt.close(fig)


def fig_calibration():
    sites = [("iburi_A", "Iburi A"), ("iburi_C", "Iburi C"), ("gorkha_A", "Gorkha A"), ("gorkha_C", "Gorkha C"), ("haiti_A", "Haiti A")]
    show = {"standard": ("standard",) + MODELS["standard"], "glm": ("steger_glm",) + MODELS["glm"], "glm_a": ("steger_glm_anchor_scaled", "Neutralised GLM, anchor-scaled", "#a26a00", "X"),
            "ours": ("ours",) + MODELS["ours"], "oracle": ("oracle",) + MODELS["oracle"]}
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 5.0), constrained_layout=True)
    ax = ax.ravel(); allr = []
    for a, (s, name) in zip(ax, sites):
        rs = [json.loads(f.read_text()) for f in sorted(R.glob(f"calib_{s}_*.json")) if not f.name.startswith("._")]
        allr += rs
        for k, (key, label, c, mk) in show.items():
            bp = np.mean([r[key]["bin_pred"] for r in rs], 0); bt = np.mean([r[key]["bin_true"] for r in rs], 0)
            a.plot(bp, bt, c=c, marker=mk, ms=3.2, lw=1.0, label=label)
        lim = 0.75
        a.plot([0, lim], [0, lim], c=MUTED, lw=0.6, ls="--"); a.set_xlim(0, lim); a.set_ylim(0, lim)
        a.set_title(name, fontweight="normal"); a.grid(color=GRID, lw=0.5)
    for i in (0, 3):
        ax[i].set_ylabel("Observed landslide rate")
    for i in (2, 3, 4):
        ax[i].set_xlabel("Mean predicted probability")
    a = ax[5]
    ks = ["standard", "steger_mlp", "steger_glm", "standard_anchor_scaled", "steger_mlp_anchor_scaled", "steger_glm_anchor_scaled", "ours", "oracle"]
    nm = ["Standard", "Neut. MLP", "Neut. GLM", "Standard, scaled", "Neut. MLP, scaled", "Neut. GLM, scaled", "Occupancy", "Oracle"]
    for i, (k, n) in enumerate(zip(ks, nm)):
        v = [r[k]["rate_ratio"] for r in allr if k in r]
        c = "#2a78d6" if k == "ours" else INK if k == "oracle" else "#8a8984"
        a.scatter(v, np.full(len(v), i), s=8, color=c, alpha=0.5, linewidths=0)
        a.scatter(np.median(v), i, s=30, color=c, marker="|", linewidths=1.6)
    a.axvline(1, c=MUTED, lw=0.7, ls="--"); a.set_xscale("log"); a.set_xticks([0.1, 0.5, 1, 2, 4]); a.set_xticklabels(["0.1", "0.5", "1", "2", "4"])
    a.minorticks_off(); a.set_yticks(range(len(nm))); a.set_yticklabels(nm, fontsize=7); a.invert_yaxis()
    a.set_xlabel("Predicted / true rate"); a.set_title("All 25 draws", fontweight="normal")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=5, fontsize=7, handlelength=1.6, columnspacing=1.0)
    fig.savefig(FIG / "fig_calibration.png"); plt.close(fig)


if __name__ == "__main__":
    fig_maps(); fig_headline(); fig_detection(); fig_rule(); fig_calibration()
    print("figures written to", FIG)
