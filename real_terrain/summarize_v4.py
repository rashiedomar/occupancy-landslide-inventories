"""Merge Stage 3 baselines with Stage 4 variants (same draws) -> table, paired wins, figure."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
V3, V4, FIG = HERE / "results_v3", HERE / "results_v4", HERE / "figures"

v3 = {}
for f in V3.glob("raw_results_*.json"):
    v3.update(json.loads(f.read_text()))
v4 = {"B": {}, "C": {}}
for f in V4.glob("raw_results_*.json"):
    v4[f.stem.split("_")[2][0]].update({int(k): v for k, v in json.loads(f.read_text()).items()})

BASE = ["Standard (terrain + access)", "Occupancy + anchor", "Occupancy + monotone + anchor (FULL)",
        "Oracle (full real inventory)"]
METS = [("auc_true_landslides", "AUC"), ("auc_remote_quarter", "AUC remote ¼"),
        ("abs_log_ratio", "|log bias|"), ("remote_near_ratio", "bias ratio (median)"),
        ("mean_pred_over_true", "rate ÷ true")]


def merged(scen):
    rows = {}
    seeds = sorted(v4[scen])
    for name in BASE:
        rows[name] = [v3[scen][s][name] for s in seeds]
    for name in v4[scen][seeds[0]]:
        if not name.startswith("_"):
            rows[name] = [v4[scen][s][name] for s in seeds]
    learned = {s: v4[scen][s]["_learned"] for s in seeds}
    return rows, seeds, learned


lines = []
for scen in ["B", "C"]:
    if not v4[scen]:
        continue
    rows, seeds, learned = merged(scen)
    lines.append(f"\n=== Scenario {scen}  (draws {seeds}; same data as Stage 3) ===")
    lines.append(f"{'model':40s}" + "".join(f"{lab:>20s}" for _, lab in METS))
    for name, rs in rows.items():
        row = f"{name:40s}"
        for met, _ in METS:
            v = np.array([r[met] for r in rs])
            row += f"{np.median(v):>20.2f}" if met == "remote_near_ratio" else f"{v.mean():>13.3f} ± {v.std():.3f}"
        lines.append(row)
    std = rows["Standard (terrain + access)"]; full = rows["Occupancy + monotone + anchor (FULL)"]
    for name, rs in rows.items():
        if name in BASE:
            continue
        w_auc = sum(a["auc_true_landslides"] > b["auc_true_landslides"] for a, b in zip(rs, std))
        w_bias = sum(a["abs_log_ratio"] < b["abs_log_ratio"] for a, b in zip(rs, std))
        w_full = sum(a["auc_true_landslides"] > b["auc_true_landslides"] for a, b in zip(rs, full))
        lines.append(f"  {name}: beats Standard on AUC {w_auc}/{len(rs)}, on bias {w_bias}/{len(rs)}; "
                     f"beats Stage-3 FULL on AUC {w_full}/{len(rs)}")
    for name in rows:
        if "short-range" in name:
            ls = [x for s in seeds for x in learned[s][name] if isinstance(x[0], float)]
            lines.append(f"  learned road-cut term ({name}): strength {np.mean([a for a, _ in ls]):.2f} "
                         f"(log-odds at the road), range {np.mean([l for _, l in ls]) * 1000:.0f} m")
txt = "\n".join(lines)
print(txt)
(V4 / "summary.txt").write_text(txt)

# figure: paired per-draw AUC and bias vs Standard practice
fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.4), constrained_layout=True)
cols = {"Occupancy + monotone + anchor (FULL)": "#08519c", "Occupancy + anchor": "#9ecae1",
        "FULL + spatial": "#e6550d", "FULL + short-range road": "#31a354",
        "FULL + spatial + short-range road": "#756bb1", "Oracle (full real inventory)": "#252525"}
for scen, mk in [("B", "o"), ("C", "s")]:
    if not v4[scen]:
        continue
    rows, seeds, _ = merged(scen)
    std = rows["Standard (terrain + access)"]
    for name, rs in rows.items():
        if name not in cols:
            continue
        x = [r["auc_true_landslides"] - s["auc_true_landslides"] for r, s in zip(rs, std)]
        y = [s["abs_log_ratio"] - r["abs_log_ratio"] for r, s in zip(rs, std)]
        lab = f"{name} [{scen}]"
        ax[0].scatter(x, y, c=cols[name], marker=mk, s=28, label=lab, alpha=0.85)
        ax[1].errorbar(np.mean(x), np.mean(y), xerr=np.std(x), yerr=np.std(y), fmt=mk, c=cols[name],
                       capsize=2, label=lab)
for a in ax:
    a.axhline(0, c="k", lw=0.6); a.axvline(0, c="k", lw=0.6)
    a.set_xlabel("AUC gain over standard practice  (→ more accurate)")
    a.set_ylabel("road-bias reduction vs standard  (↑ less biased)")
ax[0].set_title("every draw (circle = scenario B, square = C)")
ax[1].set_title("mean ± sd")
ax[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.suptitle("Stage 4 — paired against standard practice on identical draws. Top-right = better on both.",
             fontsize=10)
fig.savefig(FIG / "v4_paired.png", dpi=150)
print("figure saved")
