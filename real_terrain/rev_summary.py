"""Collect the revision checks and per-site statistics into results_rev/summary.json and print them."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

HERE = Path(__file__).parent
R = HERE / "results_rev"
sys.path.insert(0, str(HERE))
from summarize_claims import collect, draws  # noqa: E402

out = {}
FL = {}
for w in ["iburi_L", "iburi_N", "iburi_W1", "iburi_W2", "iburi_W3", "iburi_W1a", "gorkha_H1", "gorkha_L", "gorkha_N"]:
    res = {}
    for f in sorted((HERE / "results_flip" / w).glob("raw_results_A*.json")):
        res.update(json.loads(f.read_text()))
    for f in sorted(R.glob(f"flip_rerun_{w}_*.json")):       # draws absent from results_flip, re-run for the revision
        res.setdefault(f.stem.rsplit("_", 1)[1], json.loads(f.read_text()))
    FL[w] = [v for _, v in sorted(res.items(), key=lambda kv: int(kv[0]))]

auc = lambda rs, k: np.array([r[k]["auc_true_landslides"] for r in rs])
print("\n=== Rule: world, truth gap, records gap, ours-best(MLP,GLM), wins, ours-best(+GAM), wins")
rows = []
for w, rs in FL.items():
    rc = json.loads((R / f"rule_checks_{w}_A.json").read_text())
    gam = np.array([r["auc_true_landslides"] for r in json.loads((R / f"gam_{w}_A.json").read_text())])[:len(rs)]
    o, m_, g = auc(rs, "OURS full"), auc(rs, "Steger-MLP (access = mean)"), auc(rs, "Steger-GLM (access neutralised)")
    d = o - np.maximum(m_, g); d3 = o - np.maximum(np.maximum(m_, g), gam)
    rows.append(dict(world=w, truth_gap=rc["truth_gap"], records_gap=rc["records_gap"], diff=d.mean(), wins=int((d > 0).sum()),
                     diff3=d3.mean(), wins3=int((d3 > 0).sum()), gam=gam.mean(), mlp=m_.mean(), glm=g.mean(), ours=o.mean()))
sets = collect()
hb = {}
for f in sorted((HERE / "results_haiti").rglob("raw_results_A*.json")):
    hb.update(json.loads(f.read_text()))
hb = [v for _, v in sorted(hb.items(), key=lambda kv: int(kv[0]))]
sites = {"iburi_A": sets["Iburi-A"], "iburi_C": sets["Iburi-C"], "gorkha_A": sets["Gorkha-A"], "gorkha_C": sets["Gorkha-C"]}
for name, s in sites.items():
    rc = json.loads((R / f"rule_checks_{name}.json").read_text())
    gam = np.array([r["auc_true_landslides"] for r in json.loads((R / f"gam_{name}.json").read_text())])
    a = lambda k: np.array([r["auc_true_landslides"] for r in s[k]])
    o, m_, g = a("ours"), a("steger"), a("steger_glm")
    d = o - np.maximum(m_, g); d3 = o - np.maximum(np.maximum(m_, g), gam)
    rows.append(dict(world=name, truth_gap=rc["truth_gap"], records_gap=rc["records_gap"], diff=d.mean(), wins=int((d > 0).sum()),
                     diff3=d3.mean(), wins3=int((d3 > 0).sum()), gam=gam.mean(), mlp=m_.mean(), glm=g.mean(), ours=o.mean()))
rc = json.loads((R / "rule_checks_haiti_A.json").read_text())
gam = np.array([r["auc_true_landslides"] for r in json.loads((R / "gam_haiti_A.json").read_text())])
o, m_, g = auc(hb, "OURS full"), auc(hb, "Steger-MLP (access = mean)"), auc(hb, "Steger-GLM (access neutralised)")
d = o - np.maximum(m_, g); d3 = o - np.maximum(np.maximum(m_, g), gam)
rows.append(dict(world="haiti_A", truth_gap=rc["truth_gap"], records_gap=rc["records_gap"], diff=d.mean(), wins=int((d > 0).sum()),
                 diff3=d3.mean(), wins3=int((d3 > 0).sum()), gam=gam.mean(), mlp=m_.mean(), glm=g.mean(), ours=o.mean()))
for r in rows:
    print(f"{r['world']:11s} truth {r['truth_gap']:+.3f} rec {r['records_gap']:+.3f} | ours {r['ours']:.3f} mlp {r['mlp']:.3f} glm {r['glm']:.3f} gam {r['gam']:.3f} | "
          f"d2 {r['diff']:+.3f} {r['wins']}/5 | d3 {r['diff3']:+.3f} {r['wins3']}/5")
out["rule"] = rows
for key in ["truth_gap", "records_gap"]:
    rho = spearmanr([r[key] for r in rows], [r["diff3"] for r in rows])
    print(key, "spearman vs ours-best(3):", round(rho.statistic, 2), "p", round(rho.pvalue, 3))
print("real gorkha", json.loads((R / "rule_checks_gorkha_real.json").read_text())["records_gap"])
out["real_records_gap"] = json.loads((R / "rule_checks_gorkha_real.json").read_text())["records_gap"]

# rule outcomes with the three bands
def band(g):
    return "occ" if g > 0.03 else "neu" if g < 0.015 else "none"
for key in ["truth_gap", "records_gap"]:
    print(key, [(r["world"], band(r[key]), "occ" if r["diff3"] > 0 else "neu") for r in rows])

print("\n=== Per-site paired: ours vs standard")
per = {}
for name, s in sites.items():
    f = lambda m, k: np.array([r[m] for r in s[k]])
    dA = f("auc_true_landslides", "ours") - f("auc_true_landslides", "standard")
    dB = f("abs_log_ratio", "standard") - f("abs_log_ratio", "ours")
    per[name] = dict(auc_gain=dA.mean(), auc_sd=dA.std(ddof=1), auc_wins=int((dA > 0).sum()), bias_gain=dB.mean(), bias_wins=int((dB > 0).sum()),
                     **{f"auc_sd_{k}": f("auc_true_landslides", k).std(ddof=1) for k in s if k in ("standard", "steger", "steger_glm", "ours", "oracle")})
    print(name, {k: round(v, 3) for k, v in per[name].items()})
out["per_site"] = per

print("\n=== Real inventories (paired)")
q = json.loads((HERE / "results_q1/raw_results.json").read_text())
q = [q[k] for k in sorted(q, key=int)]
O = "OURS (access detection)"
real = {}
for k in q[0]:
    if k.startswith("_") or k == O or "spillover" in k:
        continue
    d = np.array([r[O]["auc_reference"] - r[k]["auc_reference"] for r in q])
    real[k] = dict(mean=d.mean(), wins=int((d > 0).sum()), p=float(wilcoxon(d).pvalue))
    print(f"ours - {k:40s} {d.mean():+.4f} wins {real[k]['wins']}/5 p {real[k]['p']:.3f}")
out["real_paired"] = real

print("\n=== Calibration")
cal = {}
for site in ["iburi_A", "iburi_C", "gorkha_A", "gorkha_C", "haiti_A"]:
    fs = sorted(R.glob(f"calib_{site}_*.json"))
    rs = [json.loads(f.read_text()) for f in fs if not f.name.startswith("._")]
    if not rs:
        continue
    print(site, "n", len(rs), "anchor slides", [r["_anchor"]["landslide_cells"] for r in rs], "label freq", [round(r["_anchor"]["label_freq"], 2) for r in rs])
    cal[site] = {}
    for m in ["standard", "steger_mlp", "steger_glm", "standard_anchor_scaled", "steger_mlp_anchor_scaled", "steger_glm_anchor_scaled", "ours", "oracle"]:
        v = [r[m] for r in rs if m in r]
        if not v:
            continue
        g = lambda k: np.array([x[k] for x in v])
        cal[site][m] = dict(auc=g("auc").mean(), rate_median=float(np.median(g("rate_ratio"))), rate_min=g("rate_ratio").min(), rate_max=g("rate_ratio").max(),
                            abs_log_rate=np.abs(np.log(g("rate_ratio"))).mean(), brier=g("brier").mean(), log_loss=g("log_loss").mean(), ece=g("ece").mean(), slope=g("cal_slope").mean())
        print(f"  {m:26s} auc {cal[site][m]['auc']:.3f} rate med {cal[site][m]['rate_median']:.2f} [{cal[site][m]['rate_min']:.2f},{cal[site][m]['rate_max']:.2f}] |log| {cal[site][m]['abs_log_rate']:.2f} brier {cal[site][m]['brier']:.4f} ll {cal[site][m]['log_loss']:.4f} ece {cal[site][m]['ece']:.4f} slope {cal[site][m]['slope']:.2f}")
out["calibration"] = cal

allr = []
for site in cal:
    allr += [json.loads(f.read_text()) for f in sorted(R.glob(f"calib_{site}_*.json")) if not f.name.startswith("._")]
print("\n=== Pooled over all calibration draws, n =", len(allr))
pool = {}
for m in ["standard", "steger_mlp", "steger_glm", "standard_anchor_scaled", "steger_mlp_anchor_scaled", "steger_glm_anchor_scaled", "ours", "oracle"]:
    v = [r[m] for r in allr if m in r]
    rr = np.array([x["rate_ratio"] for x in v])
    pool[m] = dict(n=len(v), rate_median=float(np.median(rr)), rate_min=rr.min(), rate_max=rr.max(), abs_log=np.abs(np.log(rr)).mean(),
                   ece=np.mean([x["ece"] for x in v]), brier=np.mean([x["brier"] for x in v]), ll=np.mean([x["log_loss"] for x in v]), slope=np.median([x["cal_slope"] for x in v]))
    print(f"  {m:26s} {pool[m]}")
out["pool"] = pool
for b in ["standard_anchor_scaled", "steger_mlp_anchor_scaled", "steger_glm_anchor_scaled", "steger_glm", "steger_mlp", "standard"]:
    for met, key in [("|log rate|", lambda x: abs(np.log(x["rate_ratio"]))), ("ECE", lambda x: x["ece"]), ("log loss", lambda x: x["log_loss"]), ("brier", lambda x: x["brier"])]:
        pairs = [(key(r["ours"]), key(r[b])) for r in allr if b in r]
        d = np.array([y - x for x, y in pairs])
        print(f"  ours vs {b:26s} {met:10s} ours better {(d > 0).sum()}/{len(d)} mean gain {d.mean():+.4f} p {wilcoxon(d).pvalue:.4f}")

print("\n=== Slope-road correlation, prevalence")
from run_iburi import _z, load_grid
P = HERE.parent / "data/processed"
for s in ["iburi", "gorkha", "haiti"]:
    d, m, terrain, dist = load_grid(P / f"{s}_grid.npz")
    z = d["z"][m]
    corr = {k: round(float(np.corrcoef(terrain["slope_mean"][m], np.log1p(dist[k][m]))[0, 1]), 2) for k in ["major", "public", "track"]}
    dp = dist["public"][m]
    print(s, "prev", round(float(z.mean()), 4), "cells", int(m.sum()), "slope-road r", corr, "median d_pub landslide vs other", round(float(np.median(dp[z == 1])), 2), round(float(np.median(dp[z == 0])), 2), "max d", round(float(dp.max()), 1))
(R / "summary.json").write_text(json.dumps(out, indent=1, default=float))
