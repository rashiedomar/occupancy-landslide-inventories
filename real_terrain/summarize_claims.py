"""
Collect every experiment into one claims-and-evidence table (CLAIMS.md + claims_table.json).
Each claim is tested with paired comparisons on identical draws (sign count + Wilcoxon signed-rank).
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).parent


def load(pattern_dir, key_map=None):
    out = {}
    for f in sorted((HERE / pattern_dir).glob("raw_results_*.json")):
        out.update(json.loads(f.read_text()))
    return out


def draws(dct):
    """{seed: result} or [result, ...] -> list ordered by seed."""
    if isinstance(dct, list):
        return dct
    return [dct[k] for k in sorted(dct, key=int)]


def paired(a, b, metric, higher_better=True):
    x = np.array([r[metric] for r in a]); y = np.array([r[metric] for r in b])
    d = (x - y) if higher_better else (y - x)            # positive = a better
    wins = int((d > 0).sum())
    p = float(wilcoxon(d).pvalue) if len(d) >= 5 and np.any(d != 0) else float("nan")
    return dict(n=len(d), wins=wins, mean_a=float(x.mean()), mean_b=float(y.mean()),
                mean_gain=float(d.mean()), p=p)


def _add_rate(rs):
    for r in rs:
        r["abs_log_rate"] = float(abs(np.log(r["mean_pred_over_true"])))
    return rs


def collect():
    sets = {}
    v3 = load("results_v3")
    st = json.loads((HERE / "results_steger/raw_results.json").read_text())
    v4b = {}
    for f in (HERE / "results_v4").glob("raw_results_B_*.json"):
        v4b.update(json.loads(f.read_text()))
    for scen in ["A", "C"]:
        sets[f"Iburi-{scen}"] = {
            "standard": [r["Standard (terrain + access)"] for r in draws(v3[scen])],
            "steger": [r["Steger-MLP (access = mean)"] for r in draws(st[scen])],
            "steger_glm": [r["Steger-GLM (access neutralised)"] for r in draws(st[scen])],
            "ours": [r["Occupancy + monotone + anchor (FULL)"] for r in draws(v3[scen])],
            "oracle": [r["Oracle (full real inventory)"] for r in draws(v3[scen])]}
    sets["Iburi-B"] = {"standard": [r["Standard (terrain + access)"] for r in draws(v3["B"])],
                       "ours_free_roads": [r["Occupancy + monotone + anchor (FULL)"] for r in draws(v3["B"])],
                       "ours": [r["FULL + short-range road"] for r in draws(v4b)],
                       "oracle": [r["Oracle (full real inventory)"] for r in draws(v3["B"])]}
    for scen in ["D", "E"]:
        f = HERE / f"results_extra/raw_results_{scen}.json"
        if f.exists():
            r = draws(json.loads(f.read_text()))
            sets[f"Iburi-{scen}"] = {"standard": [x["Standard (terrain + access)"] for x in r],
                                     "steger": [x["Steger-MLP (access = mean)"] for x in r],
                                     "ours": [x["OURS: + short-range road term" if scen == "D" else
                                                "OURS: terrain-only susceptibility"] for x in r],
                                     "oracle": [x["Oracle"] for x in r]}
            if scen == "D":
                sets["Iburi-D"]["ours_terrain_only"] = [x["OURS: terrain-only susceptibility"] for x in r]
                sets["Iburi-D"]["ours_free_roads"] = [x["OURS: roads enter susceptibility freely"] for x in r]
    for scen in ["A", "C"]:
        rr = {}
        for f in (HERE / "results_gorkha").glob(f"raw_results_{scen}*.json"):
            rr.update(json.loads(f.read_text()))
        if rr:
            r = draws(rr)
            sets[f"Gorkha-{scen}"] = {"standard": [x["Standard (terrain + access)"] for x in r],
                                      "steger": [x["Steger-MLP (access = mean)"] for x in r],
                                      "steger_glm": [x["Steger-GLM (access neutralised)"] for x in r],
                                      "ours": [x["OURS full"] for x in r],
                                      "ours_roadcut": [x["OURS + short-range road"] for x in r],
                                      "oracle": [x["Oracle"] for x in r]}
    for v in sets.values():
        for rs in v.values():
            _add_rate(rs)
    return sets


def main():
    sets = collect()
    table = {}
    lines = ["| setting | comparison | metric | ours better | mean (ours vs other) | Wilcoxon p |",
             "|---|---|---|---|---|---|"]
    for name, s in sets.items():
        table[name] = {}
        for other in [k for k in ["standard", "steger", "steger_glm", "ours_free_roads", "ours_terrain_only"] if k in s]:
            for met, hb, lab in [("auc_true_landslides", True, "AUC"), ("auc_remote_quarter", True, "remote-¼ AUC"),
                                 ("abs_log_ratio", False, "road bias")]:
                r = paired(s["ours"], s[other], met, hb)
                table[name][f"{other}|{met}"] = r
                lines.append(f"| {name} | ours vs {other} | {lab} | {r['wins']}/{r['n']} | "
                             f"{r['mean_a']:.3f} vs {r['mean_b']:.3f} | {r['p']:.3f} |")
        if "near_road_lift_ratio" in s["ours"][0]:
            for k in ["ours", "ours_terrain_only", "ours_free_roads", "standard", "steger", "oracle"]:
                if k in s:
                    v = [x["near_road_lift_ratio"] for x in s[k]]
                    table[name][f"lift|{k}"] = float(np.mean(v))
                    lines.append(f"| {name} | near-road lift (1 = true planted effect) | {k} | | "
                                 f"{np.mean(v):.2f} ± {np.std(v):.2f} | |")
    # pooled tests across settings (draws within one earthquake share terrain and truth -> not independent)
    lines += ["", "**Pooled across settings**", "",
              "| comparison | settings | metric | ours better | mean gain | Wilcoxon p |", "|---|---|---|---|---|---|"]
    main_sets = [k for k in sets if k.split("-")[1] in ("A", "C")]     # B/D/E are separate claims
    pools = {"ours vs steger (MLP)": ("steger", main_sets), "ours vs steger (GLM)": ("steger_glm", main_sets),
             "ours vs standard": ("standard", main_sets)}
    for lab, (other, names) in pools.items():
        for met, hb, mlab in [("auc_true_landslides", True, "AUC"), ("auc_remote_quarter", True, "remote-¼ AUC"),
                              ("abs_log_ratio", False, "road bias"), ("abs_log_rate", False, "calibration |log rate ratio|")]:
            a = [x for n in names for x in sets[n]["ours"]]; b = [x for n in names for x in sets[n][other]]
            r = paired(a, b, met, hb)
            table.setdefault("_pooled", {})[f"{lab}|{met}"] = dict(r, settings=names)
            lines.append(f"| {lab} | {', '.join(names)} | {mlab} | {r['wins']}/{r['n']} | {r['mean_gain']:+.3f} | {r['p']:.4f} |")
    (HERE / "claims_table.json").write_text(json.dumps(table, indent=1))
    (HERE / "claims_table.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
