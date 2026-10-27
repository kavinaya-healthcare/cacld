"""
Export the India CSPH Score models to compact JSON for the browser calculator in web/.
Run from the repository root:  python export_web_model.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, joblib

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "web" / "model"; OUT.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(ROOT / "model_tuning_params_india.json"))
zp = json.load(open(ROOT / "z_score_params_india.json"))
GROUP = {"VCTE-50Hz": "VCTE", "VCTE-100Hz": "VCTE", "P-SWE": "P-SWE", "2D-SWE": "2D-SWE"}

def export_forest(model, features, path):
    trees = []
    for est in model.estimators_:
        t = est.tree_
        v = t.value[:, 0, :]; p1 = v[:, 1] / v.sum(axis=1)          # class-1 probability at every node
        trees.append({"f": t.feature.tolist(), "t": t.threshold.tolist(),
                      "l": t.children_left.tolist(), "r": t.children_right.tolist(), "p": np.round(p1, 8).tolist()})
    json.dump({"features": features, "trees": trees}, open(path, "w"), separators=(",", ":"))
    return path.stat().st_size

s1 = export_forest(joblib.load(ROOT / "rf_model_india_tier1.joblib"), cfg["tier1"]["features"], OUT / "tier1.json")
s2 = export_forest(joblib.load(ROOT / "rf_model_india_tier2.joblib"), cfg["tier2"]["features"], OUT / "tier2.json")

tr = pd.read_csv(ROOT / "synthetic_india_training.csv")
g = tr["Elastography Modality"].map(GROUP)
tr["LS_z"] = [(v - zp[k]["LS"]["mean"]) / zp[k]["LS"]["std"] for v, k in zip(tr["LS"], g)]
tr["SS_z"] = [(v - zp[k]["SS"]["mean"]) / zp[k]["SS"]["std"] if pd.notna(v) else np.nan for v, k in zip(tr["SS"], g)]

def dist(x, lo, hi):
    x = np.asarray(x.dropna(), float); grid = np.linspace(lo, hi, 140)
    bw = 1.06 * x.std() * len(x) ** -0.2
    dens = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bw) ** 2).sum(1) / (len(x) * bw * np.sqrt(2 * np.pi))
    return {"grid": np.round(grid, 4).tolist(), "density": np.round(dens, 6).tolist(),
            "quantiles": np.round(np.quantile(x, np.linspace(0, 1, 101)), 4).tolist()}

meta = {
    "name": "India CSPH Score", "synthetic": True,
    "z_score_params": zp, "group": GROUP,
    "tier1": {"features": cfg["tier1"]["features"], "cutoffs": cfg["tier1"]["cutoffs"], "cv_auc": cfg["tier1"]["cv_auc"], "params": cfg["tier1"]["best_params"]},
    "tier2": {"features": cfg["tier2"]["features"], "cutoffs": cfg["tier2"]["cutoffs"], "cv_auc": cfg["tier2"]["cv_auc"], "params": cfg["tier2"]["best_params"]},
    "training": cfg["training_data"], "external": cfg["external_validation_synthetic_centre"],
    "distributions": {"LS_zscore": dist(tr["LS_z"], -2.5, 4.5), "SS_zscore": dist(tr["SS_z"], -2.5, 4.5), "PLT": dist(tr["PLT"], 0, 420)},
}
json.dump(meta, open(OUT / "meta.json", "w"), indent=1)
print(f"tier1.json {s1/1e3:.0f} kB, tier2.json {s2/1e3:.0f} kB, meta.json written")
