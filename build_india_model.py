"""
India CSPH Score: build script
==============================
Generates an India-adjusted SYNTHETIC cohort, derives India-specific z-score
parameters, tunes and trains a two-tier random forest (Tier 1 without spleen
stiffness, Tier 2 with it), sets India-specific cut-offs and validates on a
held-out synthetic centre.

RESEARCH / TEACHING USE ONLY. The data are synthetic, generated from published
summary statistics. Nothing produced here is clinically validated.

Run:  python build_india_model.py
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss
warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent
SEED = 2026

# --------------------------------------------------------------------------
# 1. Generator.  Every value is tagged:  [IN] Indian source  [INT] international
#    source used where no Indian data exist  [ASM] modelling assumption
# --------------------------------------------------------------------------
P = dict(
    hvpg_median=11.5, hvpg_logsd=0.45,          # [ASM] gives ~62% CSPH, close to ELM cACLD 63% [INT]
    p_male=0.81, age_mean=52, age_sd=11,        # [IN] Kumar 2017 (Delhi)
    bmi_median=23.0, bmi_logsd=0.15,            # [IN] Kumar 2017
    etiology={"Metabolic": 0.30, "Cryptogenic": 0.15, "Alcohol": 0.34, "Viral": 0.15, "Autoimmune": 0.02, "Other/mixed": 0.04},  # [IN] Kumar 2017: NASH/cryptogenic 45%, alcohol 34%, viral 15%, other 6%
    ls_median=19.0, ls_slope=1.0, ls_noise=0.42,   # [ASM] weaker LSM-HVPG coupling than European data, per Kumar 2017 [IN]
    ls_etio={"Alcohol": 0.12, "Metabolic": 0.06},  # [ASM] inflammation/steatosis raise LSM for a given HVPG
    ss_median=45.0, ss_slope=0.75, ss_noise=0.22,  # [INT] ELM: SSM tracks HVPG more tightly than LSM
    plt_median=118.0, plt_slope=-0.85, plt_noise=0.30,  # [INT] ELM PLT by CSPH x 0.82 India shift [IN] platelet reference 2022
    p_macro=0.04, macro_factor=0.6,                 # [IN] macrothrombocytopenia in Indian donors 2025
    p_ssm=0.35,                                     # [ASM] SSM available in a minority (proposal premise)
    device_p={"VCTE-50Hz": 0.62, "VCTE-100Hz": 0.08, "2D-SWE": 0.18, "P-SWE": 0.12},  # [ASM] FibroScan dominant in India
    device_scale={"VCTE": (1.00, 1.00), "2D-SWE": (0.86, 0.70), "P-SWE": (0.82, 0.80)},  # [ASM] (LS, SS) device offsets
)
DEV_GROUP = {"VCTE-50Hz": "VCTE", "VCTE-100Hz": "VCTE", "2D-SWE": "2D-SWE", "P-SWE": "P-SWE"}

def generate(n, seed, device_p=None):
    rng = np.random.default_rng(seed)
    hv = np.exp(rng.normal(np.log(P["hvpg_median"]), P["hvpg_logsd"], n))
    z = np.log(hv) - np.log(P["hvpg_median"])                 # shared latent severity -> realistic correlations
    df = pd.DataFrame({"Patient_ID": [f"IN-{seed}-{i:05d}" for i in range(n)]})
    df["Sex"] = (rng.random(n) < P["p_male"]).astype(int)
    df["Age"] = np.clip(rng.normal(P["age_mean"], P["age_sd"], n), 18, 85).round().astype(int)
    et = rng.choice(list(P["etiology"]), n, p=list(P["etiology"].values()))
    bmi = np.exp(rng.normal(np.log(P["bmi_median"]), P["bmi_logsd"], n)) + np.where(et == "Metabolic", 2.5, 0)
    df["BMI"] = bmi.round(1)
    dp = device_p or P["device_p"]
    dev = rng.choice(list(dp), n, p=list(dp.values())); grp = np.array([DEV_GROUP[d] for d in dev])
    ls_s = np.array([P["device_scale"][g][0] for g in grp]); ss_s = np.array([P["device_scale"][g][1] for g in grp])
    etio_off = np.array([P["ls_etio"].get(e, 0.0) for e in et])
    ls = np.exp(np.log(P["ls_median"]) + P["ls_slope"] * z + etio_off + rng.normal(0, P["ls_noise"], n)) * ls_s
    ss = np.exp(np.log(P["ss_median"]) + P["ss_slope"] * z + rng.normal(0, P["ss_noise"], n)) * ss_s
    plt_ = np.exp(np.log(P["plt_median"]) + P["plt_slope"] * z + rng.normal(0, P["plt_noise"], n))
    macro = rng.random(n) < P["p_macro"]; plt_[macro] *= P["macro_factor"]
    # Child-Pugh 5/6/7 rises with severity [INT] ELM distribution by CSPH
    u = rng.random(n); p7 = np.clip(0.04 + 0.10 * (z > 0) + 0.06 * z, 0.02, 0.3); p6 = np.clip(0.12 + 0.15 * (z > 0), 0.05, 0.4)
    cpt = np.where(u < p7, 7, np.where(u < p7 + p6, 6, 5))
    df["Elastography Modality"] = dev
    df["LS"] = np.clip(ls, 5, 75).round(1)
    df["SS"] = np.where(rng.random(n) < P["p_ssm"], np.clip(ss, 8, 100).round(1), np.nan)
    df["PLT"] = np.clip(plt_, 25, 420).round().astype(int)
    df["CPT_Score"] = cpt
    for e in P["etiology"]: df[f"{e} Etiology"] = (et == e).astype(int)
    df["Macrothrombocytopenia"] = macro.astype(int)
    df["HVPG"] = hv.round(1)
    df["CSPH"] = (hv >= 10).astype(int)
    df["Synthetic"] = 1
    return df

train = generate(2400, SEED)
external = generate(600, SEED + 1, device_p={"VCTE-50Hz": 0.45, "VCTE-100Hz": 0.10, "2D-SWE": 0.30, "P-SWE": 0.15})  # held-out "centre", different device mix

# --------------------------------------------------------------------------
# 2. India-specific z-score parameters (training data only, frozen)
# --------------------------------------------------------------------------
zp = {}
for g in ["VCTE", "P-SWE", "2D-SWE"]:
    d = train[train["Elastography Modality"].map(DEV_GROUP) == g]
    zp[g] = {"LS": {"mean": float(d["LS"].mean()), "std": float(d["LS"].std())},
             "SS": {"mean": float(d["SS"].mean()), "std": float(d["SS"].std())}}
json.dump(zp, open(OUT / "z_score_params_india.json", "w"), indent=4)

def add_z(df):
    df = df.copy(); g = df["Elastography Modality"].map(DEV_GROUP)
    df["LS_zscore"] = [(r - zp[k]["LS"]["mean"]) / zp[k]["LS"]["std"] for r, k in zip(df["LS"], g)]
    df["SS_zscore"] = [(r - zp[k]["SS"]["mean"]) / zp[k]["SS"]["std"] if pd.notna(r) else np.nan for r, k in zip(df["SS"], g)]
    return df
train, external = add_z(train), add_z(external)

TIER1 = ["Age", "Sex", "BMI", "PLT", "CPT_Score", "Alcohol Etiology", "Metabolic Etiology", "Viral Etiology", "Cryptogenic Etiology", "LS_zscore"]
TIER2 = TIER1 + ["SS_zscore"]

# --------------------------------------------------------------------------
# 3. Hyperparameter tuning (5-fold CV, AUC) and India-specific cut-offs
# --------------------------------------------------------------------------
GRID = {"n_estimators": [300], "max_depth": [4, 6, 8], "min_samples_leaf": [5, 15],
        "max_features": ["sqrt", 0.5], "criterion": ["entropy"]}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

def tune(X, y):
    gs = GridSearchCV(RandomForestClassifier(random_state=SEED, class_weight="balanced"), GRID, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(X, y)
    best = RandomForestClassifier(random_state=SEED, class_weight="balanced", **gs.best_params_)
    oof = cross_val_predict(best, X, y, cv=cv, method="predict_proba")[:, 1]
    best.fit(X, y)
    return best, gs.best_params_, float(gs.best_score_), oof

def cutoffs(y, p, target=0.95):
    ys, ps = np.asarray(y), np.asarray(p)
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    lo = max([c for c in grid if ((ps > c) & (ys == 1)).sum() / max(1, (ys == 1).sum()) >= target], default=0.05)   # rule-out: sensitivity >= target
    hi = min([c for c in grid if ((ps < c) & (ys == 0)).sum() / max(1, (ys == 0).sum()) >= target], default=0.95)   # rule-in: specificity >= target
    return float(lo), float(max(hi, lo + 0.01))

tr2 = train[train["SS"].notna()]
m1, bp1, cv1, oof1 = tune(train[TIER1], train["CSPH"])
m2, bp2, cv2, oof2 = tune(tr2[TIER2], tr2["CSPH"])
lo1, hi1 = cutoffs(train["CSPH"], oof1)
lo2, hi2 = cutoffs(tr2["CSPH"], oof2)

joblib.dump(m1, OUT / "rf_model_india_tier1.joblib")
joblib.dump(m2, OUT / "rf_model_india_tier2.joblib")

# --------------------------------------------------------------------------
# 4. External (held-out centre) evaluation, incl. Baveno VII and original ELM
# --------------------------------------------------------------------------
def classify(p, lo, hi): return np.where(p <= lo, "Rule-out", np.where(p >= hi, "Rule-in", "Grey zone"))
def summary(y, cls, prob=None):
    y = np.asarray(y); cls = np.asarray(cls)
    ro, ri = cls == "Rule-out", cls == "Rule-in"
    r = {"grey_zone_pct": round(100 * (cls == "Grey zone").mean(), 1),
         "NPV": round(float((y[ro] == 0).mean()), 3) if ro.any() else None,
         "PPV": round(float((y[ri] == 1).mean()), 3) if ri.any() else None,
         "n": int(len(y))}
    if prob is not None: r.update(AUC=round(float(roc_auc_score(y, prob)), 3), Brier=round(float(brier_score_loss(y, prob)), 3))
    return r

ex = external.copy()
p1 = m1.predict_proba(ex[TIER1])[:, 1]; c1 = classify(p1, lo1, hi1)
final = c1.copy(); has_ss = ex["SS"].notna().values
p2 = np.full(len(ex), np.nan)
if has_ss.any(): p2[has_ss] = m2.predict_proba(ex.loc[has_ss, TIER2])[:, 1]
step2 = (c1 == "Grey zone") & has_ss
final[step2] = classify(p2[step2], lo2, hi2)
bav = np.where((ex["LS"] <= 15) & (ex["PLT"] >= 150), "Rule-out", np.where(ex["LS"] >= 25, "Rule-in", "Grey zone"))
res = {"Tier 1 alone (no SSM)": summary(ex["CSPH"], c1, p1),
       "Tiered pathway (Tier 1, then Tier 2 if SSM available)": summary(ex["CSPH"], final),
       "Baveno VII (LSM + platelets)": summary(ex["CSPH"], bav)}
ws = ex[has_ss]
res["Tier 2 on patients with SSM"] = summary(ws["CSPH"], classify(p2[has_ss], lo2, hi2), p2[has_ss])

orig = Path("/mnt/user-data/uploads/rf_model.joblib")
if orig.exists():  # benchmark: original ELM (European) on the same synthetic Indian patients
    em = joblib.load(orig); ezp = json.load(open("/mnt/user-data/uploads/z_score_params.json"))
    e = ws.copy(); g = e["Elastography Modality"].map(DEV_GROUP)
    e["LS_zscore"] = [(v - ezp[k]["LS"]["mean"]) / ezp[k]["LS"]["std"] for v, k in zip(e["LS"], g)]
    e["SS_zscore"] = [(v - ezp[k]["SS"]["mean"]) / ezp[k]["SS"]["std"] for v, k in zip(e["SS"], g)]
    pe = em.predict_proba(e[["Age", "PLT", "CPT_Score", "Viral Etiology", "Sex", "LS_zscore", "SS_zscore"]])[:, 1]
    res["Original ELM (European model, published cut-offs) on patients with SSM"] = summary(e["CSPH"], classify(pe, 0.45, 0.60), pe)

imp1 = dict(sorted(zip(TIER1, np.round(m1.feature_importances_, 3).tolist()), key=lambda x: -x[1]))
imp2 = dict(sorted(zip(TIER2, np.round(m2.feature_importances_, 3).tolist()), key=lambda x: -x[1]))
tuning = {
    "model": "India CSPH Score (two-tier random forest), SYNTHETIC research prototype",
    "random_state": SEED,
    "training_data": {"file": "synthetic_india_training.csv", "n": len(train), "n_with_SSM": len(tr2), "CSPH_prevalence": round(float(train["CSPH"].mean()), 3)},
    "search": {"method": "GridSearchCV", "cv": "StratifiedKFold(5, shuffle=True)", "scoring": "roc_auc", "grid": GRID, "class_weight": "balanced"},
    "tier1": {"features": TIER1, "best_params": bp1, "cv_auc": round(cv1, 3), "cutoffs": {"rule_out_max": lo1, "rule_in_min": hi1},
              "cutoff_rule": "from out-of-fold training predictions: rule-out keeps sensitivity >= 0.95, rule-in keeps specificity >= 0.95", "feature_importance": imp1},
    "tier2": {"features": TIER2, "best_params": bp2, "cv_auc": round(cv2, 3), "cutoffs": {"rule_out_max": lo2, "rule_in_min": hi2},
              "applies_to": "Tier 1 grey-zone patients with spleen stiffness", "feature_importance": imp2},
    "external_validation_synthetic_centre": res,
    "generator_parameters": {k: v for k, v in P.items()},
}
json.dump(tuning, open(OUT / "model_tuning_params_india.json", "w"), indent=2, default=str)

keep = ["Patient_ID", "Age", "Sex", "BMI", "PLT", "CPT_Score", "Alcohol Etiology", "Metabolic Etiology", "Viral Etiology", "Cryptogenic Etiology",
        "Autoimmune Etiology", "Other/mixed Etiology", "LS", "SS", "Elastography Modality", "Macrothrombocytopenia", "HVPG", "CSPH", "Synthetic"]
train[keep].to_csv(OUT / "synthetic_india_training.csv", index=False)
external[keep].to_csv(OUT / "synthetic_india_external_validation.csv", index=False)
print(json.dumps({"tier1": {"params": bp1, "cv_auc": round(cv1, 3), "cutoffs": [lo1, hi1]},
                  "tier2": {"params": bp2, "cv_auc": round(cv2, 3), "cutoffs": [lo2, hi2]}, "external": res}, indent=1, default=str))
