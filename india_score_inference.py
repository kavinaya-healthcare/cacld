"""
India CSPH Score: tiered inference script
=========================================
Tier 1 (no spleen stiffness) for every patient; Tier 2 (adds spleen stiffness)
only for Tier 1 grey-zone patients who have an SSM value.

RESEARCH / TEACHING USE ONLY. The bundled models were trained on SYNTHETIC
India-adjusted data and are not clinically validated.

Usage:
  python india_score_inference.py -i example_input_india.xlsx
  python india_score_inference.py -i data.csv -p z_score_params_india.json -t model_tuning_params_india.json \
      --tier1 rf_model_india_tier1.joblib --tier2 rf_model_india_tier2.joblib -o results.csv
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, joblib


class IndiaCSPHInference:
    REQUIRED_COLUMNS = ["Age", "Sex", "BMI", "PLT", "CPT_Score", "Alcohol Etiology", "Metabolic Etiology", "Viral Etiology",
                        "Cryptogenic Etiology", "Autoimmune Etiology", "Other/mixed Etiology", "LS", "SS", "Elastography Modality"]
    MANUFACTURER_MAP = {"VCTE-50Hz": "VCTE", "VCTE-100Hz": "VCTE", "P-SWE": "P-SWE", "2D-SWE": "2D-SWE"}

    def __init__(self, z_params, tuning, tier1, tier2):
        self.zp = json.load(open(z_params))
        self.cfg = json.load(open(tuning))
        self.m1, self.m2 = joblib.load(tier1), joblib.load(tier2)
        self.f1, self.f2 = self.cfg["tier1"]["features"], self.cfg["tier2"]["features"]
        self.c1, self.c2 = self.cfg["tier1"]["cutoffs"], self.cfg["tier2"]["cutoffs"]
        print("✓ Loaded India z-score parameters, tuning file and Tier 1/Tier 2 models")

    def load(self, path):
        p = Path(path)
        if not p.exists(): sys.exit(f"✗ Input file not found: {p}")
        df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)
        miss = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if miss: sys.exit("✗ Missing required columns (case-sensitive): " + ", ".join(miss))
        bad = ~df["Elastography Modality"].isin(self.MANUFACTURER_MAP)
        if bad.any(): print(f"⚠ Skipping {bad.sum()} rows with unrecognised modality")
        df = df[~bad].copy()
        nan_core = df[[c for c in self.REQUIRED_COLUMNS if c != "SS"]].isna().any(axis=1)
        if nan_core.any(): print(f"⚠ Skipping {nan_core.sum()} rows with missing Tier 1 inputs"); df = df[~nan_core].copy()
        print(f"✓ {len(df)} rows ready ({df['SS'].notna().sum()} with spleen stiffness)")
        return df

    def zscores(self, df):
        g = df["Elastography Modality"].map(self.MANUFACTURER_MAP)
        df["Manufacturer_normalized"] = g
        df["LS_zscore"] = [(v - self.zp[k]["LS"]["mean"]) / self.zp[k]["LS"]["std"] for v, k in zip(df["LS"], g)]
        df["SS_zscore"] = [(v - self.zp[k]["SS"]["mean"]) / self.zp[k]["SS"]["std"] if pd.notna(v) else np.nan for v, k in zip(df["SS"], g)]
        return df

    @staticmethod
    def classify(p, c):
        return np.where(p <= c["rule_out_max"], "Rule-Out", np.where(p >= c["rule_in_min"], "Rule-In", "Gray Zone"))

    def run(self, input_file, output_file=None):
        df = self.zscores(self.load(input_file))
        df["Tier1_Score"] = self.m1.predict_proba(df[self.f1])[:, 1]
        df["Tier1_Category"] = self.classify(df["Tier1_Score"].values, self.c1)
        df["Tier2_Score"] = np.nan
        need = (df["Tier1_Category"] == "Gray Zone") & df["SS"].notna()
        if need.any(): df.loc[need, "Tier2_Score"] = self.m2.predict_proba(df.loc[need, self.f2])[:, 1]
        final = df["Tier1_Category"].copy()
        final[need] = self.classify(df.loc[need, "Tier2_Score"].values, self.c2)
        df["Final_Category"] = final
        df["Pathway"] = np.where(df["Tier1_Category"] != "Gray Zone", "Resolved at Tier 1",
                         np.where(need, np.where(final == "Gray Zone", "Gray zone after Tier 2: consider HVPG", "Resolved at Tier 2"),
                                  "Gray zone, no SSM: consider SSM or HVPG"))
        out = output_file or Path(input_file).with_name(Path(input_file).stem + "_india_predictions.csv")
        df.to_csv(out, index=False)
        print("\nFinal categories:"); print(df["Final_Category"].value_counts().to_string())
        print("\nPathway:"); print(df["Pathway"].value_counts().to_string())
        print(f"\n✓ Results saved to: {out}")
        return df


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="India CSPH Score (research prototype, synthetic training data)")
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-p", "--params", default=str(here / "z_score_params_india.json"))
    ap.add_argument("-t", "--tuning", default=str(here / "model_tuning_params_india.json"))
    ap.add_argument("--tier1", default=str(here / "rf_model_india_tier1.joblib"))
    ap.add_argument("--tier2", default=str(here / "rf_model_india_tier2.joblib"))
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args()
    IndiaCSPHInference(a.params, a.tuning, a.tier1, a.tier2).run(a.input, a.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
