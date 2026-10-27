# India CSPH Score: a tiered, spleen-stiffness-sparing machine-learning prototype

**Research and teaching prototype for predicting clinically significant portal hypertension (CSPH, HVPG ≥10 mmHg) in compensated advanced chronic liver disease (cACLD), designed for Indian patients.**

> **SYNTHETIC DATA. RESEARCH AND TEACHING USE ONLY. NOT FOR CLINICAL DECISION-MAKING.**
>
> Every model in this package was trained on a **synthetic** India-adjusted cohort generated from published summary statistics. No real patient data were used. The models demonstrate a working pipeline; they are **not clinically validated**, and their performance numbers describe the synthetic data, not real patients.
>
> Do not use the outputs to establish or exclude CSPH, start or change beta-blocker therapy, or decide whether to perform HVPG measurement, endoscopy or any other investigation.

## What this is, and how it differs from the ELM score

This package follows the structure of the open-source ELM score (Giuffrè et al., *J Hepatol* 2026; github.com/liver-mainds/ELM-SCORE) so the two can be compared directly, but the data, inputs, model design, parameters and cut-offs are all India-specific:

| | ELM score (original) | India CSPH Score (this package) |
|---|---|---|
| Training data | 943 patients, Europe and Korea (real) | 2,400 India-adjusted patients (**synthetic**) |
| Design | One random forest | **Two tiers:** Tier 1 without spleen stiffness for everyone; Tier 2 adds spleen stiffness only for Tier 1 grey-zone patients |
| Spleen stiffness | Required by the deployed model | Optional (Tier 2 only) |
| Etiology | Viral vs other | Alcohol, metabolic, viral and **cryptogenic** as separate inputs |
| BMI | Not used (over 36% missing) | Used |
| Normalisation | `z_score_params.json` (European data) | `z_score_params_india.json` (India synthetic data) |
| Hyperparameters | 125 trees, depth 6, entropy, min leaf 3 | Tuned by 5-fold grid search: 300 trees, depth 6, entropy, min leaf 5, max_features 0.5, balanced classes |
| Cut-offs | Rule-out ≤0.45, rule-in ≥0.60 | Tier 1: ≤0.24 / ≥0.77. Tier 2: ≤0.27 / ≥0.76 (set for sensitivity and specificity ≥0.95 on out-of-fold training predictions) |

## Package contents

| File | Purpose |
|---|---|
| `build_india_model.py` | Generates the synthetic cohort, derives z-score parameters, tunes and trains both tiers, sets cut-offs, validates. Re-running it reproduces every artefact (fixed seed 2026). |
| `synthetic_india_training.csv` | 2,400 synthetic patients used for training (848 with spleen stiffness). |
| `synthetic_india_external_validation.csv` | 600 synthetic patients from a separate "centre" with a different device mix, never used in training. |
| `z_score_params_india.json` | India-specific normalisation parameters (mean and SD of LS and SS per device group, training data only, frozen). |
| `model_tuning_params_india.json` | Features, tuning grid, chosen hyperparameters, cut-offs, feature importances, external results and every generator parameter. |
| `rf_model_india_tier1.joblib` | Tier 1 random forest (no spleen stiffness). |
| `rf_model_india_tier2.joblib` | Tier 2 random forest (with spleen stiffness). |
| `india_score_inference.py` | Tiered inference script (CSV or Excel input). |
| `example_input_india.xlsx` | 16 synthetic example rows plus a legend sheet. |
| `example_input_india_predictions.csv` | Output of the script on the example file. |
| `requirements.txt` | Package versions used to build the artefacts. |

## Quick start

```bash
pip install -r requirements.txt

# Run the tiered model on the example file
python india_score_inference.py -i example_input_india.xlsx

# Rebuild everything from scratch (about 3–4 minutes on one CPU)
python build_india_model.py
```

Models were serialised with scikit-learn 1.8.0. Load them with the same version; joblib files are not guaranteed to load across scikit-learn versions. Only load model files you trust.

## Input format

Exact, case-sensitive column names:

| Column | Description | Format |
|---|---|---|
| `Age` | Age | years |
| `Sex` | Sex | 0 = female, 1 = male |
| `BMI` | Body mass index | kg/m² |
| `PLT` | Platelet count | ×10⁹/L |
| `CPT_Score` | Child-Pugh score | 5, 6 or 7 |
| `Alcohol Etiology` | Alcohol-related | 0/1 |
| `Metabolic Etiology` | MASLD / metabolic | 0/1 |
| `Viral Etiology` | HBV or HCV | 0/1 |
| `Cryptogenic Etiology` | Cryptogenic | 0/1 |
| `Autoimmune Etiology` | Autoimmune | 0/1 |
| `Other/mixed Etiology` | Other or mixed | 0/1 |
| `LS` | Liver stiffness | kPa |
| `SS` | Spleen stiffness | kPa; **may be blank** |
| `Elastography Modality` | Device | `VCTE-50Hz`, `VCTE-100Hz`, `P-SWE`, `2D-SWE` |

Exactly one etiology column should be 1 per row. An `HVPG` column may be included for evaluation; it is not used for prediction. Rows missing any Tier 1 input, or with an unrecognised modality, are skipped with a warning.

## How the tiered pathway works

1. **Tier 1** scores every patient using: age, sex, BMI, platelets, Child-Pugh, the four main etiology flags and harmonised liver stiffness.
2. Tier 1 rule-out (≤0.24) or rule-in (≥0.77) is final.
3. **Tier 2** is applied only to Tier 1 grey-zone patients who have spleen stiffness, adding harmonised spleen stiffness, with its own cut-offs (≤0.27 / ≥0.76).
4. Patients still in the grey zone are flagged: "consider SSM or HVPG" (no spleen stiffness) or "consider HVPG" (after Tier 2).

## Outputs

| Field | Meaning |
|---|---|
| `Manufacturer_normalized` | Device group used for normalisation |
| `LS_zscore`, `SS_zscore` | Harmonised stiffness values |
| `Tier1_Score`, `Tier1_Category` | Tier 1 probability and category |
| `Tier2_Score` | Tier 2 probability (grey-zone patients with SSM only) |
| `Final_Category` | `Rule-Out`, `Rule-In` or `Gray Zone` after the pathway |
| `Pathway` | Where the patient was resolved, or what test is suggested next |

## The synthetic India cohort

Each synthetic patient has an HVPG value first; liver stiffness, spleen stiffness and platelets are then derived from it with noise, so the variables are correlated with each other as they would be in real patients (an improvement on the earlier generator, which drew them independently). CSPH is HVPG ≥10 mmHg.

| Variable | Value used | Source type | Source |
|---|---|---|---|
| Sex | 81% male | Indian | Kumar 2017 (Delhi, n = 326) |
| Age | mean 52, SD 11 | Indian + assumption | Kumar 2017 |
| BMI | median 23 (+2.5 in metabolic disease) | Indian + assumption | Kumar 2017 |
| Etiology | metabolic 30%, cryptogenic 15%, alcohol 34%, viral 15%, autoimmune 2%, other 4% | Indian | Kumar 2017 (NASH/cryptogenic 45%, alcohol 34%, viral 15%, other 6%) |
| HVPG / CSPH | median 11.5 mmHg; 62% CSPH | International + assumption | ELM cACLD cohort (63% CSPH) |
| Liver stiffness | weaker coupling to HVPG than European data; higher in alcohol and metabolic disease | Indian + assumption | Kumar 2017 (rho 0.361, AUROC 0.74) |
| Spleen stiffness | tighter coupling to HVPG than liver stiffness; available in 35% | International + assumption | ELM 2026; proposal premise |
| Platelets | ELM levels × 0.82; 4% constitutional low counts | Indian + international | Indian platelet reference 2022; macrothrombocytopenia study 2025; ELM 2026 |
| Child-Pugh | 5/6/7 rising with severity | International + assumption | ELM 2026 |
| Devices | VCTE 70%, 2D-SWE 18%, p-SWE 12% (validation centre: more 2D-SWE) | Assumption | — |

All generator parameters are stored in `model_tuning_params_india.json`.

## Results on the synthetic external centre (n = 600)

| Approach | Grey zone | NPV | PPV | AUC |
|---|---:|---:|---:|---:|
| Baveno VII (LSM ≤15 + PLT ≥150 / LSM ≥25) | 48.5% | 0.93 | 0.90 | — |
| India Tier 1 alone (no spleen stiffness) | 27.3% | 0.89 | 0.96 | 0.94 |
| India tiered pathway (Tier 1, then Tier 2 if SSM available) | 22.5% | 0.89 | 0.96 | — |
| India Tier 2, patients with SSM (n = 226) | 24.8% | 0.92 | 0.99 | 0.95 |
| Original ELM model with published cut-offs, same patients with SSM (n = 226) | 10.2% | 0.91 | **0.85** | 0.93 |

### How to read these numbers

- **They describe synthetic data generated from our own assumptions.** They show the pipeline works and illustrate the questions a real study would ask; they say nothing about real-world accuracy.
- **Tier 1's NPV (0.89) falls just below the 0.90 target.** In a real study the rule-out cut-off would be tightened, accepting a larger grey zone.
- **The original ELM model's PPV fell to 0.85 on these Indian-adjusted patients**, below the 0.90 target, mainly because lower Indian platelet counts push patients toward rule-in. This is the kind of miscalibration the project aims to test with real data; here it partly reflects our generator's assumptions.
- **Feature importance reflects the generator, not biology.** Tier 1 relies most on platelets (0.57), then liver stiffness (0.32), because the generator assumes a weak liver stiffness–HVPG link (from the mostly decompensated Delhi cohort). In Tier 2, spleen stiffness ranks first (0.43), consistent with ELM. Real data could rank them differently.

## Limitations

- Synthetic data only; no real patients, no external clinical validation.
- The Delhi cohort used for calibration was mostly decompensated, not cACLD.
- Device offsets, spleen stiffness availability and the HVPG distribution are assumptions.
- Regional variation within India (etiology, platelet norms) is not modelled.
- Not applicable to non-cirrhotic portal hypertension, decompensated disease or acute illness.
- Any real use needs ethics approval, real Indian paired HVPG data, prospective validation and regulatory review.

## Next step with real data

Replace the synthetic files with a real, governed Indian cohort in the same column format (including `HVPG`), then re-run `build_india_model.py` with the generator section removed. The same code will derive new z-score parameters, re-tune both tiers, set new cut-offs and benchmark against Baveno VII and the original ELM model.

## References

1. Giuffrè M, et al. Validation of a pan-ELastography Machine-learning (ELM) score to predict clinically significant portal hypertension in compensated advanced chronic liver disease. *J Hepatol*. 2026;85:959–972. Code: github.com/liver-mainds/ELM-SCORE
2. Baveno VIII Faculty. Baveno VIII – Advancing consensus in portal hypertension. *J Hepatol*. 2026;85:993–1018.
3. Kumar A, et al. Correlation of transient elastography with hepatic venous pressure gradient in patients with cirrhotic portal hypertension: a study of 326 patients from India. *World J Gastroenterol*. 2017;23:687–696.
4. Reference interval of platelet counts and other platelet indices in apparently healthy blood donors in North India. *Asian J Transfus Sci*. 2022.
5. Prevalence of asymptomatic macrothrombocytopenia in voluntary blood donors: a multicentric study. *Asian J Transfus Sci*. 2025.
6. Semmler G, Jachs M, Mandorfer M. Non-invasive tests-based risk stratification: Baveno VII and beyond. *Clin Mol Hepatol*. 2023;29:105–109.
