// India CSPH Score: in-browser scoring engine.
// Mirrors scikit-learn RandomForestClassifier.predict_proba (mean of per-tree leaf probabilities;
// inputs compared as float32, as scikit-learn trees do). Feature contributions use the
// tree-path decomposition: root probability + sum of probability changes along each split.

const f32 = Math.fround;

export function scoreForest(forest, row) {
  const x = forest.features.map((name) => row[name]);
  const contrib = Object.fromEntries(forest.features.map((n) => [n, 0]));
  let total = 0, bias = 0;
  for (const tr of forest.trees) {
    let node = 0;
    bias += tr.p[0];
    while (tr.l[node] !== -1) {
      const fi = tr.f[node];
      const next = f32(x[fi]) <= tr.t[node] ? tr.l[node] : tr.r[node];
      contrib[forest.features[fi]] += tr.p[next] - tr.p[node];
      node = next;
    }
    total += tr.p[node];
  }
  const n = forest.trees.length;
  for (const k in contrib) contrib[k] /= n;
  return { prob: total / n, bias: bias / n, contrib };
}

export function category(p, c) {
  if (p <= c.rule_out_max) return "Rule-out";
  if (p >= c.rule_in_min) return "Rule-in";
  return "Grey zone";
}

// Shear-wave speed (m/s) to kPa, the same conversion the ELM code uses: E = 3 * 1050 kg/m3 * v^2.
export const msToKpa = (v) => (3 * 1050 * v * v) / 1000;

export function zscore(meta, modality, ls, ss) {
  const p = meta.z_score_params[meta.group[modality]];
  return {
    LS_zscore: (ls - p.LS.mean) / p.LS.std,
    SS_zscore: ss == null || Number.isNaN(ss) ? null : (ss - p.SS.mean) / p.SS.std,
  };
}

export function percentile(dist, v) {
  const q = dist.quantiles;
  if (v <= q[0]) return 0;
  if (v >= q[q.length - 1]) return 100;
  for (let i = 1; i < q.length; i++) if (v <= q[i]) return i - 1 + (v - q[i - 1]) / Math.max(1e-9, q[i] - q[i - 1]);
  return 100;
}

// Full tiered pathway for one case.
export function runPathway(meta, tier1, tier2, input) {
  const z = zscore(meta, input.modality, input.LS, input.SS);
  const etio = input.etiology;
  const row = {
    Age: input.Age, Sex: input.Sex, BMI: input.BMI, PLT: input.PLT, CPT_Score: input.CPT,
    "Alcohol Etiology": +(etio === "Alcohol"), "Metabolic Etiology": +(etio === "Metabolic"),
    "Viral Etiology": +(etio === "Viral"), "Cryptogenic Etiology": +(etio === "Cryptogenic"),
    LS_zscore: z.LS_zscore, SS_zscore: z.SS_zscore,
  };
  const t1 = scoreForest(tier1, row);
  const c1 = category(t1.prob, meta.tier1.cutoffs);
  const out = { row, tier1: { ...t1, category: c1 }, tier2: null, final: c1, resolvedAt: "Tier 1" };
  if (c1 === "Grey zone") {
    if (z.SS_zscore != null) {
      const t2 = scoreForest(tier2, row);
      const c2 = category(t2.prob, meta.tier2.cutoffs);
      out.tier2 = { ...t2, category: c2 };
      out.final = c2;
      out.resolvedAt = c2 === "Grey zone" ? "Unresolved after Tier 2" : "Tier 2";
    } else {
      out.resolvedAt = "Unresolved, no spleen stiffness";
    }
  }
  // Out-of-distribution check, as on the ELM site: flag when 2+ of LS, SS, PLT fall outside the 5th-95th percentile.
  const d = meta.distributions;
  const pct = { LS_zscore: percentile(d.LS_zscore, z.LS_zscore), PLT: percentile(d.PLT, input.PLT) };
  if (z.SS_zscore != null) pct.SS_zscore = percentile(d.SS_zscore, z.SS_zscore);
  const outside = Object.values(pct).filter((p) => p < 5 || p > 95).length;
  out.percentiles = pct;
  out.outOfDistribution = outside >= 2;
  return out;
}
