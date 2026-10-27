import { runPathway, msToKpa } from "./model.js";

const $ = (id) => document.getElementById(id);
const NAMES = {
  Age: "Age", Sex: "Sex", BMI: "BMI", PLT: "Platelets", CPT_Score: "Child-Pugh",
  "Alcohol Etiology": "Alcohol-related", "Metabolic Etiology": "Metabolic", "Viral Etiology": "Viral",
  "Cryptogenic Etiology": "Cryptogenic", LS_zscore: "Liver stiffness", SS_zscore: "Spleen stiffness",
};
let META, T1, T2;

// ---------- research-use gate ----------
const gate = $("gate");
gate.showModal();
$("gate-ok").addEventListener("change", (e) => ($("gate-accept").disabled = !e.target.checked));
$("gate-accept").addEventListener("click", () => gate.close());
gate.addEventListener("cancel", (e) => e.preventDefault());

// ---------- load model ----------
Promise.all(["meta", "tier1", "tier2"].map((f) => fetch(`model/${f}.json`).then((r) => r.json())))
  .then(([m, a, b]) => { META = m; T1 = a; T2 = b; fillTables(); })
  .catch(() => { $("form-error").textContent = "The model files could not be loaded. Refresh the page to try again."; });

// ---------- linked sliders ----------
for (const id of ["plt", "age"]) {
  const r = $(`${id}-range`), n = $(id);
  r.addEventListener("input", () => (n.value = r.value));
  n.addEventListener("input", () => (r.value = n.value));
}

// ---------- form ----------
$("case-form").addEventListener("submit", (e) => {
  e.preventDefault();
  if (!META) return;
  const err = (msg) => { $("form-error").textContent = msg; };
  const modality = $("modality").value, unit = $("unit").value;
  const num = (id) => ($(id).value === "" ? null : Number($(id).value));
  let ls = num("ls"), ss = num("ss");
  const plt = num("plt"), age = num("age"), bmi = num("bmi"), etiology = $("etiology").value;
  if (!modality) return err("Choose the elastography device.");
  if (unit === "ms" && modality.startsWith("VCTE")) return err("FibroScan reports stiffness in kPa. Switch the unit to kPa.");
  if (ls == null) return err("Enter the liver stiffness.");
  if (unit === "ms") {
    if (ls < 0.5 || ls > 5.5 || (ss != null && (ss < 0.5 || ss > 6))) return err("Values in m/s are usually between 0.5 and 5.5. Check the unit.");
    ls = msToKpa(ls); if (ss != null) ss = msToKpa(ss);
  }
  if (ls < 2 || ls > 75) return err("Liver stiffness must be between 2 and 75 kPa.");
  if (ss != null && (ss < 5 || ss > 100)) return err("Spleen stiffness must be between 5 and 100 kPa, or left blank.");
  if (plt == null || plt < 20 || plt > 450) return err("Platelets must be between 20 and 450 ×10⁹/L.");
  if (!etiology) return err("Choose the cause of liver disease.");
  if (age == null || age < 18 || age > 90) return err("Age must be between 18 and 90.");
  if (bmi == null || bmi < 12 || bmi > 60) return err("BMI must be between 12 and 60.");
  err("");
  const input = { modality, LS: ls, SS: ss, PLT: plt, Age: age, BMI: bmi, CPT: Number($("cpt").value),
                  Sex: Number(document.querySelector("input[name=sex]:checked").value), etiology };
  render(runPathway(META, T1, T2, input), input);
});

// ---------- result ----------
const fmt = (p) => p.toFixed(2);
function render(r, input) {
  $("result-empty").hidden = true; $("result-body").hidden = false; $("dashboard").hidden = false;
  const usedTier2 = r.tier2 != null;
  const used = usedTier2 ? r.tier2 : r.tier1, cuts = usedTier2 ? META.tier2.cutoffs : META.tier1.cutoffs;
  const cls = { "Rule-out": "out", "Grey zone": "grey", "Rule-in": "in" }[r.final];
  const texts = {
    "Rule-out": "Score at or below the rule-out cut-off: lower modelled probability of CSPH.",
    "Rule-in": "Score at or above the rule-in cut-off: higher modelled probability of CSPH.",
    "Grey zone": r.tier2 ? "Still indeterminate after spleen stiffness. In a study, this case would go to HVPG."
                         : "Indeterminate from routine data. Adding spleen stiffness could resolve this case.",
  };
  $("verdict").className = `verdict ${cls}`;
  $("verdict-label").textContent = r.final;
  $("verdict-text").textContent = texts[r.final];

  $("track-title").textContent = usedTier2 ? "Tier 2 score (with spleen stiffness)" : "Tier 1 score";
  $("track-score").textContent = fmt(used.prob);
  const [lo, hi] = [cuts.rule_out_max, cuts.rule_in_min];
  const bands = $("track").querySelectorAll(".band");
  bands[0].style.width = `${lo * 100}%`; bands[1].style.width = `${(hi - lo) * 100}%`; bands[2].style.width = `${(1 - hi) * 100}%`;
  $("marker").style.left = `${used.prob * 100}%`;
  $("cut-lo").textContent = fmt(lo); $("cut-lo").style.left = `${lo * 100}%`;
  $("cut-hi").textContent = fmt(hi); $("cut-hi").style.left = `${hi * 100}%`;

  const s2 = r.tier2 ? ["active", `${fmt(r.tier2.prob)} · ${r.tier2.category}`, "Added spleen stiffness"]
           : r.tier1.category !== "Grey zone" ? ["skipped", "Not needed", "Tier 1 resolved the case"]
           : ["skipped", "No spleen stiffness", "Enter spleen stiffness to run Tier 2"];
  const needHvpg = r.final === "Grey zone" && r.tier2;
  $("pathway").innerHTML = `
    <li class="active"><span class="step-name">Tier 1<span class="step-note">Routine data, no spleen stiffness</span></span><span class="step-val">${fmt(r.tier1.prob)} · ${r.tier1.category}</span></li>
    <li class="${s2[0]}"><span class="step-name">Tier 2<span class="step-note">${s2[2]}</span></span><span class="step-val">${s2[1]}</span></li>
    <li class="${needHvpg ? "active" : "skipped"}"><span class="step-name">HVPG<span class="step-note">Reserved for cases still unresolved</span></span><span class="step-val">${needHvpg ? "Would be considered" : "Not needed"}</span></li>`;
  $("ood").hidden = !r.outOfDistribution;

  const d = META.distributions;
  density($("plot-ls"), d.LS_zscore, r.row.LS_zscore); $("pct-ls").textContent = `This case: ${ord(r.percentiles.LS_zscore)} percentile of training patients`;
  if (r.row.SS_zscore != null) { density($("plot-ss"), d.SS_zscore, r.row.SS_zscore); $("pct-ss").textContent = `This case: ${ord(r.percentiles.SS_zscore)} percentile`; }
  else { density($("plot-ss"), d.SS_zscore, null); $("pct-ss").textContent = "Not measured for this case"; }
  density($("plot-plt"), d.PLT, input.PLT); $("pct-plt").textContent = `This case: ${ord(r.percentiles.PLT)} percentile`;
  $("contrib-title").textContent = `Contribution of each input to the ${usedTier2 ? "Tier 2" : "Tier 1"} score`;
  waterfall($("plot-contrib"), used, r.row);
}
const ord = (p) => { const n = Math.round(p); const s = n % 10 === 1 && n !== 11 ? "st" : n % 10 === 2 && n !== 12 ? "nd" : n % 10 === 3 && n !== 13 ? "rd" : "th"; return n + s; };

// ---------- charts ----------
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs, text) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text != null) e.textContent = text; return e; };

function density(svg, dist, value) {
  svg.innerHTML = "";
  const g = dist.grid, y = dist.density, x0 = 12, x1 = 288, yb = 112, yt = 14;
  const ymax = Math.max(...y), sx = (v) => x0 + ((v - g[0]) / (g[g.length - 1] - g[0])) * (x1 - x0), sy = (v) => yb - (v / ymax) * (yb - yt);
  const q5 = dist.quantiles[5], q95 = dist.quantiles[95];
  svg.append(el("rect", { x: sx(q5), y: yt, width: sx(q95) - sx(q5), height: yb - yt, fill: "#EDF2F1" }));
  let d = `M${sx(g[0])},${yb}`; g.forEach((v, i) => (d += ` L${sx(v)},${sy(y[i])}`)); d += ` L${sx(g[g.length - 1])},${yb} Z`;
  svg.append(el("path", { d, fill: "#9CC3BC", "fill-opacity": 0.55, stroke: "#0F3B3A", "stroke-width": 1.5 }));
  svg.append(el("line", { x1: x0, x2: x1, y1: yb, y2: yb, stroke: "#B8C4C2" }));
  svg.append(el("text", { x: x0, y: 130, "font-size": 10, fill: "#4A5B5A" }, g[0].toFixed(0)));
  svg.append(el("text", { x: x1, y: 130, "font-size": 10, fill: "#4A5B5A", "text-anchor": "end" }, g[g.length - 1].toFixed(0)));
  svg.append(el("text", { x: (sx(q5) + sx(q95)) / 2, y: 126, "font-size": 9, fill: "#4A5B5A", "text-anchor": "middle" }, "shaded: 5th–95th percentile"));
  if (value != null) {
    const xv = Math.max(x0, Math.min(x1, sx(value)));
    svg.append(el("line", { x1: xv, x2: xv, y1: yt - 4, y2: yb, stroke: "#E3A019", "stroke-width": 3 }));
    svg.append(el("circle", { cx: xv, cy: yt - 4, r: 5, fill: "#E3A019", stroke: "#0F3B3A", "stroke-width": 1.5 }));
  }
}

function waterfall(svg, res, row) {
  svg.innerHTML = "";
  const items = Object.entries(res.contrib).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const L = 230, R = 610, top = 16, h = 25, sx = (p) => L + p * (R - L);
  const rows = items.length + 2;
  svg.setAttribute("viewBox", `0 0 640 ${top + rows * h + 30}`);
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    svg.append(el("line", { x1: sx(t), x2: sx(t), y1: top - 6, y2: top + rows * h, stroke: "#E3E9E8" }));
    svg.append(el("text", { x: sx(t), y: top + rows * h + 16, "font-size": 11, fill: "#4A5B5A", "text-anchor": "middle" }, t.toFixed(2)));
  }
  const label = (i, txt, val) => {
    svg.append(el("text", { x: 8, y: top + i * h + 16, "font-size": 12.5, fill: "#0F3B3A", "font-weight": 600 }, txt));
    if (val != null) svg.append(el("text", { x: L - 10, y: top + i * h + 16, "font-size": 11.5, fill: "#4A5B5A", "text-anchor": "end" }, val));
  };
  label(0, "Average training score", null);
  svg.append(el("rect", { x: sx(0), y: top + 5, width: sx(res.bias) - sx(0), height: h - 10, fill: "#C9D5D3", rx: 3 }));
  svg.append(el("text", { x: sx(res.bias) + 6, y: top + 16, "font-size": 11.5, fill: "#0F3B3A" }, res.bias.toFixed(2)));
  let cum = res.bias;
  items.forEach(([k, c], i) => {
    const y = top + (i + 1) * h, a = Math.min(cum, cum + c), b = Math.max(cum, cum + c);
    label(i + 1, NAMES[k] || k, showVal(k, row[k]));
    svg.append(el("rect", { x: sx(a), y: y + 5, width: Math.max(1.5, sx(b) - sx(a)), height: h - 10, fill: c >= 0 ? "#C4483F" : "#2F6FA8", rx: 3 }));
    svg.append(el("text", { x: sx(b) + 6, y: y + 16, "font-size": 11, fill: "#4A5B5A" }, (c >= 0 ? "+" : "") + c.toFixed(3)));
    cum += c;
  });
  const yf = top + (items.length + 1) * h;
  label(items.length + 1, "This case's score", null);
  svg.append(el("rect", { x: sx(0), y: yf + 5, width: sx(res.prob) - sx(0), height: h - 10, fill: "#0F3B3A", rx: 3 }));
  svg.append(el("text", { x: sx(res.prob) + 6, y: yf + 16, "font-size": 12, fill: "#0F3B3A", "font-weight": 700 }, res.prob.toFixed(2)));
}
function showVal(k, v) {
  if (v == null) return "";
  if (k === "Sex") return v ? "male" : "female";
  if (k.endsWith("Etiology")) return v ? "yes" : "no";
  if (k.endsWith("zscore")) return `z = ${v.toFixed(2)}`;
  return String(Math.round(v * 10) / 10);
}

// ---------- evidence tables ----------
function fillTables() {
  const c = $("cutoff-table").querySelector("tbody");
  for (const [name, t] of [["Tier 1 (no spleen stiffness)", META.tier1], ["Tier 2 (with spleen stiffness)", META.tier2]]) {
    const { rule_out_max: lo, rule_in_min: hi } = t.cutoffs;
    c.insertAdjacentHTML("beforeend", `<tr><td>${name}</td><td>≤ ${lo.toFixed(2)}</td><td>${lo.toFixed(2)} to ${hi.toFixed(2)}</td><td>≥ ${hi.toFixed(2)}</td><td>${t.cv_auc.toFixed(2)}</td></tr>`);
  }
  const rt = $("results-table").querySelector("tbody");
  for (const [name, v] of Object.entries(META.external)) {
    const f = (x) => (x == null ? "–" : x.toFixed(2));
    rt.insertAdjacentHTML("beforeend", `<tr><td>${name}</td><td>${v.n}</td><td>${v.grey_zone_pct.toFixed(1)}%</td><td>${f(v.NPV)}</td><td>${f(v.PPV)}</td><td>${f(v.AUC)}</td></tr>`);
  }
}
