# Project Completion Plan — Mechanistic Interpretability of AI Safety Guardrails

> **Course:** MA-INF 4330 — Lab Explainable AI and Applications, University of Bonn
> **Student:** Pranav Yadav · **Repo:** `safety-circuits`
> **Last updated:** 2026-05-30
>
> Living tracker for finishing the project. Companion to `RESEARCH_PLAN.md` (hypotheses/method) and `FINDINGS.md` (results so far). Check off items as they land.

---

## TL;DR

- **Code/experiments: ~80% done and strong.** The TransformerLens patching + ablation pipeline works and the core result (sparse, causal, ablatable "safety heads") is real across 4 models.
- **Graded deliverables: ~10% done.** The grade is **70% paper + 30% defence**. There is **no paper, no midterm slides, no final slides** yet. **Protect writing time above all** — experiments are inputs, the paper is the deliverable.
- **Must-fix before the paper:** perplexity control (G1), Llama K-sweep (G2), metric audit (G5), statistical rigor (G10).
- **Publication-grade additions:** Phi-3 fix + pending models (G3/G4), jailbreak stress test (G9), last-token + mean-ablation comparisons (G6/G7).

### Locked decisions
- **Model set:** fix Phi-3 **and** run pending models (Falcon3-1B, OLMo-2-1B) → target 6–7 valid models for H4.
- **Ambition:** publication-grade (include stretch experiments).
- **Paper:** ~8-page conference/workshop style.

---

## Deadlines (from `XAI_Orga_SS202.pdf`)

| Date | Milestone | Format | Weight |
|---|---|---|---|
| ~~17/04/2026~~ | Organisation | — | done |
| ~~24/04/2026~~ | Pitch | 5 min + 5 Q&A | done (`XAI_Lab_v1.pptx`) |
| **19/06/2026** | **Midterm presentation** | 15 min + 5 Q&A | part of 30% |
| **24/07/2026** | **Final presentation** (in person, Friedrich-Hirzebruch-Allee 5, Rm 0.108) | 15 min + 5 Q&A | 30% |
| **31/08/2026 23:59 AOE** | **Paper submission** → `amllab@bit.uni-bonn.de` | ~8-page paper | 70% |

---

## Part A — Status: what is DONE

### A1. Infrastructure & code (≈90%)
| Component | File | Status |
|---|---|---|
| Model configs (8 models, refusal tokens, dtype, RoPE flags) | `src/safety_circuits/config.py` | ✅ |
| Model loading (TL-native + HF fallback port) | `src/safety_circuits/models.py` | ✅ |
| Data loaders: AdvBench, HarmBench, RTP, HH-RLHF | `src/safety_circuits/data.py` | ✅ |
| Matched-pair construction | `data.py:build_matched_pairs` | ✅ |
| Refusal metric: logit-margin + regex | `src/safety_circuits/refusal.py` | ✅ |
| Activation caching | `src/safety_circuits/activations.py` | ✅ |
| Patching: head-`z`, MLP-output, residual-stream | `src/safety_circuits/patching.py` | ✅ |
| Ablation: zero **and** mean hooks; `perplexity()` fn | `src/safety_circuits/ablation.py` | ⚠️ (perplexity not wired in) |
| Aggregation + heatmap plotting | `src/safety_circuits/analysis.py` | ✅ |
| CLI `run-mvp` | `src/safety_circuits/cli.py` | ✅ |
| Kaggle runner | `kaggle/run_experiment.py` | ✅ (zero-ablation only) |
| Docker / Makefile / pyproject / smoke tests | repo root + `tests/` | ✅ |
| Notebooks 01–05 | `notebooks/` | ✅ |

### A2. Experiments run (see `FINDINGS.md`, artifacts in `results/kaggle/`)
| Model | Arch | Result | Status |
|---|---|---|---|
| Qwen2.5-1.5B | 28L×12H | 100% → **0%** refusal (top-10) | ✅ |
| Qwen3-1.7B | 28L×16H | 93.75% → **0%** | ✅ |
| Gemma-3-1B | 26L×4H | 68.75% → **0%** | ✅ |
| Llama-3.2-3B | 28L×24H | 93.75% → **75%** (incomplete) | ⚠️ |
| Phi-3-mini | 32L×32H | 0% baseline → inconclusive (broken port) | ❌ |
| Falcon3-1B / OLMo-2-1B | — | configured, not run | ⬜ |

### A3. Hypothesis scorecard (current)
- **H2 (causal):** ✅ confirmed in 4/4 valid models.
- **H1 (sparse):** 🟡 partial — sparse everywhere; full suppression only Qwen/Gemma.
- **H3 (ablation):** 🟡 partial — Llama stalls at 75%; **perplexity control never measured**.
- **H4 (cross-model):** 🟡 partial — L0 dominance in 3/4; Gemma is L24. Needs more models.

---

## Part B — Gap analysis: what is LEFT

### B1. Experimental / scientific gaps
| # | Gap | Why it matters | Where | Effort | Done |
|---|---|---|---|---|---|
| G1 | **Perplexity / capability check never measured** | H3 success criterion (≤5% Δ). `perplexity()` exists but `evaluate_ablation` hardcoded `None`. Highest-priority. | `ablation.py`, `data.py`, `cli.py`, `run_experiment.py` | M | 🟡 code done — Kaggle re-runs pending |
| G2 | **Llama ablation incomplete (75%)** | K-sweep (10→40) to find threshold or document distributed circuit. | `run_experiment.py` (`SC_TOP_K`) | S | ☐ |
| G3 | **Phi-3 broken (0% baseline)** | Correct combined-QKV/RoPE TL port, or documented swap. | `models.py:_load_via_hf_port` | M–L | ☐ |
| G4 | **Pending models not run** (Falcon3-1B, OLMo-2-1B) | More models → stronger H4; OLMo-2 fully-open-data. | `config.py` | S each | ☐ |
| G5 | **No refusal-metric validation** | ≥90% agreement vs human labels on 50-prompt audit. | new `notebooks/06_metric_audit.ipynb` | M | ☐ |
| G6 | **Mean-ablation never run** | Wang et al. confound control; code exists, unused. | `ablation.py` (`mode="mean"`) | S | ☐ |
| G7 | **Last-token-only patching not used** | Reviewers ask if effect concentrates at decision token. | `patching.py:patch_each_head` | S | ☐ |
| G8 | **Attention-pattern patching not implemented** | Pattern vs value-weighted output. Optional. | new fn in `patching.py` | M | ☐ |
| G9 | **HarmBench jailbreak stress test not run** | Do safety heads still fire under jailbreaks? | `data.load_harmbench` | M | ☐ |
| G10 | **No statistical rigor** | Error bars / CIs / fixed seed across pairs. | `analysis.py` | M | ☐ |
| G11 | **Residual-trace "flat" artifact** | Frame as sanity check, not a finding. | `FINDINGS.md`, paper | trivial | ☐ |

### B2. Deliverable gaps (THE GRADE)
| # | Gap | Weight | Done |
|---|---|---|---|
| D1 | **Scientific paper (~8 pages)** | **70%** | ☐ |
| D2 | **Final presentation slides** (15 min) | **30%** | ☐ |
| D3 | **Midterm presentation slides** (15 min, 19/06) | part of 30% | ☐ |
| D4 | Reproducibility polish (pinned deps, seed, one-command repro, results archived, README status) | supports paper | ☐ |
| D5 | Publication-quality figures | supports both | ☐ |

---

## Part C — Phased plan

### Phase 0 — Setup & tracker (this week)
- [x] Create this `PROJECT_PLAN.md`.
- [x] Update `README.md` "Status" (no longer just a scaffold); also fixed the stale CLI quick-start (`safety_circuits.cli run-mvp`) and model references.
- [x] Write the proposal-vs-actual model-swap rationale (added to `README.md` Status note; proposal said TinyLlama/Phi-3 → moved to Qwen/Gemma/Llama, anticipated in `RESEARCH_PLAN.md` risk table).
- [ ] Confirm paper length/format on Discord / Thursday office hours (2–3pm). _(your action)_

### Phase 1 — Must-have experiments → unblock midterm (now → 18/06)
1. **G1** Wire `perplexity()` into `evaluate_ablation`; populate CSV columns; re-run on the 4 valid models.
   - [x] `load_wikitext2()` added (`data.py`) — WikiText-2 test-split snippets for the control.
   - [x] `perplexity()` accepts `fwd_hooks` → measures perplexity *under ablation* (`ablation.py`).
   - [x] `evaluate_ablation(..., perplexity_texts=)` computes clean + ablated; `AblationReport.perplexity_pct_change` property (H3 ≤5% target).
   - [x] Wired into Kaggle runner (`run_experiment.py`, `SC_PPL_TEXTS` env, saves `perplexity_pct_change` column) and CLI (`--ppl_texts`).
   - [x] Model-free tests pass; all files compile.
   - [ ] **Re-run on Kaggle** for the 4 valid models → fill perplexity columns in each `*_ablation.csv`, update `FINDINGS.md`. _(GPU run)_
2. [ ] **G2** Llama K-sweep `SC_TOP_K ∈ {10,15,20,30,40}`; plot refusal-rate vs K.
3. [ ] **G4** Run Falcon3-1B and OLMo-2-1B end-to-end (patch + ablate + perplexity).
4. [ ] **G5** Hand-label 50 prompts; compute agreement vs logit+regex (≥90% target).
5. [ ] **G10** Add per-pair variance / 95% CI to aggregated |Δmargin|; fix and report a seed.

### Phase 2 — Midterm presentation (deliver 19/06)
- [ ] 15-min deck (~12–15 slides): problem → method → working result (sparse heads, ablation collapse) → cross-model picture → next steps.
- [ ] Lead visual: clean per-head heatmap + "10/336 heads → 100%→0%" headline.
- [ ] Honest status slide (H2 confirmed; H1/H3/H4 partial; perplexity now controlled).
- [ ] Rehearse to 15 min + backup Q&A slides.

### Phase 3 — Stretch experiments → publication-grade (20/06 → ~20/07)
6. [ ] **G3** Fix Phi-3 port (or documented swap + exclusion rationale).
7. [ ] **G9** HarmBench jailbreak stress test on identified heads.
8. [ ] **G7** Last-token vs position-agnostic patching (side-by-side heatmaps).
9. [ ] **G6** Mean-ablation vs zero-ablation comparison table.
10. [ ] **G8** Attention-pattern patching (optional; cut first if time-pressed).
11. [ ] **G11** Reframe residual-trace as sanity check everywhere.

### Phase 4 — Paper writing (~20/07 → 25/08, overlaps Phase 3)
Target ~8-page conference/workshop format:
- [ ] Abstract + Intro ("Is safety sparse and localizable in small instruct LMs?")
- [ ] Related work (IOI / Wang 2022, ROME causal tracing / Meng, activation patching, refusal-direction work)
- [ ] Method (matched pairs, refusal-margin metric + audit, patching head/MLP/resid + last-token, ablation zero & mean, perplexity control)
- [ ] Results (heatmaps, sparsity/top-K mass, ablation collapse + perplexity, K-sweep, cross-model table, jailbreak)
- [ ] Discussion (L0 vs L24, distributed vs concentrated, "safety is mechanistic")
- [ ] Limitations (small models, toxic axis only, no weight editing/SAEs, port caveats)
- [ ] Reproducibility appendix (models, seeds, deps, one-command repro)
- [ ] **D5** figures: heatmaps, cross-model comparison, refusal-vs-K curve, zero-vs-mean bars, perplexity-vs-refusal scatter, jailbreak deltas
- Cadence: outline → method+results first → intro/related last → 2 self-reviews → 1 peer/office-hours review.

### Phase 5 — Final presentation (deliver 24/07, in person)
- [ ] 15-min deck = paper's narrative arc with publication-grade figures, H4 verdict, jailbreak result.

### Phase 6 — Submit (by 31/08 23:59 AOE)
- [ ] Proofread, figure/caption check, references, reproducibility appendix.
- [ ] Submit to `amllab@bit.uni-bonn.de`.
- [ ] Tag repo at submitted commit; ensure artifacts reproducible.

---

## Part D — Indicative timeline

| Window | Focus | Output |
|---|---|---|
| 30/05 – 06/06 | Phase 0 + G1 + G2 | H3 fixed; tracker live |
| 07/06 – 18/06 | G4, G5, G10; build midterm deck | Midterm-ready |
| **19/06** | **Midterm presentation** | ✔ |
| 20/06 – 10/07 | G3, G9, G7, G6 | Stretch results |
| 08/07 – 24/07 | Figures + paper method/results; final deck | Final-ready |
| **24/07** | **Final presentation** | ✔ |
| 25/07 – 25/08 | Paper writing + revisions | Draft → polished |
| 26/08 – 31/08 | Proofread + submit | ✔ Submitted |

---

## Part E — Acceptance per gap
- **Code gaps (G1, G2, G6, G7, G8, G10):** `pytest -q` green; re-run relevant cell; new columns/figures in `results/`.
- **Model runs (G3, G4, G9):** full artifact set per model (`*_patch_z.csv`, `*_heatmap.png`, `*_ablation.csv` with perplexity, `*_safety_heads.json`) + FINDINGS row.
- **Metric audit (G5):** labeled CSV + computed agreement %.
- **Deliverables (D1–D3):** paper compiles to ~8 pages; decks rehearsed to time; dry-run Q&A.
- **End-to-end repro:** fresh clone → `python -m safety_circuits.cli run-mvp --model qwen` reproduces a headline number.

---

## Part F — Risks
| Risk | Mitigation |
|---|---|
| Phi-3 port stays broken | Time-box G3; fall back to Falcon3/OLMo-2 + documented exclusion. |
| Underestimating writing time (the 70%) | Start method+results in Phase 3, not after. |
| Kaggle/Colab session limits | Checkpoint per (layer,head); results persist as kernel output. |
| Scope creep from stretch experiments | G8 is optional; cut first. |
| Model mismatch raised in defence | Address proactively (deliberate, documented swap). |
