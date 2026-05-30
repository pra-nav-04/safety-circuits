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
| G1 | **Perplexity / capability check never measured** | H3 success criterion (≤5% Δ). `perplexity()` exists but `evaluate_ablation` hardcoded `None`. Highest-priority. | `ablation.py`, `data.py`, `cli.py`, `run_experiment.py` | M | 🟡 code done — Kaggle run pending |
| G2 | **Llama ablation incomplete (75%)** | K-sweep (10→40) to find threshold or document distributed circuit. | `ablation.ablation_k_sweep`, `analysis.plot_k_sweep`, runner `SC_K_SWEEP` | S | 🟡 code done — Kaggle run pending |
| G3 | **Phi-3 broken (0% baseline)** | Correct combined-QKV/RoPE TL port, or documented swap. | `models.quick_coherence_check` (diagnostic) + `_load_via_hf_port` doc | M–L | 🟡 diagnostic added; **manual QKV remap still needs GPU iteration** |
| G4 | **Pending models not run** (Falcon3-1B, OLMo-2-1B) | More models → stronger H4; OLMo-2 fully-open-data. | `config.py` (already configured) | S each | 🟡 no code needed — Kaggle run pending |
| G5 | **No refusal-metric validation** | ≥90% agreement vs human labels on 50-prompt audit. | `audit.py`, `notebooks/06_metric_audit.ipynb` | M | 🟡 harness done — needs model run + human labels |
| G6 | **Mean-ablation never run** | Wang et al. confound control; code existed but mean-cache shape was buggy. | `ablation.compute_mean_z_cache` (fixed) + runner `SC_MEAN_ABLATION` | S | 🟡 code done — Kaggle run pending |
| G7 | **Last-token-only patching not used** | Reviewers ask if effect concentrates at decision token. | runner `SC_LASTTOK` (uses `patch_each_head(position=-1)`) | S | 🟡 code done — Kaggle run pending |
| G8 | **Attention-pattern patching not implemented** | Pattern vs value-weighted output. | `patching.patch_each_head_pattern`, runner `SC_PATTERN` | M | 🟡 code done — Kaggle run pending |
| G9 | **HarmBench jailbreak stress test not run** | Do safety heads still fire under jailbreaks? | `jailbreak.py`, runner `SC_JAILBREAK` | M | 🟡 code done — Kaggle run pending |
| G10 | **No statistical rigor** | Error bars / CIs / fixed seed across pairs. | `analysis.aggregate_pairs` (std/sem/ci95), runner `SC_SEED` | M | ✅ code done (std/sem/ci95 columns + seed) |
| G11 | **Residual-trace "flat" artifact** | Frame as sanity check, not a finding. | `FINDINGS.md`, paper | trivial | ☐ (writeup) |

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
> **All code below is now implemented, tested (model-free), committed, and pushed.** What remains is GPU execution on Kaggle (see Part G) + folding numbers into `FINDINGS.md`.
1. [x] **G1** perplexity wired into `evaluate_ablation` + runner + CLI; `load_wikitext2()`; `perplexity_pct_change`. → Kaggle run pending.
2. [x] **G2** `ablation_k_sweep()` + `plot_k_sweep()` + runner `SC_K_SWEEP`. → Kaggle run pending.
3. [x] **G4** Falcon3/OLMo-2 already configured (no code). → Kaggle run pending.
4. [x] **G5** `audit.py` (`build_audit_sheet`/`compute_agreement`) + `notebooks/06_metric_audit.ipynb`. → needs model run + human labels.
5. [x] **G10** `aggregate_pairs` now emits std/sem/ci95; runner sets/report seed.

### Phase 2 — Midterm presentation (deliver 19/06)
- [ ] 15-min deck (~12–15 slides): problem → method → working result (sparse heads, ablation collapse) → cross-model picture → next steps.
- [ ] Lead visual: clean per-head heatmap + "10/336 heads → 100%→0%" headline.
- [ ] Honest status slide (H2 confirmed; H1/H3/H4 partial; perplexity now controlled).
- [ ] Rehearse to 15 min + backup Q&A slides.

### Phase 3 — Stretch experiments → publication-grade (20/06 → ~20/07)
> **Code for G6–G9 is implemented, tested, committed, pushed.** Enable via the Part G flags on the Kaggle run. G3 has a diagnostic; the underlying port fix still needs GPU iteration.
6. [~] **G3** `quick_coherence_check()` diagnostic added + documented. Manual QKV remap (real fix) still needs GPU iteration, else keep Phi-3 excluded.
7. [x] **G9** `jailbreak.py` + runner `SC_JAILBREAK`. → Kaggle run pending.
8. [x] **G7** runner `SC_LASTTOK` (position=-1 sweep + heatmap). → Kaggle run pending.
9. [x] **G6** mean-ablation fixed (`compute_mean_z_cache`) + runner `SC_MEAN_ABLATION`. → Kaggle run pending.
10. [x] **G8** `patch_each_head_pattern()` + runner `SC_PATTERN`. → Kaggle run pending.
11. [ ] **G11** Reframe residual-trace as sanity check everywhere. _(writeup)_

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

## Part G — Kaggle run recipe (after coding)

The runner `kaggle/run_experiment.py` is driven by env vars. Defaults give the core
experiment + perplexity (G1) + K-sweep (G2); the heavier analyses are opt-in.

| Env var | Default | Effect |
|---|---|---|
| `SC_MODEL` | `qwen` | model key from `config.MODELS` (`qwen`, `qwen3`, `gemma3-1b`, `llama3-3b`, `phi3`, `falcon3-1b`, `olmo2-1b`) |
| `SC_N_PAIRS` | `32` | matched pairs for the sweep |
| `SC_TOP_K` | `10` | heads ablated in the headline ablation |
| `SC_SEED` | `0` | determinism (G10) |
| `SC_PPL_TEXTS` | `64` | WikiText-2 snippets for perplexity (G1); `0` skips |
| `SC_K_SWEEP` | `5,10,15,20,30,40` | ablation K-sweep (G2); empty skips |
| `SC_COHERENCE` | `1` | port sanity check (G3) |
| `SC_MEAN_ABLATION` | `0` | also run mean-ablation (G6) |
| `SC_LASTTOK` | `0` | also run last-token head sweep (G7) — doubles sweep time |
| `SC_PATTERN` | `0` | also run attention-pattern sweep (G8) — doubles sweep time |
| `SC_JAILBREAK` | `0` | HarmBench jailbreak stress test (G9) |

**Suggested passes per model** (T4 time budget): one "full" run with everything on —
`SC_MEAN_ABLATION=1 SC_LASTTOK=1 SC_PATTERN=1 SC_JAILBREAK=1` — for the 4 valid models;
core-only for Falcon3/OLMo-2 first, then extras if time allows. Phi-3: run with
`SC_COHERENCE=1` and check the completions are coherent before trusting any numbers.

Outputs per model in `/kaggle/working/`: `*_patch_z.csv` (now with std/sem/ci95),
`*_heatmap.png`, `*_ablation.csv` (now with perplexity), `*_ksweep.csv`+`.png`,
`*_safety_heads.json`, plus `*_ablation_mean.csv`, `*_patch_z_lasttok.csv`+heatmap,
`*_patch_pattern.csv`+heatmap, `*_jailbreak.csv` when those flags are on.

The 50-prompt metric audit (G5) runs separately via `notebooks/06_metric_audit.ipynb`.

---

## Part F — Risks
| Risk | Mitigation |
|---|---|
| Phi-3 port stays broken | Time-box G3; fall back to Falcon3/OLMo-2 + documented exclusion. |
| Underestimating writing time (the 70%) | Start method+results in Phase 3, not after. |
| Kaggle/Colab session limits | Checkpoint per (layer,head); results persist as kernel output. |
| Scope creep from stretch experiments | G8 is optional; cut first. |
| Model mismatch raised in defence | Address proactively (deliberate, documented swap). |
