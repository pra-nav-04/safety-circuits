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

### A2. Experiments run (see `FINDINGS.md`, artifacts in `results/kaggle_neo/`)
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

## Part G — Kaggle run recipe (one-shot, all models)

`kaggle/run_experiment.py` is now a **multi-model orchestrator**: one kernel run loops
**every model sequentially, cheapest-first** (single T4 → no parallelism), runs the full
suite per model, **skips failures with a logged traceback**, flushes results as it goes,
and zips everything. The kernel is a **thin bootstrap** (`kaggle/kernel.ipynb`) that
git-pulls the repo and `runpy`s the orchestrator — so logic updates ship via `git push`;
the browser notebook is set up once and never edited again.

**Run it:** `git push origin main` → open the notebook → "Save & Run All".
Then download: `python scripts/kaggle_api.py output` (→ `results/kaggle_neo/`), or grab
`/kaggle/working/safety_circuits_results.zip` and commit it under `results/kaggle_neo/`.

| Env var | Default | Effect |
|---|---|---|
| `SC_MODELS` | default set (cheapest-first) | comma list to subset / resume, e.g. `llama3-3b,phi3` |

> **Default model set** (when `SC_MODELS` is unset) — 9 TL-supported models for within-family generational sweeps:
> `qwen1.5-1.8b, qwen2-1.5b, qwen2.5, qwen3` (Qwen ×4 generations) · `gemma1-2b, gemma2-2b, gemma3-1b` (Gemma ×3 generations) · `llama3.2-1b, llama3-3b` (Llama ×2 sizes).
> **Run one model per session** via `SC_MODELS="<key>"` — loading several back-to-back OOMs the ~13 GB system RAM (kernel restart). Gemma + Llama keys are gated (accept terms under your HF token).
> **Excluded** (unsupported by the pinned TransformerLens / broken): `tinyllama`, `falcon3-1b`, `olmo2-1b` (not in `OFFICIAL_MODEL_NAMES`), `phi3` (garbage logits / OOM-kills kernel), `gemma4-e2b` (Gemma 4: not in TL list + multimodal arch — kept as a 1-min preflight probe). Opt into any via `SC_MODELS=<key>`.
>
> **Jailbreak (G9) needs gated data:** `walledai/HarmBench` is gated — accept its terms at hf.co/datasets/walledai/HarmBench under the account whose `HF_TOKEN` the run uses, else the jailbreak add-on is skipped with a clear message (non-fatal; everything else still runs).
| `SC_SKIP_EXISTING` | `0` | skip models with a `_DONE.json` already present (resume) |
| `SC_N_PAIRS` | `32` | matched pairs for the main z-sweep |
| `SC_HEAVY_PAIRS` | `8` | pairs for the doubler sweeps (last-token, pattern) — bounds cost |
| `SC_TOP_K` | `10` · `SC_K_SWEEP` `5,10,15,20,30,40` · `SC_PPL_TEXTS` `64` · `SC_SEED` `0` | as before |
| `SC_COHERENCE` / `SC_MEAN_ABLATION` / `SC_LASTTOK` / `SC_PATTERN` / `SC_JAILBREAK` | **`1`** | all add-ons ON by default |

**Feasibility:** "everything × ~7 models" will **not** fit in one ~12h T4 session. The loop
is built to resume — run once, then for whatever didn't finish set
`SC_MODELS=<remaining>` (+ `SC_SKIP_EXISTING=1` if you attached the prior output) and run
again. Cheapest-first means the small models (gemma3-1b, qwen) complete early. Prefer an
**interactive** session (or 2 chunks) so flushed per-model results survive a timeout.

**Outputs** in `/kaggle/working/`: `results/<model>/` with `*_patch_z.csv` (std/sem/ci95),
`*_heatmap.png`, `*_ablation.csv` (perplexity), `*_ksweep.csv`+`.png`, `*_ablation_mean.csv`,
`*_patch_z_lasttok.csv`+heatmap, `*_patch_pattern.csv`+heatmap, `*_jailbreak.csv`,
`*_coherence.json`, `*_safety_heads.json`, `_DONE.json`; plus top-level `_run_summary.json`,
`_run_log.txt`, and `safety_circuits_results.zip`.

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
