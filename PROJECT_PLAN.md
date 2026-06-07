# Project Completion Plan — Mechanistic Interpretability of AI Safety Guardrails

> **Course:** MA-INF 4330 — Lab Explainable AI and Applications, University of Bonn
> **Student:** Pranav Yadav · **Repo:** `safety-circuits`
> **Last updated:** 2026-06-07
>
> Living tracker. Companion to `RESEARCH_PLAN.md` (hypotheses/method), `FINDINGS.md` (results), and `paper/paper.md` (draft). Check off items as they land.

---

## TL;DR

- **Experiments + code + findings: ✅ DONE.** All **9 models** (Qwen ×4, Gemma ×3, Llama ×2) ran at **N=50** with the full pipeline; `FINDINGS.md` has four synthesized findings; the analysis code/tests are complete.
- **Figures + paper Results: ✅ DONE.** `scripts/make_figures.py` → 5 figures + `summary_table.csv`; `paper/paper.md` has the Results section fully drafted.
- **What's left is the GRADE: writing & slides** (70% paper + 30% defence). Remaining = **midterm slides (19/06)**, the rest of the paper sections, final slides (24/07), submission (31/08).
- **The story got stronger than the original hypotheses:** refusal is *concentrated but not modular* — and that's the headline, not a clean "safety switch."

### Locked decisions
- **Model set (final):** Qwen1.5-1.8B, Qwen2-1.5B, Qwen2.5-1.5B, Qwen3-1.7B · Gemma1-2B, Gemma2-2B, Gemma3-1B · Llama-3.2-1B, Llama-3.2-3B — **within-family generational sweeps**. *(The earlier "fix Phi-3 + run Falcon3/OLMo-2" plan was dropped — none are in the pinned TransformerLens `OFFICIAL_MODEL_NAMES`; documented exclusions.)*
- **Ambition:** publication-grade — all stretch experiments run.
- **Paper:** ~8-page conference/workshop style (markdown draft → LaTeX later).

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

### A1. Infrastructure & code (✅ complete)
| Component | File | Status |
|---|---|---|
| Model configs (8 models, refusal tokens, dtype, RoPE flags) | `src/safety_circuits/config.py` | ✅ |
| Model loading (TL-native + HF fallback port) | `src/safety_circuits/models.py` | ✅ |
| Data loaders: AdvBench, HarmBench, RTP, HH-RLHF | `src/safety_circuits/data.py` | ✅ |
| Matched-pair construction | `data.py:build_matched_pairs` | ✅ |
| Refusal metric: logit-margin + regex | `src/safety_circuits/refusal.py` | ✅ |
| Activation caching | `src/safety_circuits/activations.py` | ✅ |
| Patching: head-`z`, MLP-output, residual-stream | `src/safety_circuits/patching.py` | ✅ |
| Ablation: zero **and** mean hooks; perplexity wired in | `src/safety_circuits/ablation.py` | ✅ |
| Aggregation + heatmap plotting (+ std/sem/ci95) | `src/safety_circuits/analysis.py` | ✅ |
| RTP toxicity probe + audit harness | `src/safety_circuits/{toxicity,audit}.py` | ✅ |
| CLI `run-mvp` | `src/safety_circuits/cli.py` | ✅ |
| Kaggle runner (full pipeline, multi-model, preflight) | `kaggle/run_experiment.py` | ✅ |
| Figure generation | `scripts/make_figures.py` | ✅ |
| Docker / Makefile / pyproject / smoke tests | repo root + `tests/` | ✅ |
| Notebooks 01–06 | `notebooks/` | ✅ |

### A2. Experiments run — all 9 at N=50, full pipeline (`FINDINGS.md`, `results/kaggle_neo/`)
| Model | Arch | Top head | Refusal clean→zero-abl | Δ PPL | Jailbreak refusal (plain→jb) |
|---|---|---|---|---|---|
| Qwen1.5-1.8B | 24L×16H | L12H10 (mid) | 80%→24% | +1.4% | 80→42 |
| Qwen2-1.5B | 28L×12H | L0H9 | 100%→0% | ×10,600 | 100→74 |
| Qwen2.5-1.5B | 28L×12H | L0H10 | 100%→0% | ×61,000 | 100→94 |
| Qwen3-1.7B | 28L×16H | L0H3 (8.87) | 88%→0% | ×128 | 88→**42** |
| Gemma1-2B | 18L×8H | L0H5 | 88%→72% | +23% | 88→82 |
| Gemma2-2B | 26L×8H | L13H2 (mid) | 96%→88% | +24% | **96→96** |
| Gemma3-1B | 26L×4H | L24H0 (late) | 44%→0% | +217% | 44→36 |
| Llama-3.2-1B | 16L×32H | L9H22 (mid) | 96%→92% | +23% | 96→94 |
| Llama-3.2-3B | 28L×24H | L0H20 (+L24) | 88%→44% | +5% | 88→96 |

*(Excluded — not TL-supported / broken: Phi-3, Falcon3, OLMo-2, TinyLlama. See `FINDINGS.md`.)*

### A3. Hypothesis scorecard (final — see `FINDINGS.md`)
- **H1 (sparse):** ✅ a dominant head + short tail in all 9.
- **H2 (causal):** ✅ patching flips the refusal logit in all 9.
- **H3 (ablation removes refusal):** 🟡 partial — met by 5/9 (Qwen2/2.5/3, Gemma3, Qwen1.5).
- **H3b (…and capability preserved, ΔPPL ≤5%):** ❌ **falsified** — full removal always wrecks capability (gibberish). *Refusal is concentrated but not modular* (Finding A).
- **H4 (cross-model structure):** 🟡 richer than predicted — location **migrates across generations** (Gemma L0→L13→L24; Qwen mid→L0), and modularity scales with depth (Finding B/C).

---

## Part B — Gap analysis: what is LEFT

### B1. Experimental / scientific gaps — ALL CLOSED
| # | Gap | Status |
|---|---|---|
| G1 | Perplexity / capability control | ✅ run on all 9 — surfaced the headline (refusal removal ⇄ capability damage) |
| G2 | Llama / K-sweep | ✅ run (Llama-3B 88%→44% at K=10; K-sweep per model) |
| G3 | Phi-3 broken port | ✅ resolved as **documented exclusion** (not TL-supported); coherence-check diagnostic added |
| G4 | "Pending models" Falcon3/OLMo-2 | ✅ resolved — **dropped** (not in TL `OFFICIAL_MODEL_NAMES`); replaced by the Qwen/Gemma generational siblings |
| G5 | Refusal-metric validation (50-prompt audit) | 🟡 harness ready (`audit.py`, `06_metric_audit.ipynb`) — **needs you to fill human labels** (optional, supports Method section) |
| G6 | Mean-ablation | ✅ run (mean-cache bug fixed); compared vs zero |
| G7 | Last-token patching | ✅ run (`SC_LASTTOK`) |
| G8 | Attention-pattern patching | ✅ run (`SC_PATTERN`) |
| G9 | HarmBench jailbreak stress test | ✅ run on all 9 (Finding D) |
| G10 | Statistical rigor (std/sem/ci95, seed) | ✅ done — CIs in every `patch_z.csv` |
| G11 | Residual-trace reframed as sanity check | ✅ done in `FINDINGS.md` |

### B2. Deliverable gaps (THE GRADE) — what's left
| # | Gap | Weight | Done |
|---|---|---|---|
| D5 | Publication-quality figures | supports both | ✅ (5 figs + summary table) |
| D1 | **Scientific paper (~8 pages)** | **70%** | 🟡 **Results drafted**; Method/Intro/Related/Discussion/Limitations/Abstract = stubs |
| D3 | **Midterm slides** (15 min, **19/06**) | part of 30% | ☐ **next, highest priority** |
| D2 | **Final slides** (15 min, 24/07) | **30%** | ☐ |
| D4 | Reproducibility polish (one-command repro, README status, LaTeX) | supports paper | 🟡 partial (repro artifacts + seed done; LaTeX pending) |

---

## Part C — How to proceed (forward roadmap, from 2026-06-07)

**The science is done. Everything below is writing & presenting — the 100% of the grade.**

### 1. Midterm slides (D3) — **DO NOW, hard deadline 19/06** (~12 days)
- ~12 slides, all visuals already exist in `paper/figures/`:
  1. Problem (safety is a black box; jailbreaks work) · 2. Method (patching + ablation, 9 models, no training)
  3. Sparse & causal (`fig_sparsity`, `fig_heatmaps`) · 4. **Not modular** — removal⇄capability coupling (`fig_coupling`) — the headline
  5. Depth→modularity (Gemma-3 L24 vs Qwen L0) · 6. **Generational migration** (`fig_migration`) · 7. Jailbreak brittleness (`fig_jailbreak`)
  8. Status + what's next (paper).
- Honest framing: H1/H2 ✅, H3b ✗ (the interesting result), H4 → migration.
- [ ] Build deck · [ ] rehearse to 15 min · [ ] 2–3 backup Q&A slides (method validity, capability control).

### 2. Finish the paper (D1) — parallel, the 70%
`paper/paper.md` has **Results drafted**. Remaining sections (I can draft each):
- [ ] Method · [ ] Introduction · [ ] Related Work · [ ] Discussion · [ ] Limitations · [ ] Abstract
- [ ] Reproducibility appendix · [ ] References.
- Cadence: Method → Discussion → Intro/Related → Abstract; 2 self-reviews + 1 office-hours review.

### 3. Optional rigor (supports the paper, not blocking)
- [ ] 50-prompt human metric audit (`notebooks/06_metric_audit.ipynb`) — fills the Method validity claim (≥90% agreement).
- [ ] Llama-3B higher-K ablation (does refusal fully drop past K=10?).

### 4. Final presentation (D2) — 24/07, in person
- [ ] 15-min deck = the paper's narrative arc with the publication figures (extends the midterm deck).

### 5. Submit (by 31/08 23:59 AOE)
- [ ] Convert to LaTeX (after confirming format) · [ ] proofread + captions + refs · [ ] submit to `amllab@bit.uni-bonn.de` · [ ] tag repo.

### 0. Your action
- [ ] Confirm paper length/format on Discord / Thursday office hours (2–3pm).

---

## Part D — Indicative timeline

| Window | Focus | Output |
|---|---|---|
| ~~…→06/06~~ | ~~experiments + code + findings~~ | ✅ done (9 models, FINDINGS, figures, Results draft) |
| **07/06 – 18/06** | **Midterm deck** + draft paper Method/Discussion | Midterm-ready |
| **19/06** | **Midterm presentation** | ◻ |
| 20/06 – 23/07 | Finish paper (Intro/Related/Abstract); optional audit + Llama-K; final deck | Final-ready |
| **24/07** | **Final presentation** | ◻ |
| 25/07 – 30/08 | Paper polish + LaTeX + revisions | Draft → polished |
| **31/08** | **Submit paper** | ◻ |

---

## Part E — Acceptance per gap
- **Code gaps (G1, G2, G6, G7, G8, G10):** `pytest -q` green; re-run relevant cell; new columns/figures in `results/`.
- **Model runs (G3, G4, G9):** full artifact set per model (`*_patch_z.csv`, `*_heatmap.png`, `*_ablation.csv` with perplexity, `*_safety_heads.json`) + FINDINGS row.
- **Metric audit (G5):** labeled CSV + computed agreement %.
- **Deliverables (D1–D3):** paper compiles to ~8 pages; decks rehearsed to time; dry-run Q&A.
- **End-to-end repro:** fresh clone → `python -m safety_circuits.cli run-mvp --model qwen2.5` reproduces a headline number.

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
| `SC_N_PAIRS` | `50` | matched pairs for the main z-sweep |
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

## Part F — Risks (current)
| Risk | Mitigation |
|---|---|
| **Underestimating writing time (the 70%+30%)** — the only real risk now | Experiments are done; start the midterm deck + Method section this week. Don't re-run models. |
| Midterm deadline (19/06) crowds the paper | Reuse the existing figures verbatim in slides; the deck *is* a subset of the paper narrative. |
| "Why not TinyLlama/Phi-3 from the proposal?" in defence | Documented: not TL-supported; swapped to Qwen/Gemma/Llama generational sweeps (a stronger design). |
| Headline is a *negative* result (not a clean switch) | Frame as the contribution: "concentrated but not modular" + capability control is the methodological lesson; pre-registered as a publishable failure mode in `RESEARCH_PLAN.md`. |
| Paper format/length uncertain | Confirm with instructors; markdown draft converts cleanly to LaTeX. |
