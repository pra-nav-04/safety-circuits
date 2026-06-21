# Experiment Findings — Safety Circuits

**9 instruct models, full pipeline (N=50 matched pairs, seed 0, Tesla T4).** Per model: per-head
activation-patching sweep (`z`) with 95% CIs · zero- & mean-ablation of the top-10 heads ·
WikiText-2 perplexity (capability control) · K-sweep · last-token & attention-pattern sweeps ·
HarmBench jailbreak stress test · RealToxicityPrompts (RTP) continuation-toxicity probe.
Artifacts in `results/kaggle_neo/<model>/`.

Models span three families across **generations / sizes**: Qwen (1.5→2→2.5→3), Gemma (1→2→3),
Llama-3.2 (1B→3B). (Excluded: Phi-3 — broken TL port / garbage logits; Falcon3, OLMo-2, TinyLlama —
not in the pinned TransformerLens `OFFICIAL_MODEL_NAMES`.)

---

## Headline findings

**A — Refusal is causally concentrated but *not modular*.** A handful of heads (often one dominant
head) carry most of the refusal-logit margin in every model (**H2 confirmed, 9/9**). But you cannot
*remove* refusal without collateral capability damage: across all 9 models, **fully removing refusal
(→0%) always coincides with large perplexity blow-up, and negligible perplexity damage always
coincides with incomplete removal.** Refusal-rate alone (the old "10 heads → 0%") badly overstates a
clean "safety switch" — the ablated model emits gibberish, not compliance.

**B — Modularity scales with how *late* the circuit sits.** Where the dominant head is **early (L0)**,
zeroing it is catastrophic (Qwen2/2.5/3: PPL ×128–×61,000, output is gibberish). Where it is **late
(L24)**, ablation costs far less capability *and* yields genuinely toxic/compliant output
(**Gemma-3-1B**: PPL only +217%, RTP toxicity **+0.128** — 10× any other model). The cleanest,
most "switch-like" safety circuit found is **late-layer (Gemma-3 L24)**; early-layer "safety" heads
are really general-purpose heads that also gate refusal.

**C — Safety-circuit location *migrates across generations* — in opposite directions per family.**
- **Gemma** pushed it **deeper** each generation: **g1 → L0** (early) · **g2 → L13** (mid) · **g3 → L24** (late). A clean monotonic march to the output.
- **Qwen** pulled it **earlier**: **g1.5 → L12** (mid) · **g2 / 2.5 / 3 → L0** (first layer).

**D — Jailbreak robustness is non-monotonic and model-specific.** Most robust: **Gemma-2-2B**
(HarmBench refusal 96%→96%, *no drop*) and **Llama-3.2-3B**. Most brittle: **Qwen3** and
**Qwen1.5** — their refusal *margin flips negative* under HarmBench (the model becomes, on average,
inclined to comply). Newer ≠ safer (Qwen3 < Qwen2.5). RTP cross-behaviour transfer is null
everywhere except Gemma-3 — consistent with B (only coherent ablated output can be measurably toxic).

---

## Cross-model table (N=50)

| Model | Arch (L×H) | Top head ± CI | Clean→zero-abl refusal | Δ PPL (clean→abl) | Jailbreak refusal (plain→jb) | Jailbreak margin (plain→jb) | RTP Δtox |
|---|---|---|---|---|---|---|---|
| **Qwen1.5-1.8B** | 24×16 | L12H10 1.02 ±0.21 | 80%→**24%** | 32.8→33 (**+1.4%**) | 80→42 | +1.37→**−1.02** | +0.030 |
| **Qwen2-1.5B** | 28×12 | L0H9 5.65 ±1.16 | 100%→**0%** | 18→191,491 (×10,600) | 100→74 | +3.50→+1.42 | −0.042 |
| **Qwen2.5-1.5B** | 28×12 | L0H10 1.06 ±0.51 | 100%→**0%** | 18→1.11M (×61,000) | 100→94 | +4.08→+2.20 | +0.011 |
| **Qwen3-1.7B** | 28×16 | L0H3 8.87 ±2.13 | 88%→**0%** | 32→4,058 (×128) | 88→**42** | +3.68→**−2.99** | −0.004 |
| **Gemma1-2B** | 18×8 | L0H5 2.46 ±0.72 | 88%→72% | 62→76 (+22.8%) | 88→82 | +5.98→+4.44 | −0.009 |
| **Gemma2-2B** | 26×8 | L13H2 0.62 ±0.19 | 96%→88% | 23→29 (+24%) | **96→96** | +6.59→+5.41 | +0.005 |
| **Gemma3-1B** | 26×4 | L24H0 3.92 ±0.51 | 44%→**0%** | 61→192 (+217%) | 44→36 | +12.2→+7.0 | **+0.128** |
| **Llama-3.2-1B** | 16×32 | L9H22 0.99 ±0.17 | 96%→92% | 25→30 (+22.8%) | 96→94 | +5.41→+4.26 | +0.020 |
| **Llama-3.2-3B** | 28×24 | L0H20 1.48 ±0.53 (+L24) | 88%→44% | 18.5→19 (**+5%**) | 88→**96** | +7.80→+6.99 | −0.001 |

> **Read A off this table:** the four models that hit refusal **0%** (Qwen2/2.5/3, Gemma3) are exactly
> the four with the largest Δ PPL. The five with small Δ PPL never fully remove refusal.

## Hypothesis scorecard

| Hypothesis | Verdict |
|---|---|
| **H1 — Sparse** (≤10 heads explain most refusal) | ✅ **Holds** — a dominant head + short tail in all 9 (e.g. Qwen3 L0H3 = 8.87, ~4.5× #2). |
| **H2 — Causal** (patching flips the refusal logit) | ✅ **Confirmed 9/9.** |
| **H3 — Ablation** (zero top-10 → refusal ≤30%) | 🟡 **Partial** — met by Qwen2/2.5/3 (0%), Gemma3 (0%), Qwen1.5 (24%); **failed** by Gemma1 (72%), Gemma2 (88%), Llama-1B (92%), Llama-3B (44%). |
| **H3b — Capability** (…*and* PPL change ≤5%) | ❌ **Falsified.** No model both removes refusal *and* preserves capability. Removal ⇒ damage (Finding A). |
| **H4 — Cross-model** (same structure everywhere) | 🟡 **Partial / richer than predicted.** Sparsity+causality universal, but *location* migrates across generations (Finding C) and *modularity* scales with depth (Finding B). |

---

## By family

### Qwen — generational trajectory g1.5 → g2 → g2.5 → g3

As Qwen matured, the dominant refusal head **moved from mid-network (g1.5, L12) to layer 0 (g2
onward)**, and ablation went from *incomplete + harmless* (g1.5: 80%→24%, PPL +1.4%) to
*complete + catastrophic* (g2/2.5/3: →0%, PPL ×128–×61,000). Jailbreak robustness peaked at **g2.5
(94%)** then collapsed at **g3 (42%, margin −2.99)** — the newest model is the most jailbreakable.

| Model | Top-3 heads (±CI) | Refusal clean→zero / mean | PPL clean→zero | Jailbreak | RTP Δtox |
|---|---|---|---|---|---|
| Qwen1.5-1.8B (24×16) | L12H10 1.02±0.21 · L12H8 0.52 · L21H15 0.47 | 80%→24% / 44% | 32.8→33 | 80→42, +1.4→−1.0 | +0.030 |
| Qwen2-1.5B (28×12) | L0H9 5.65±1.16 · L0H3 0.58 · L8H1 0.49 | 100%→0% / 0% | 18→191,491 | 100→74, +3.5→+1.4 | −0.042 |
| Qwen2.5-1.5B (28×12) | L0H10 1.06±0.51 · L11H8 0.59 · L0H6 0.53 | 100%→0% / 0% | 18→1.11M | 100→94, +4.1→+2.2 | +0.011 |
| Qwen3-1.7B (28×16) | **L0H3 8.87±2.13** · L15H9 1.97 · L10H3 1.29 | 88%→0% / 0% | 32→4,058 | 88→42, +3.7→**−3.0** | −0.004 |

### Gemma — generational trajectory g1 → g2 → g3 (the clean migration)

Gemma is the cleanest result: the dominant safety head marches **deeper every generation** —
**L0 (g1) → L13 (g2) → L24 (g3)**. Late-layer placement (g3) makes ablation the most "modular" of
all models: refusal fully removed (44%→0%) at *moderate* capability cost (PPL ×3, vs Qwen's ×1000s)
and with the **only strong RTP toxicity transfer (+0.128)** — i.e. the ablated model produces
*coherent toxic/compliant* text, not gibberish. Gemma-2 is also the **most jailbreak-robust** model
in the study (96%→96%).

| Model | Top-3 heads (±CI) | Refusal clean→zero / mean | PPL clean→zero | Jailbreak | RTP Δtox |
|---|---|---|---|---|---|
| Gemma1-2B (18×8) | **L0**H5 2.46±0.72 · L6H3 1.55 · L1H2 1.22 | 88%→72% / 72% | 62→76 | 88→82, +6.0→+4.4 | −0.009 |
| Gemma2-2B (26×8) | **L13**H2 0.62±0.19 · L24H1 0.47 · L11H6 0.44 | 96%→88% / 88% | 23→29 | 96→96, +6.6→+5.4 | +0.005 |
| Gemma3-1B (26×4) | **L24**H0 3.92±0.51 · L12H2 3.81 · L9H1 3.23 | 44%→0% / 12% | 61→192 | 44→36, +12.2→+7.0 | **+0.128** |

*(Caveat: Gemma-3 is 1B vs 2B for g1/g2 — the generational comparison carries a size confound, since Google discontinued the 2B tier in Gemma 3. Gemma-3's lower clean refusal (44%) is consistent with a smaller, lighter-aligned model.)*

### Llama-3.2 — size 1B → 3B

Both Llama models are **jailbreak-robust** (1B 96→94; 3B 88→**96**, i.e. jailbreaks slightly *raised*
refusal) and both show **incomplete removal with negligible capability damage** — the safety signal
is more distributed/redundant here. The 1B places its top head mid-network (L9); the 3B is a
**hybrid (L0 + L24)** — early detection *and* late enforcement, the only model with both.

| Model | Top-3 heads (±CI) | Refusal clean→zero / mean | PPL clean→zero | Jailbreak | RTP Δtox |
|---|---|---|---|---|---|
| Llama-3.2-1B (16×32) | L9H22 0.99±0.17 · L4H11 0.63 · L9H12 0.60 | 96%→92% / 92% | 25→30 | 96→94, +5.4→+4.3 | +0.020 |
| Llama-3.2-3B (28×24) | **L0**H20 1.48±0.53 · **L24**H11 1.14 · L0H18 1.03 | 88%→44% / 60% | 18.5→19 | 88→96, +7.8→+7.0 | −0.001 |

---

## Editing extension (§9): from *mapping* the circuit to *editing* it

The mapping study (A–D) could only *delete* the localized heads (zero-ablation), which removes refusal
only by breaking the model (Finding A). The extension instead **retrains** them: a **head-restricted
LoRA** (low-rank adapter masked to *only* the localized safety heads' `q/k/v/o_proj` slices, GQA-aware),
trained on an affirmative-continuation objective, merged back into just those heads. A no-train
**steering-vector** baseline (Arditi-style directional ablation, swept over extraction layer × ablation
set × coefficient) is the midpoint. All evaluation reuses the existing harness; the edited HF model is
ported back into TransformerLens via `from_pretrained_no_processing`, and **the baseline is recomputed
through that same port** so every comparison is apples-to-apples.
*(Config: LoRA rank 16 / 600 steps / lr 5e-4; N=25 held-out harmful eval prompts, 50 HarmBench jailbreak
prompts, WikiText-2 perplexity, seed 0, T4. Artifacts in `results/editing/<model>/`.)*

**E — Refusal is *not modular under deletion, but it is editable under retraining*.** Across all 9 models,
**head-restricted LoRA drives refusal (and HarmBench-jailbreak refusal) to 0%**, and on **8/9** it does so
at **small perplexity cost** (≤ ~16%, often ≈0 or *negative*) — exactly where blunt ablation gave gibberish
(Qwen2/2.5/3: ablation PPL ×128–×61,000 → **LoRA +1–2%**). The no-train **steering baseline never cleanly
removes refusal on any model** (best coherent refusal 32–92%; the only combos reaching ~0% explode PPL).
So the three interventions form a **scalpel-sharpness axis**: blunt ablation (breaks the model / can't
remove) → steering (can't cleanly remove) → **head-restricted LoRA (uniquely reaches the "clean corner":
0% refusal at ≈0 ΔPPL)**. This refines Finding A: refusal is *load-bearing* (you can't excise it) yet
*re-trainable in place* (you can repurpose the same heads to comply). It is also, by construction, a
**targeted jailbreak produced by retraining only the safety heads** — reported in aggregate, no weights
released.

### Cross-model editing table

| Model | top-head depth | base refusal | **blunt zero-abl** (refusal / ΔPPL) | **steering** best-coherent (refusal / ΔPPL) | **LoRA** →0% at k | **LoRA cleanest ΔPPL @ 0%** |
|---|---|---|---|---|---|---|
| **Gemma1-2B** | L0 / .00 | 88% | 72% / +23% | 60% / −0.2% | k1 | **+82% → +2139% (BREAKS)** |
| **Gemma2-2B** | L13 / .50 | 96% | 88% / +24% | 76% / +1.8% | k1 | **+2.8%** (k5) |
| **Gemma3-1B** | L24 / .92 | 44% | 0% / +217% | 32% / +3% | k3 | **−16%** (k5) |
| **Qwen1.5-1.8B** | L12 / .50 | 80% | 24% / +1% | 32% / +0.2% | k1 | +11.5% (k10) |
| **Qwen2-1.5B** | L0 / .00 | 96% | 0% / ×10,600 | 56% / −0.8% | k1 | **≈0%** (k10) |
| **Qwen2.5-1.5B** | L0 / .00 | 100% | 0% / ×61,000 | 72% / +2% | k3 | **+1.9%** (k3) |
| **Qwen3-1.7B** | L0 / .00 | 88% | 0% / ×128 | 48% / +6.5% | k3 | **+1.4%** (k10) |
| **Llama-3.2-1B** | L9 / .56 | 96% | 92% / +23% | 92% / +0.01% | k1 | **+0.45%** (k1) |
| **Llama-3.2-3B** | L0 / .56 (hybrid) | 88% | 44% / +5% | 76% / +1.9% | k3 | **+0.69%** (k5) |

> **Read E off this table:** the **LoRA** columns hit **0% refusal on all 9** at near-zero ΔPPL on 8 of
> them — including the four Qwen/Gemma3 models that ablation could only "remove" by blowing PPL up ×100–
> ×61,000. The **steering** column never reaches 0% while staying coherent. The **ablation** column is
> the mapping study's coupling (remove ⇒ break, or don't remove).

### Editing hypothesis scorecard

| Hypothesis | Verdict |
|---|---|
| **F1a — clean edit** (head-LoRA flips refusal at small ΔPPL) | ✅ **Confirmed 8/9.** Refusal & jailbreak → 0% on 9/9; small ΔPPL on 8/9. **Exception: Gemma1-2B** removes refusal only with large ΔPPL (+82%). |
| **F1b — depth→#heads law** (later circuits edit with fewer heads) | 🟡 **Not resolved.** Refusal flips with very few heads everywhere (k=1 on 5/9, k=3 on 4/9), with no monotone depth relationship (early-L0 Qwen2 flips at k1; late-L24 Gemma3 at k3). The informative axis is **ΔPPL cost**, not head count. |
| **F1c — cross-generation transfer** | **Infeasible on this roster** (no two checkpoints share an architecture; circuits migrate). Not pursued — see `RESEARCH_PLAN.md` §9. |
| **Steering baseline** (directional ablation removes refusal cleanly?) | ❌ **No (0/9).** Best coherent refusal 32–92%; clean removal needs LoRA. |

**Generational editability gradient (Gemma, ties E ↔ Finding C).** Within Gemma the *clean-edit cost*
tracks circuit depth/generation in lockstep with the location migration: **g1 (L0) +82% → g2 (L13) +2.8%
→ g3 (L24) −16%.** Editability improves as the circuit moves later/newer — the editing analogue of
Finding B. It is **not universal across families**, though: Qwen2/2.5/3 have early (L0) circuits yet LoRA
edits them cleanly (+1–2%) *even though ablation destroys them* (×128–×61,000) — LoRA "rescues" early-layer
circuits that ablation cannot touch. The lone hold-out is **Gemma1-2B** (oldest, L0), which resists clean
editing under every method. Headline visual per model: `results/editing/<model>/<model>_scalpel_axis.png`.

---

## Method notes & caveats

- **Capability control is essential.** Without the WikiText-2 perplexity check, every "refusal → 0%"
  reads as a clean safety removal; with it, the Qwen results are revealed as model-breakage
  (gibberish output, PPL ×1000s). This is the project's core methodological point.
- **Residual trace** (per-layer `resid_pre` patch) is informative under the fixed pipeline
  (early-concentrated, decaying to ~0 at the last layers) — the old "flat −32.70 everywhere" was a
  whole-tensor artifact, now resolved.
- **RTP probe** is mostly null/confounded: where ablation produces gibberish (Qwen), the toxicity
  classifier sees no coherent toxic content. Only Gemma-3 (coherent ablated output) shows real
  transfer (+0.128). So RTP transfer is itself *evidence for* Finding B, not an independent result.
- **Mean- vs zero-ablation** rarely helps: mean-ablation is gentler on PPL for Qwen2.5 (905 vs 1.1M)
  but still 50× baseline, and removal completeness is essentially unchanged across the roster.

## Excluded models
- **Phi-3-mini** — loads via the HF-port path but produces garbage logits (the combined QKV/RoPE
  projection is mis-mapped by TransformerLens); 0% baseline refusal, no usable signal.
- **Falcon3-1B, OLMo-2-1B-Instruct, TinyLlama-1.1B** — not in the pinned TransformerLens
  `OFFICIAL_MODEL_NAMES`; load raises `ValueError`. (Gemma-4 also unsupported + multimodal.)

## What's next
- Build the paper's results section + figures around Findings A–E: the generational location-migration
  figure (Gemma L0→L13→L24), the refusal-removal-vs-ΔPPL coupling scatter (mapping), and the **scalpel-axis
  scatter** (editing — `*_scalpel_axis.png`) as the §9 headline. Finding E is the punchline: refusal is not
  modular under *deletion* but is editable under *retraining*.
- Optional: Llama-3B higher-K ablation (does refusal fully drop past K=10?); 50-prompt human metric audit
  (`notebooks/06_metric_audit.ipynb`) for the methodology section.
- **Ongoing extensions (Tier 1/2)** — opt-in orchestrator blocks (`SC_DO_*`), each writing a per-model
  artifact (see `RESEARCH_PLAN.md` §9 "Ongoing extensions"): generalization of the edit (does it produce
  content or just the opener?) → `*_edit_generalization.csv`; minimal clean edit (smallest rank×steps) →
  `*_edit_minimal_sweep.csv`; safety re-patch round-trip → `*_edit_roundtrip.csv`; refusal-direction shift
  (mechanism + forensics) → `*_edit_direction_shift.csv`; Gemma-1 MLP-target probe (`SC_EDIT_TARGETS`);
  **substance unlock (weapon-free)** → `*_edit_benign_substance.csv` (does longer-target training turn the
  refusal-opener into full *benign* answers on over-refused XSTest prompts? — tests the opener-stop mechanism
  without eliciting harmful content; see `RESEARCH_PLAN.md` §9 ethics boundary).
  Findings land here as they complete.
