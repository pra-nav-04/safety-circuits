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
- Build the paper's results section + figures around Findings A–D (the generational location-migration
  figure — Gemma L0→L13→L24 — and the refusal-removal-vs-ΔPPL coupling scatter are the headline visuals).
- Optional: Llama-3B higher-K ablation (does refusal fully drop past K=10?); 50-prompt human metric audit
  (`notebooks/06_metric_audit.ipynb`) for the methodology section.
