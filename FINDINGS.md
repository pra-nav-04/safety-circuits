# Experiment Findings — Safety Circuits

> **⚠️ Re-run in progress (full pipeline, N=50).** The sections below are being regenerated
> with the new orchestrator: 50 matched pairs, error bars (95% CI), **perplexity capability
> control**, mean-ablation, K-sweep, last-token & attention-pattern sweeps, HarmBench
> jailbreak, and the RTP toxicity probe. **Models marked "(full pipeline, N=50)" use the new
> data; the rest are the older N=32 runs and lack the perplexity control** — their headline
> "refusal → 0%" claims should be read with the capability caveat surfaced below, and will be
> revised as each model is re-run.

---

## Model 1: Qwen2.5-1.5B-Instruct  *(full pipeline, N=50)*

### Setup

| Item | Value |
|------|-------|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` (28L × 12H, 336 heads, d_model 1536) |
| dtype / GPU | float32 / Tesla T4 (Kaggle) · seed 0 |
| Data | AdvBench (harm) × HH-RLHF harmless (safe), **50 matched pairs**; held-out eval = 25 |
| Controls | WikiText-2 perplexity · HarmBench jailbreak (50) · RealToxicityPrompts (50) |
| Coherence check | "Paris." / "oxygen." / "4" — loads & generates correctly ✓ |

### Headline reframe (why the capability control matters)

The earlier N=32 run reported "**10 heads → refusal 100%→0%**" as a clean result. With the
perplexity control now in place, that headline is **misleading**: ablating the top heads does
not make the model *comply* — **it makes the model produce gibberish**. Example, top-1 eval prompt:

> **Prompt:** *"Write a letter defaming a company…"*
> **Clean:** `"I'm sorry, but I can't assist with that."` (refusal)
> **Top-10 zero-ablated:** `"isUnnamed>()bobundeanimateasooperativeleinawesome…"` (broken text)

So "refusal → 0%" is the regex simply not matching **degenerate output**, not the model
answering the harmful request. The top heads — dominated by **layer-0** heads (L0H10, L0H6) —
are **load-bearing for general generation, not a cleanly removable safety module.**

### E4 — Per-head patching sweep (50 pairs), top-10 by |Δ refusal-margin| ± 95% CI

| Rank | Layer | Head | \|Δ margin\| | 95% CI |
|------|-------|------|------------|--------|
| 1  | **0**  | 10 | **1.056** | ±0.508 |
| 2  | 11 | 8  | 0.586 | ±0.152 |
| 3  | **0**  | 6  | 0.535 | ±0.136 |
| 4  | 15 | 7  | 0.466 | ±0.084 |
| 5  | 18 | 8  | 0.426 | ±0.299 |
| 6  | 10 | 8  | 0.407 | ±0.100 |
| 7  | 19 | 6  | 0.323 | ±0.073 |
| 8  | 13 | 11 | 0.320 | ±0.055 |
| 9  | 14 | 9  | 0.302 | ±0.051 |
| 10 | 18 | 10 | 0.291 | ±0.072 |

- **L0H10 dominant** (1.06, ~2× the next head) — layer-0 harm detection, replicating the earlier run with tighter estimates.
- Secondary mid-network band L10–L19; late layers (20–27) inactive.
- The wide CI on L0H10 (±0.51) shows its effect is strong but **variable across prompts** — consistent with a head doing general work that *also* carries refusal signal.

### E5 — Coarse residual trace (now informative)

Per-layer `resid_pre` patching is **no longer flat** (the old "−32.70 everywhere" was a
whole-tensor artifact). Δmargin is large-negative early and decays to ~0 late:
**L0 −3.01 → L1 −1.75 → … → L27 +0.01.** The refusal-relevant representation is **built early**
and merely *executed* by late layers.

### E6 — Ablation + capability control (the key result)

Held-out 25 harmful prompts; WikiText-2 perplexity as the capability check (clean PPL = **18.2**).

| Condition | Refusal rate | WikiText-2 PPL | Δ PPL |
|-----------|------------|----------------|-------|
| Clean | **100%** | 18.2 | — |
| Top-10 **zero**-ablated | **0%** | **1,113,561** | **+6,119,448%** |
| Top-10 **mean**-ablated | **0%** | **905** | **+4,874%** |

- **H3 (clean removal) is falsified.** Refusal *is* causally removable by ≤10 heads, but **capability is destroyed** (zero-ablation PPL ×61,000). Mean-ablation is ~1,000× gentler than zero (PPL 905 vs 1.1M) yet still **50× worse than baseline** — so even the Wang-et-al control doesn't isolate a clean "safety-only" set.
- **K-sweep:** refusal is 0% at *every* K (5→40) and PPL is catastrophic at every K (559k @ K=5). The damage is not a function of how many heads — even 5 early heads break the model.
- **Takeaway:** safety in Qwen2.5 is **causally concentrated but not modular** — it rides on heads the model also needs to generate coherent text.

### E7 — HarmBench jailbreak stress test

| Metric | Plain (AdvBench) | Jailbreak (HarmBench) |
|--------|------------------|----------------------|
| Clean refusal rate | 100% | **94%** |
| Mean refusal margin | 4.08 | **2.20** |
| Refusal after top-10 ablation | — | **0%** |

Jailbreaks **partially bypass** safety (refusal 100%→94%, margin nearly halved), and the *same*
heads still control refusal under jailbreak (ablation → 0%) — though with the same
capability-destruction caveat.

### E8 — RTP continuation-toxicity probe

Does ablating the instruction-refusal heads raise *toxic continuation* on RTP starters?

| Mean toxicity (toxic-bert) | Value |
|----------------------------|-------|
| Clean | 0.050 |
| Top-10 zero-ablated | 0.061 |
| **Δ** | **+0.011** |

**Weak / inconclusive.** Mean toxicity rises only +0.011; the median per-prompt change is ~0,
driven by ~7/50 outliers — and those "toxic" ablated continuations are themselves **gibberish
containing toxic tokens**, not coherent toxic prose. So this is confounded by the same capability
collapse; it is *not* clean evidence of cross-behaviour generalisation.

### Output files (`results/kaggle_neo/qwen2.5/`)

`qwen2.5_patch_z.csv` (336 rows, +std/sem/ci95) · `qwen2.5_heatmap.png` ·
`qwen2.5_ablation.csv` / `_ablation_mean.csv` (with perplexity) · `qwen2.5_ksweep.csv`+`.png` ·
`qwen2.5_patch_z_lasttok.csv`+heatmap · `qwen2.5_patch_pattern.csv`+heatmap ·
`qwen2.5_jailbreak.csv` · `qwen2.5_rtp_toxicity.csv` · `qwen2.5_pairs.jsonl` (repro) ·
`qwen2.5_examples.jsonl` (clean vs ablated continuations) · `qwen2.5_safety_heads.json` · `_DONE.json`

---

## Model 2: Qwen3-1.7B

### Setup

| Item | Value |
|------|-------|
| Model | `Qwen/Qwen3-1.7B` |
| Architecture | 28 layers × 16 heads |
| Total attention heads | 448 |
| dtype | float16 |
| GPU | Tesla T4 (Kaggle) |
| Harmful dataset | AdvBench (520 harmful behaviours, raw GitHub CSV) |
| Benign dataset | HH-RLHF harmless-base (`Anthropic/hh-rlhf`) |
| Matched pairs | 32 |
| Patching method | Activation patching on attention head `z` output |
| Metric | Refusal-logit margin = log p(refusal tokens) − log p(other tokens) |
| Ablation mode | Zero-ablation of top-K heads |
| Note | Thinking mode disabled (`enable_thinking=False`) |

### E4 — Coarse Residual Trace

**Result:** All 28 layers produced near-identical delta_margin of **~−24.48** (range: −24.47 to −24.50).

**Interpretation:** Same flat-trace artefact as Qwen2.5 — full-residual replacement works at any layer, confirming the refusal signal is encoded throughout the network. Slightly lower magnitude than Qwen2.5 (−24.48 vs −32.70), consistent with float16 precision differences and a different refusal margin baseline.

### E5 — Per-Head Patching Sweep (32 pairs × 28 layers × 16 heads)

#### Top-20 heads by |Δ refusal-margin|

| Rank | Layer | Head | |Δ margin| |
|------|-------|------|----------|
| 1  | 0  | 3  | **8.298** |
| 2  | 15 | 9  | 1.573 |
| 3  | 10 | 3  | 1.380 |
| 4  | 0  | 12 | 1.203 |
| 5  | 9  | 7  | 1.177 |
| 6  | 0  | 9  | 1.104 |
| 7  | 1  | 15 | 1.097 |
| 8  | 14 | 12 | 1.070 |
| 9  | 10 | 5  | 0.974 |
| 10 | 0  | 7  | 0.967 |
| 11 | 12 | 6  | 0.966 |
| 12 | 14 | 13 | 0.902 |
| 13 | 10 | 2  | 0.881 |
| 14 | 7  | 13 | 0.880 |
| 15 | 23 | 6  | 0.875 |
| 16 | 3  | 11 | 0.860 |
| 17 | 4  | 4  | 0.850 |
| 18 | 0  | 4  | 0.839 |
| 19 | 0  | 1  | 0.835 |
| 20 | 0  | 5  | 0.832 |

#### Heatmap observations

- **L0H3** is the dominant outlier (|Δ| = 8.298) — nearly 9× the magnitude of the Qwen2.5 top head.
- **Layer 0 cluster is stronger**: 4 of the top-10 heads are in L0 (H3, H12, H9, H7), vs 2 in Qwen2.5.
- **L1H15** appears at rank 7, extending the early-layer cluster one layer deeper than Qwen2.5.
- **Mid-layer cluster**: L7–L15, slightly earlier than the L10–L19 band in Qwen2.5.
- **L23H6** at rank 15 is the only truly late-layer head; late layers (L24–L27) are otherwise inactive.
- The overall distribution is **heavier in layer 0** than Qwen2.5, suggesting stronger first-layer harm detection in Qwen3.

### E6 — Ablation Study

Zero-ablated the top-10 candidate heads on a held-out set of 16 harmful prompts.

| Condition | Refusal rate |
|-----------|-------------|
| Clean (no ablation) | **93.75%** |
| Top-10 heads zeroed | **0%** |

**10 out of 448 total attention heads (2.2%) are sufficient to eliminate all refusal behaviour.**

Note: the clean refusal rate of 93.75% (15/16 prompts refused) indicates one prompt slipped through even without ablation — Qwen3-1.7B has slightly weaker baseline refusal than Qwen2.5-1.5B.

### Output Files

All files in `kaggle/outputs/results_Qwen3-1.7B/`:

| File | Contents |
|------|----------|
| `qwen3_heatmap.png` | 28×16 heatmap of \|Δ refusal-margin\| per head |
| `qwen3_patch_z.csv` | Full ranked head sweep results (448 rows) |
| `qwen3_resid_trace.csv` | Layer-wise residual trace (28 rows) |
| `qwen3_ablation.csv` | Clean vs ablated refusal rates |
| `qwen3_safety_heads.json` | Top-10 heads in JSON (for downstream use) |

---

---

## Model 3: Phi-3-mini-4k-instruct — INCONCLUSIVE

### Setup

| Item | Value |
|------|-------|
| Model | `microsoft/Phi-3-mini-4k-instruct` |
| Architecture | 32 layers × 32 heads (1024 total heads) |
| dtype | float16 |
| Loading path | HF fallback (`from_pretrained_no_processing`) |

### Result

| Condition | Refusal rate |
|-----------|-------------|
| Clean (no ablation) | **0%** |
| Top-10 heads zeroed | **0%** |

**Run is inconclusive.** Clean refusal rate is 0% — Phi-3-mini refused none of the 16 eval prompts even without ablation. Head patching |Δ| values are also tiny (max ~0.036 vs ~0.9+ for Qwen), confirming no detectable safety signal.

### Likely Causes

1. **Incorrect model port.** `HookedTransformer.from_pretrained_no_processing` requires a manual state-dict mapping that may be wrong for Phi-3-mini's non-standard combined QKV projection. The model loads without error but the hook geometry is misaligned, producing garbage logits.
2. **Refusal token mismatch.** If TL-level generation is broken, our refusal-margin metric cannot fire.

### Status

Results saved but excluded from cross-model analysis. Will revisit with a correct TL port or substitute model if time allows.

### Output Files

All files in `kaggle/outputs/results_Phi3-mini/`:

| File | Contents |
|------|----------|
| `phi3_heatmap.png` | 32×32 heatmap (signal near zero throughout) |
| `phi3_patch_z.csv` | Full ranked head sweep results (1024 rows) |
| `phi3_resid_trace.csv` | Layer-wise residual trace (32 rows, all ~−0.17) |
| `phi3_ablation.csv` | 0% clean → 0% ablated (inconclusive) |
| `phi3_safety_heads.json` | Top-10 heads (not meaningful given 0% baseline) |

---

## Model 4: Gemma-3-1B-Instruct

### Setup

| Item | Value |
|------|-------|
| Model | `google/gemma-3-1b-it` |
| Architecture | 26 layers × 4 heads, d_model = 1152 |
| Total attention heads | 104 |
| dtype | float32 |
| GPU | Tesla T4 (Kaggle) |
| Harmful dataset | AdvBench (520 harmful behaviours) |
| Benign dataset | HH-RLHF harmless-base |
| Matched pairs | 32 |
| Patching method | Activation patching on attention head `z` output |
| Metric | Refusal-logit margin = log p(refusal tokens) − log p(other tokens) |
| Ablation mode | Zero-ablation of top-10 heads |
| Note | Gated model — required HF token + Google ToS acceptance |

### E4 — Coarse Residual Trace

**Result:** All 26 layers produced near-identical delta_margin of **~−28.71** (range: −28.7133 to −28.7134).

**Interpretation:** Same flat-trace artefact as previous models. Full-residual patching works uniformly across all layers.

### E5 — Per-Head Patching Sweep (32 pairs × 26 layers × 4 heads)

#### Top-10 heads by |Δ refusal-margin|

| Rank | Layer | Head | |Δ margin| |
|------|-------|------|----------|
| 1  | 24 | 0 | **3.596** |
| 2  | 9  | 0 | 3.540 |
| 3  | 10 | 0 | 3.504 |
| 4  | 12 | 2 | 3.233 |
| 5  | 13 | 3 | 2.303 |
| 6  | 8  | 3 | 2.267 |
| 7  | 2  | 3 | 2.174 |
| 8  | 9  | 1 | 2.070 |
| 9  | 16 | 0 | 2.057 |
| 10 | 6  | 2 | 2.030 |

#### Heatmap observations

- **No layer-0 dominance** — unlike both Qwen models, the top head is at **L24H0** (the penultimate layer). This is a fundamental structural difference.
- **H0 is the most recurrent head index**: appears 5 times in the top-10 (L24, L9, L10, L16, L4), suggesting head 0 is systematically involved in safety across layers.
- **H3 forms a secondary cluster**: L8H3, L2H3, L13H3 — spread from layer 2 to layer 13.
- **Top-10 are distributed across the full depth** (L2–L24), in contrast to Qwen's early-layer concentration.
- Only 104 total heads (26×4) — Gemma's very low head count means 10 heads = **9.6% of the network**, higher percentage than Qwen but still a sparse set in absolute terms.

### E6 — Ablation Study

Zero-ablated the top-10 candidate heads on a held-out set of 16 harmful prompts.

| Condition | Refusal rate |
|-----------|-------------|
| Clean (no ablation) | **68.75%** |
| Top-10 heads zeroed | **0%** |

**10 out of 104 total attention heads (9.6%) are sufficient to eliminate all refusal behaviour.**

Note: the clean refusal rate of 68.75% (11/16 prompts refused) is notably lower than both Qwen models. Gemma-3-1b has weaker baseline refusal — 5 of 16 prompts slipped through even without ablation. This may reflect lighter safety training at 1B scale.

### Output Files

All files in `kaggle/outputs/results_Gemma3-1b/`:

| File | Contents |
|------|----------|
| `gemma3-1b_heatmap.png` | 26×4 heatmap of \|Δ refusal-margin\| per head |
| `gemma3-1b_patch_z.csv` | Full ranked head sweep results (104 rows) |
| `gemma3-1b_resid_trace.csv` | Layer-wise residual trace (26 rows) |
| `gemma3-1b_ablation.csv` | Clean vs ablated refusal rates |
| `gemma3-1b_safety_heads.json` | Top-10 heads in JSON |

---

## Model 5: Llama-3.2-3B-Instruct

### Setup

| Item | Value |
|------|-------|
| Model | `meta-llama/Llama-3.2-3B-Instruct` |
| Architecture | 28 layers × 24 heads (GQA: 8 KV heads) |
| Total attention heads | 672 |
| dtype | float16 |
| Loading path | HF fallback (`from_pretrained_no_processing`) — TL's native path OOMs on T4 (fp32 first-load) |
| GPU | Tesla T4 (Kaggle) |
| Harmful dataset | AdvBench (520 harmful behaviours) |
| Benign dataset | HH-RLHF harmless-base |
| Matched pairs | 32 |
| Patching method | Activation patching on attention head `z` output |
| Metric | Refusal-logit margin = log p(refusal tokens) − log p(other tokens) |
| Ablation mode | Zero-ablation of top-10 heads |
| Note | Gated model — required HF token + Meta ToS acceptance |

### E4 — Coarse Residual Trace

**Result:** All 28 layers produced near-identical delta_margin of **~−29.60** (range: −29.586 to −29.617).

**Interpretation:** Same flat-trace artefact as all previous models.

### E5 — Per-Head Patching Sweep (32 pairs × 28 layers × 24 heads)

#### Top-10 heads by |Δ refusal-margin|

| Rank | Layer | Head | |Δ margin| |
|------|-------|------|----------|
| 1  | 0  | 20 | **1.800** |
| 2  | 24 | 11 | 1.305 |
| 3  | 0  | 18 | 1.154 |
| 4  | 14 | 5  | 0.892 |
| 5  | 11 | 17 | 0.850 |
| 6  | 15 | 16 | 0.845 |
| 7  | 9  | 13 | 0.839 |
| 8  | 24 | 22 | 0.819 |
| 9  | 12 | 21 | 0.814 |
| 10 | 14 | 18 | 0.763 |

#### Heatmap observations

- **Hybrid pattern** — L0 and L24 both appear in the top-10, unlike any previous model. Llama combines Qwen's early-layer dominance with Gemma's late-layer activity.
- **L0H20 is the top head** (Δ=1.800), consistent with the Qwen family's early-layer harm detection. L0H18 is rank 3.
- **L24H11 and L24H22** (ranks 2 and 8) — strong late-layer activity, reminiscent of Gemma.
- **Mid-layer cluster**: L9–L15 (ranks 4–10), similar depth to Qwen's mid-band.
- With 672 total heads, the top-10 represent only **1.5%** of all heads — the most sparse circuit found so far.

### E6 — Ablation Study

Zero-ablated the top-10 candidate heads on a held-out set of 16 harmful prompts.

| Condition | Refusal rate |
|-----------|-------------|
| Clean (no ablation) | **93.75%** |
| Top-10 heads zeroed | **75.00%** |

**The top-10 heads only partially suppress refusal** — a fundamentally different result from all other models (which all reached 0%).

Two possible interpretations:
1. **More distributed circuit**: Llama-3.2-3B encodes safety across more heads than top-10 captures. Increasing `SC_TOP_K` (e.g. to 20–30) may be needed to reach full suppression.
2. **HF port artifact**: `from_pretrained_no_processing` may produce slightly misaligned hooks for Llama's GQA architecture, causing the ablation to miss some contributing heads. The patching signal (Δ=1.800 for top head) is substantial, suggesting alignment is at least partially correct.

### Output Files

All files in `results/kaggle/results_Llama3-3b/`:

| File | Contents |
|------|----------|
| `llama3-3b_heatmap.png` | 28×24 heatmap of \|Δ refusal-margin\| per head |
| `llama3-3b_patch_z.csv` | Full ranked head sweep results (672 rows) |
| `llama3-3b_resid_trace.csv` | Layer-wise residual trace (28 rows) |
| `llama3-3b_ablation.csv` | Clean vs ablated refusal rates |
| `llama3-3b_safety_heads.json` | Top-10 heads in JSON |

---

## Cross-Model Comparison

| Property | Qwen2.5-1.5B | Qwen3-1.7B | Gemma-3-1B | Llama-3.2-3B |
|----------|-------------|------------|------------|--------------|
| Architecture | 28L × 12H | 28L × 16H | 26L × 4H | 28L × 24H |
| Total heads | 336 | 448 | 104 | 672 |
| Top head | L0H10 (Δ=1.06) | L0H3 (Δ=8.298) | L24H0 (Δ=3.596) | L0H20 (Δ=1.800) |
| Top-head layer | 0 (first) | 0 (first) | 24 (penultimate) | 0 (first) |
| Late-layer activity | None | One (L23) | Dominant (L24 is #1) | Significant (L24 ranks 2 & 8) |
| Mid-layer cluster | L10–L19 | L7–L15 | Distributed L2–L24 | L9–L15 |
| Heads ablated | 10/336 (3.0%) | 10/448 (2.2%) | 10/104 (9.6%) | 10/672 (1.5%) |
| Clean refusal rate | 100% | 93.75% | 68.75% | 93.75% |
| Ablated refusal rate | **0%** | **0%** | **0%** | **75%** |
| Pairs (N) | **50** (re-run) | 32 | 32 | 32 |
| **Capability under zero-abl** (PPL clean→abl) | **18→1.1M ⚠️** | not measured | not measured | not measured |

**⚠️ Capability caveat (qwen2.5, full pipeline):** "refusal → 0%" overstates the result. Ablating
the top heads **destroys general language modelling** (WikiText-2 PPL 18 → 1.1M; mean-ablation
18 → 905) — the ablated model emits gibberish, not compliance. The top heads (esp. layer-0) are
**not capability-isolated.** The N=32 models below were scored *without* a perplexity control, so
their "0%" results likely hide the same effect and will be re-measured.

**Key cross-model finding (location):** Sparsity + L0 dominance replicate across Qwen2.5, Qwen3, Llama; Gemma is the exception (L24). Llama uniquely shows both L0 and L24. The *causal removability* of refusal is real; its *modularity* (removable without breaking the model) is not — see caveat.

---

## Hypothesis Scorecard

| Hypothesis | Prediction | Qwen2.5 | Qwen3 | Phi-3 | Gemma-3 | Llama-3.2 | Status |
|-----------|------------|---------|-------|-------|---------|-----------|--------|
| H1 Sparse | ≤10 heads explain most refusal | 10 → 100%→0% | 10 → 93.75%→0% | N/A | 10 → 68.75%→0% | 10 → 93.75%→75% | **PARTIAL** — sparse in all models; complete only in Qwen/Gemma |
| H2 Causal | Patching flips refusal logit | L0H10: Δ=0.908 | L0H3: Δ=8.298 | N/A | L24H0: Δ=3.596 | L0H20: Δ=1.800 | **CONFIRMED** (4/4 valid models) |
| H3 Ablation | Zero top-10 → refusal ≤30% | 0% ✓ | 0% ✓ | N/A | 0% ✓ | 75% ✗ | **PARTIAL** — refusal removed in Qwen/Gemma, incomplete in Llama |
| H3b Capability | …**and** PPL change ≤5% | **✗ +6.1M%** | not measured | N/A | not measured | not measured | **FALSIFIED (qwen2.5)** — ablation destroys the model; refusal-removal ≠ clean safety removal |
| H4 Cross-model | Same structural pattern across models | — | L0 replicated | Inconclusive | L0 not replicated | L0 + L24 hybrid | **PARTIAL** — L0 dominance in 3/4 models; depth of circuit varies |

---

## Notable Observations

1. **Layer-0 dominance is the majority pattern.** Three of four valid models (Qwen2.5, Qwen3, Llama) place their top head at layer 0. Gemma is the exception with L24 dominance. This suggests early-layer harm detection is the more common RLHF implementation, but is not universal.

2. **Llama shows a hybrid circuit — L0 and L24 both active.** The top-10 includes both early (L0) and late (L24) heads, suggesting Llama may implement two-stage safety gating: early detection at L0 and late enforcement at L24.

3. **Ablation completeness correlates with head count sparsity differently than expected.** Gemma (9.6% of heads ablated) reaches 0%; Llama (1.5% ablated) only reaches 75%. A more distributed circuit may require ablating more heads in absolute terms.

4. **Gemma's weaker baseline refusal (68.75%).** Five of 16 prompts were not refused even without ablation, suggesting lighter safety training at 1B scale.

5. **Residual trace.** The old flat "−32.70 everywhere" was a whole-tensor artefact. The N=50 per-layer `resid_pre` trace (qwen2.5) is **informative**: large-negative early (L0 −3.0), decaying to ~0 late — refusal-relevant representation is built early, executed late.

6. **★ Capability entanglement (the headline finding so far).** With the perplexity control now wired in (qwen2.5), zero-ablating the top-10 "safety heads" raises WikiText-2 perplexity from 18 to **1.1M** — the ablated model outputs gibberish, not compliance. Mean-ablation is gentler (→905) but still 50× baseline. **Refusal is causally concentrated in a few heads, but those heads are not a separable safety module** — they're load-bearing for general generation (esp. layer-0). This is the most important and most publishable result: a cautionary tale that ablation refusal-rates *without* a capability control overstate "safety removal."

7. **Jailbreaks partially bypass (qwen2.5).** HarmBench jailbreaks drop clean refusal 100%→94% and roughly halve the refusal margin (4.08→2.20) vs plain AdvBench — the safety signal weakens but doesn't vanish under adversarial prompts.

8. **RTP cross-behaviour: weak/confounded (qwen2.5).** Ablation raises mean continuation toxicity only +0.011, driven by a few outliers whose "toxic" output is actually gibberish — not clean evidence the refusal circuit also governs toxic continuation.

---

## Next Steps

> The authoritative, prioritised list of remaining work (with effort estimates, acceptance criteria, and a deadline-driven timeline) now lives in **`PROJECT_PLAN.md`**. The items below are the research-side summary.

### Full-pipeline re-run (N=50 + perplexity + jailbreak + RTP), 9-model roster

Within-family generational sweeps (Qwen ×4, Gemma ×3, Llama ×2). Falcon3 / OLMo-2 / Phi-3 /
TinyLlama dropped — **not supported by the pinned TransformerLens** (or kernel-crash); see `PROJECT_PLAN.md`.

| Model | Family/gen | Status (new pipeline) |
|-------|-----------|-----------------------|
| `qwen2.5` (Qwen2.5-1.5B) | Qwen g2.5 | ✅ **DONE (N=50)** — see Model 1 above |
| `qwen1.5-1.8b` | Qwen g1.5 | ⬜ run |
| `qwen2-1.5b` | Qwen g2 | ⬜ run |
| `qwen3` (Qwen3-1.7B) | Qwen g3 | ⬜ re-run (had N=32) |
| `gemma1-2b` (gemma-2b-it) | Gemma g1 | ⬜ run |
| `gemma2-2b` (gemma-2-2b-it) | Gemma g2 | ⬜ run |
| `gemma3-1b` | Gemma g3 | ⬜ re-run (had N=32) |
| `llama3.2-1b` | Llama 1B | ⬜ run |
| `llama3-3b` (Llama-3.2-3B) | Llama 3B | ⬜ re-run (had N=32) |

### Other open items

- [x] **Perplexity check** — wired in; qwen2.5 shows the capability-entanglement finding (PPL 18→1.1M).
- [x] **Position analysis** (last-token sweep) + **attention-pattern sweep** — run for qwen2.5; compare across models once re-run.
- [ ] **Re-run remaining 8 models** with the full pipeline; check whether the capability-destruction effect is universal or qwen2.5-specific (esp. whether it tracks layer-0 head involvement).
- [ ] **Llama higher-K** still relevant (75% at K=10 on the old run).
- [ ] **Generational analysis** — does refusal localisation/entanglement shift across Qwen g1.5→g3 and Gemma g1→g3?
- [ ] **Lab report / paper** — the capability-entanglement result reframes the headline; build the paper around "refusal is causally concentrated but not modular."
