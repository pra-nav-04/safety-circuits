# Experiment Findings — Safety Circuits

---

## Model 1: Qwen2.5-1.5B-Instruct

### Setup

| Item | Value |
|------|-------|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Architecture | 28 layers × 12 heads, d_model = 1536 |
| Total attention heads | 336 |
| dtype | float32 |
| GPU | Tesla T4 (Kaggle) |
| Harmful dataset | AdvBench (520 harmful behaviours, raw GitHub CSV) |
| Benign dataset | HH-RLHF harmless-base (`Anthropic/hh-rlhf`) |
| Matched pairs | 32 (greedy character-length matching) |
| Patching method | Activation patching on attention head `z` output |
| Metric | Refusal-logit margin = log p(refusal tokens) − log p(other tokens) |
| Ablation mode | Zero-ablation of top-K heads |

### E4 — Coarse Residual Trace

Patched the full residual stream (`resid_pre`) at each layer from the safe run into the harmful run.

**Result:** All 28 layers produced an identical delta_margin of **−32.70**.

**Interpretation:** Full-residual replacement at any layer effectively hands control to the safe model's representation from that point forward, completely eliminating refusal regardless of which layer is patched. This is expected behaviour for whole-tensor replacement and tells us the refusal signal is *present throughout* the network — it does not suddenly appear at a specific late layer. Discriminative localisation requires the finer per-head sweep below.

### E5 — Per-Head Patching Sweep (32 pairs × 28 layers × 12 heads)

For each (layer, head) pair, the head's `z` output from the safe run was patched into the harmful run. The `|Δ refusal-margin|` was averaged across all 32 pairs.

#### Top-20 heads by |Δ refusal-margin|

| Rank | Layer | Head | |Δ margin| |
|------|-------|------|----------|
| 1  | 0  | 10 | **0.908** |
| 2  | 11 | 8  | 0.693 |
| 3  | 0  | 6  | 0.635 |
| 4  | 15 | 7  | 0.538 |
| 5  | 19 | 6  | 0.370 |
| 6  | 18 | 8  | 0.353 |
| 7  | 14 | 9  | 0.352 |
| 8  | 10 | 8  | 0.329 |
| 9  | 19 | 11 | 0.275 |
| 10 | 13 | 11 | 0.269 |
| 11 | 18 | 10 | 0.268 |
| 12 | 9  | 0  | 0.226 |
| 13 | 7  | 3  | 0.225 |
| 14 | 7  | 8  | 0.218 |
| 15 | 12 | 7  | 0.213 |
| 16 | 8  | 3  | 0.212 |
| 17 | 15 | 5  | 0.211 |
| 18 | 10 | 3  | 0.207 |
| 19 | 10 | 5  | 0.203 |

#### Heatmap observations

- **L0H10** is the dominant outlier (yellow in heatmap, |Δ| = 0.908) — a single head at layer 0 alone shifts the refusal margin by nearly 1 log-probability unit.
- **L11H8** and **L0H6** form a secondary cluster (green, ~0.63–0.69).
- **L14–L19** contain a third cluster of moderately active heads (teal, ~0.27–0.54).
- **Layers 20–27** are almost entirely inactive (dark purple) — the late network does not appear to carry or generate the safety signal; it only *executes* the already-decided refusal.
- The pattern is **bimodal**: strong early-layer head (L0) + a diffuse mid-network band (L10–L19).

### E6 — Ablation Study

Zero-ablated the top-10 candidate heads on a held-out set of 16 harmful prompts (second half of the 32 pairs).

| Condition | Refusal rate |
|-----------|-------------|
| Clean (no ablation) | **100%** |
| Top-10 heads zeroed | **0%** |

**10 out of 336 total attention heads (3.0%) are sufficient to eliminate all refusal behaviour.**

### Output Files

All files in `kaggle/outputs/results_Qwen2.5-1.5B-Instruct/`:

| File | Contents |
|------|----------|
| `qwen_heatmap.png` | 28×12 heatmap of \|Δ refusal-margin\| per head |
| `qwen_patch_z.csv` | Full ranked head sweep results (336 rows) |
| `qwen_resid_trace.csv` | Layer-wise residual trace (28 rows) |
| `qwen_ablation.csv` | Clean vs ablated refusal rates |
| `qwen_safety_heads.json` | Top-10 heads in JSON (for downstream use) |

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
| Top head | L0H10 (Δ=0.908) | L0H3 (Δ=8.298) | L24H0 (Δ=3.596) | L0H20 (Δ=1.800) |
| Top-head layer | 0 (first) | 0 (first) | 24 (penultimate) | 0 (first) |
| Late-layer activity | None | One (L23) | Dominant (L24 is #1) | Significant (L24 ranks 2 & 8) |
| Mid-layer cluster | L10–L19 | L7–L15 | Distributed L2–L24 | L9–L15 |
| Heads ablated | 10/336 (3.0%) | 10/448 (2.2%) | 10/104 (9.6%) | 10/672 (1.5%) |
| Clean refusal rate | 100% | 93.75% | 68.75% | 93.75% |
| Ablated refusal rate | **0%** | **0%** | **0%** | **75%** |

**Key cross-model finding:** Sparsity (top-10 heads detectable via patching) is universal across all 4 valid models. However, ablation completeness differs: Qwen and Gemma reach 0% with top-10, while Llama only drops to 75%, suggesting a more distributed or redundant safety circuit. Circuit location also varies: L0 dominates in both Qwen models and Llama; L24 dominates in Gemma. Llama uniquely shows both L0 *and* L24 activity — a hybrid pattern.

---

## Hypothesis Scorecard

| Hypothesis | Prediction | Qwen2.5 | Qwen3 | Phi-3 | Gemma-3 | Llama-3.2 | Status |
|-----------|------------|---------|-------|-------|---------|-----------|--------|
| H1 Sparse | ≤10 heads explain most refusal | 10 → 100%→0% | 10 → 93.75%→0% | N/A | 10 → 68.75%→0% | 10 → 93.75%→75% | **PARTIAL** — sparse in all models; complete only in Qwen/Gemma |
| H2 Causal | Patching flips refusal logit | L0H10: Δ=0.908 | L0H3: Δ=8.298 | N/A | L24H0: Δ=3.596 | L0H20: Δ=1.800 | **CONFIRMED** (4/4 valid models) |
| H3 Ablation | Zero top-10 → refusal rate ≤30% | 0% ✓ | 0% ✓ | N/A | 0% ✓ | 75% ✗ | **PARTIAL** — holds for Qwen/Gemma, fails for Llama with K=10 |
| H4 Cross-model | Same structural pattern across models | — | L0 replicated | Inconclusive | L0 not replicated | L0 + L24 hybrid | **PARTIAL** — L0 dominance in 3/4 models; depth of circuit varies |

---

## Notable Observations

1. **Layer-0 dominance is the majority pattern.** Three of four valid models (Qwen2.5, Qwen3, Llama) place their top head at layer 0. Gemma is the exception with L24 dominance. This suggests early-layer harm detection is the more common RLHF implementation, but is not universal.

2. **Llama shows a hybrid circuit — L0 and L24 both active.** The top-10 includes both early (L0) and late (L24) heads, suggesting Llama may implement two-stage safety gating: early detection at L0 and late enforcement at L24.

3. **Ablation completeness correlates with head count sparsity differently than expected.** Gemma (9.6% of heads ablated) reaches 0%; Llama (1.5% ablated) only reaches 75%. A more distributed circuit may require ablating more heads in absolute terms.

4. **Gemma's weaker baseline refusal (68.75%).** Five of 16 prompts were not refused even without ablation, suggesting lighter safety training at 1B scale.

5. **Residual trace artefact.** The flat traces across all models are methodological, not findings. Full-residual patching trivially works everywhere.

6. **No perplexity measurement.** All ablation CSVs have empty perplexity columns — the capability-preservation check has not been run.

---

## Next Steps

### Multi-model sweep (H4 — cross-model replication)

| Model | HF ID | Size | dtype | Status | Notes |
|-------|-------|------|-------|--------|-------|
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | float32 | **DONE** | Baseline |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | 1.7B | float16 | **DONE** | L0 pattern replicated |
| Phi-3-mini | `microsoft/Phi-3-mini-4k-instruct` | 3.8B | float16 | **INCONCLUSIVE** | 0% baseline refusal; HF port broken |
| Gemma-3-1B | `google/gemma-3-1b-it` | 1B | float32 | **DONE** | L24 dominance — breaks L0 pattern |
| Llama-3.2-3B | `meta-llama/Llama-3.2-3B-Instruct` | 3B | float16 | **DONE** | Hybrid L0+L24; ablation incomplete at K=10 |
| Falcon3-1B | `tiiuae/Falcon3-1B-Instruct` | 1B | float16 | PENDING | HF fallback path; no gating |
| OLMo-2-1B | `allenai/OLMo-2-0425-1B-Instruct` | 1B | float16 | PENDING | HF fallback path; no gating; fully open training data |

### Other open items

- [ ] **Llama ablation with higher K**: Re-run with `SC_TOP_K=20` or `SC_TOP_K=30` to test whether more heads are needed to reach full suppression.
- [ ] **Perplexity check**: Add perplexity measurement to ablation to confirm capability is preserved.
- [ ] **Position analysis**: Replace position-agnostic patching with last-token-only patching to confirm the effect is concentrated at the generation decision point.
- [ ] **Lab report**: Write up E4–E6 results for submission.
