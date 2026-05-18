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

## Cross-Model Comparison

| Property | Qwen2.5-1.5B | Qwen3-1.7B |
|----------|-------------|------------|
| Architecture | 28L × 12H | 28L × 16H |
| Total heads | 336 | 448 |
| Top head | L0H10 (Δ=0.908) | L0H3 (Δ=8.298) |
| L0 heads in top-10 | 2 (H10, H6) | 4 (H3, H12, H9, H7) |
| Mid-layer cluster | L10–L19 | L7–L15 |
| Late layers (≥L20) | Inactive | Mostly inactive (one L23 head) |
| Heads controlling refusal | 10/336 (3.0%) | 10/448 (2.2%) |
| Clean refusal rate | 100% | 93.75% |
| Ablated refusal rate | 0% | 0% |

**Key cross-model finding:** Both Qwen models show the same structural pattern — a dominant layer-0 head plus a diffuse mid-layer cluster — despite different head counts and training recipes. The specific head indices differ but the architectural motif (early + mid, late inert) is preserved. This supports H4: safety circuits are a *structural pattern* in the Qwen family, not a quirk of one checkpoint.

---

## Hypothesis Scorecard

| Hypothesis | Prediction | Qwen2.5 | Qwen3 | Phi-3 | Status |
|-----------|------------|---------|-------|-------|--------|
| H1 Sparse | ≤10 heads explain most refusal | 10 → 100%→0% | 10 → 93.75%→0% | N/A (0% baseline) | **CONFIRMED** (Qwen family) |
| H2 Causal | Patching flips refusal logit | L0H10: Δ=0.908 | L0H3: Δ=8.298 | N/A | **CONFIRMED** (Qwen family) |
| H3 Ablation | Zero top-10 → refusal rate ≤30% | 0% (stronger) | 0% (stronger) | N/A | **CONFIRMED** (Qwen family) |
| H4 Cross-model | Same structural pattern across models | — | L0 dominance replicated | Inconclusive (loading issue) | **PARTIAL** (within-family; Mistral/OLMo pending) |

---

## Notable Observations

1. **Layer-0 dominance is conserved.** Both Qwen versions place the single largest safety head at layer 0. This strongly suggests the Qwen family encodes harm/safe discrimination at the very first attention layer, likely reading it directly from token embeddings. The specific head index (H10 vs H3) is irrelevant — the positional pattern is what matters.

2. **Sparsity is extreme in both models.** 3.0% and 2.2% of heads respectively fully control refusal. This is consistent with the "safety tax" literature: RLHF installs narrow, localised safety features rather than distributing safety broadly across the network.

3. **Late layers are inert in both models.** Layers 20–27 show near-zero patching effect. They likely transform and output the refusal *token stream* but do not originate or amplify the safety signal.

4. **|Δ| magnitude gap between models.** The Qwen3 top head (8.298) is ~9× larger than Qwen2.5's (0.908). This is likely a combination of float16 vs float32 precision, a different refusal margin baseline, and Qwen3's stronger layer-0 safety encoding. Raw magnitudes should not be compared directly across models.

5. **Residual trace artefact.** The flat traces (−32.70 and −24.48) are a methodological note, not findings. Full-residual patching trivially works everywhere.

6. **No perplexity measurement.** Both ablation CSVs have empty perplexity columns — the capability-preservation check has not been run. Should be added before H3 is considered fully rigorous.

---

## Next Steps

### Multi-model sweep (H4 — cross-model replication)

| Model | HF ID | Size | dtype | Status | Notes |
|-------|-------|------|-------|--------|-------|
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | float32 | **DONE** | Baseline |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | 1.7B | float16 | **DONE** | L0 pattern replicated |
| Phi-3-mini | `microsoft/Phi-3-mini-4k-instruct` | 3.8B | float16 | **INCONCLUSIVE** | 0% baseline refusal; HF port broken |
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.1` | 7B | float16 | PENDING | TL-native; next up |
| OLMo-2-7B | `allenai/OLMo-2-1124-7B-Instruct` | 7B | float16 | PENDING | No HF auth needed |
| Gemma-3-1B | `google/gemma-3-1b-it` | 1B | float32 | PENDING | **GATED** — needs HF token + Google terms |

Change `SC_MODEL` in the Kaggle notebook cell to run each model.

### Other open items

- [ ] **Perplexity check**: Add perplexity measurement to ablation to confirm capability is preserved.
- [ ] **Position analysis**: Replace position-agnostic patching with last-token-only patching to confirm the effect is concentrated at the generation decision point.
- [ ] **Lab report**: Write up E4–E6 results for submission.
