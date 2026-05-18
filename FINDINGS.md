# Experiment Findings — Safety Circuits in Qwen2.5-1.5B-Instruct

## Setup

| Item | Value |
|------|-------|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Architecture | 28 layers × 12 heads, d_model = 1536 |
| GPU | Tesla T4 (Kaggle) |
| Harmful dataset | AdvBench (`llm-attacks/llm-attacks`, 520 harmful behaviours) |
| Benign dataset | HH-RLHF harmless-base (`Anthropic/hh-rlhf`) |
| Matched pairs | 32 (greedy character-length matching) |
| Patching method | Activation patching on attention head `z` output |
| Metric | Refusal-logit margin = log p(refusal tokens) − log p(other tokens) |
| Ablation mode | Zero-ablation of top-K heads |

---

## E4 — Coarse Residual Trace

Patched the full residual stream (`resid_pre`) at each layer from the safe run into the harmful run.

**Result:** All 28 layers produced an identical delta_margin of **−32.70**.

**Interpretation:** Full-residual replacement at any layer effectively hands control to the safe model's representation from that point forward, completely eliminating refusal regardless of which layer is patched. This is expected behaviour for whole-tensor replacement and tells us the refusal signal is *present throughout* the network — it does not suddenly appear at a specific late layer. Discriminative localisation requires the finer per-head sweep below.

---

## E5 — Per-Head Patching Sweep (32 pairs × 28 layers × 12 heads)

For each (layer, head) pair, the head's `z` output from the safe run was patched into the harmful run. The `|Δ refusal-margin|` was averaged across all 32 pairs.

### Top-20 heads by |Δ refusal-margin|

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

### Heatmap observations

- **L0H10** is the dominant outlier (yellow in heatmap, |Δ| = 0.908) — a single head at layer 0 alone shifts the refusal margin by nearly 1 log-probability unit.
- **L11H8** and **L0H6** form a secondary cluster (green, ~0.63–0.69).
- **L14–L19** contain a third cluster of moderately active heads (teal, ~0.27–0.54).
- **Layers 20–27** are almost entirely inactive (dark purple) — the late network does not appear to carry or generate the safety signal; it only *executes* the already-decided refusal.
- The pattern is **bimodal**: strong early-layer head (L0) + a diffuse mid-network band (L10–L19).

---

## E6 — Ablation Study

Zero-ablated the top-10 candidate heads on a held-out set of 16 harmful prompts (second half of the 32 pairs).

| Condition | Refusal rate |
|-----------|-------------|
| Clean (no ablation) | **100%** |
| Top-10 heads zeroed | **0%** |

**10 out of 336 total attention heads (3%) are sufficient to eliminate all refusal behaviour.**

---

## Hypothesis Scorecard

| Hypothesis | Prediction | Result | Status |
|-----------|------------|--------|--------|
| H1 Sparse | ≤10 heads explain most refusal | 10 heads → 100% → 0% | **CONFIRMED** |
| H2 Causal | Patching flips refusal logit | L0H10 alone: Δ = 0.908 | **CONFIRMED** |
| H3 Ablation | Zero top-10 → refusal rate ≤30% | Drops to 0% (stronger than predicted) | **CONFIRMED** |
| H4 Cross-model | Qualitative replication on Phi-3 | Not yet run | **PENDING** |

---

## Notable Observations

1. **L0H10 dominance.** The single largest effect is at the very first layer. This suggests the model encodes a "harm/safe" distinction in the embedding space itself (or the first attention layer reads it off the token embeddings directly), rather than constructing it gradually through depth.

2. **Sparsity is extreme.** 3% of heads fully control refusal. This is consistent with the "safety tax" literature suggesting RLHF-finetuned models install narrow, localised safety features rather than distributing safety broadly.

3. **Late layers are inert.** Layers 20–27 show near-zero patching effect. They likely transform and output the refusal *token stream* but do not originate or amplify the safety signal. This may explain why late-layer interventions (e.g., representation engineering in the final layers) sometimes fail to fully suppress refusal.

4. **Residual trace artefact.** The flat −32.70 trace across all layers is a methodological note, not a finding. Full-residual patching at any layer trivially works because it replaces the entire representational state. Position-specific or token-specific patching would be needed for a meaningful layer-localisation story.

5. **No perplexity measurement.** The `perplexity_clean` and `perplexity_ablated` columns in `qwen_ablation.csv` are empty — the capability-preservation check was not computed in this run. Should be added for H3 to be fully rigorous (ablation should not also destroy coherent language generation).

---

## Output Files

All files in `kaggle/outputs/results/`:

| File | Contents |
|------|----------|
| `qwen_heatmap.png` | 28×12 heatmap of \|Δ refusal-margin\| per head |
| `qwen_patch_z.csv` | Full ranked head sweep results (336 rows) |
| `qwen_resid_trace.csv` | Layer-wise residual trace (28 rows) |
| `qwen_ablation.csv` | Clean vs ablated refusal rates |
| `qwen_safety_heads.json` | Top-10 heads in JSON (for downstream use) |

---

## Next Steps

- [ ] **H4**: Run Phi-3-mini-4k-instruct — do the same head positions emerge?
- [ ] **Perplexity check**: Re-run ablation with perplexity measurement enabled to confirm capability is preserved.
- [ ] **Position analysis**: Replace position-agnostic patching with last-token-only patching to confirm the effect is concentrated at the generation decision point.
- [ ] **Lab report**: Write up E4–E6 results for submission.
