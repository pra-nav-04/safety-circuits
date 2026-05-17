# Research plan — Mechanistic Interpretability of AI Safety Guardrails

> Pranav Yadav · MA-INF 4330 · University of Bonn
> Working title: *Finding the safety switches inside small language models.*

---

## 1. Research question

**Are there a small number of attention heads / MLP layers in a safety-tuned small LM that causally produce its refusal behaviour on toxic prompts — and if so, can we identify them with activation patching and confirm their role by ablation?**

Hypotheses:

- **H1 (sparse):** A handful (≤ 10) of (layer, head) pairs explain most of the refusal logit on a TinyLlama / Phi-3-class model. Sparsity-of-circuits prior, as observed in IOI (Wang et al. 2022) and Anthropic's safety-circuit work.
- **H2 (causal):** Activation patching from a harmless → harmful run on these components flips the model's refusal token logit by ≥ X% (X to be fixed during MVP).
- **H3 (ablation-confirmed):** Zero-ablating the identified components drops refusal rate on a held-out toxic split from ≥ 80% to ≤ 20%, without breaking general-language perplexity by more than 5%.
- **H4 (cross-model):** The *location* (relative layer depth, attention vs MLP) of the circuit replicates qualitatively between TinyLlama and Phi-3, even though absolute indices differ.

If H1–H3 hold on TinyLlama and at least the *shape* of H4 holds on Phi-3, the lab's central claim — that "safety" is mechanistic, not diffuse — is supported on the small-model regime.

---

## 2. Method

Frozen models. Forward passes + hooks. No gradient updates anywhere.

### 2.1 Setup

- Models: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `microsoft/Phi-3-mini-4k-instruct`. Both loaded into TransformerLens `HookedTransformer`.
- Compute: Colab Pro single GPU (A100 / L4 if available).
- Determinism: temperature 0, fixed seed, greedy decode for the *first refusal token*.

### 2.2 Data

| Source | Role | Use |
|---|---|---|
| **AdvBench** (`harmful_behaviors.csv`) | Toxic / harmful instructions | Primary positive trigger set |
| **HarmBench** | Adversarial jailbreaks | Held-out generalisation test |
| **RealToxicityPrompts** (toxic-starter slice) | Toxic continuation prompts | Secondary positive set |
| **HH-RLHF** (harmless split) | Benign instructions | Negative / contrast set for patching |

We construct **matched pairs** `(p_harm, p_safe)` of approximately equal length and topic — required for clean activation patching (only the harmful content differs).

### 2.3 Refusal metric

Two metrics, computed at the **first generated token**:

1. **Refusal-token logit margin:**
   `r(x) = max(logit("I"), logit("Sorry"), logit("As"), ...) − max(logit(any non-refusal continuation))`
   (See `src/safety_circuits/refusal.py` for the token set per tokenizer.)
2. **String-match refusal rate** on a 30-token continuation: regex on "I cannot / I'm sorry / I can't / As an AI / I will not".

Metric (1) is the inner-loop signal for patching (differentiable in logit space; cheap). Metric (2) is the outer-loop human-readable confirmation.

### 2.4 Activation patching

For each (layer ℓ, head h) we:

1. Forward `p_harm` and cache all `z` (head output) activations → `C_harm`.
2. Forward `p_safe` and cache → `C_safe`.
3. Re-run `p_harm`, but **patch** `z[ℓ, h]` with the value from `C_safe[ℓ, h]` via a forward hook.
4. Measure the change in refusal-logit margin: `Δr = r_patched − r_harm`.

Large positive `Δr` → that head **caused** part of the refusal (replacing it with the safe-context value preserved refusal? or destroyed it?).
We also patch:
- entire MLP outputs per layer,
- residual stream at each layer (coarse causal trace),
- attention pattern only (vs value-weighted output).

This gives us a (n_layers × n_heads) heatmap per model.

### 2.5 Ablation

For the top-K components by |Δr|:
- **Zero-ablate**: replace activation with zeros.
- **Mean-ablate**: replace with the mean activation over a benign-prompt batch (Wang et al. — better controls for the "removing all signal" confound).
- Measure (a) refusal rate on a held-out toxic split, (b) perplexity on WikiText-2 to confirm the model didn't just break.

### 2.6 Cross-model replication

Repeat 2.4 and 2.5 on Phi-3. Compare:
- Relative layer depth of top components (e.g., "60–80% depth" rather than absolute layer).
- Attention-vs-MLP ratio of the circuit.

---

## 3. Experiments

| # | Notebook | Question | Decision |
|---|---|---|---|
| E1 | `01_setup_and_smoke_test.ipynb` | Does the env load both models and reproduce one refusal end-to-end? | Go / no-go on environment |
| E2 | `02_data_pipeline.ipynb` | Do we have matched (harm, safe) pairs with reasonable length parity? | Lock the dataset |
| E3 | `03_refusal_signal.ipynb` | Is the refusal-logit metric monotone in actual refusal? Per-model calibration. | Choose threshold |
| E4 | `04_activation_patching.ipynb` | Which heads/MLPs cause refusal on TinyLlama? Heatmap. | Identify top-K candidates |
| E5 | `05_ablation_study.ipynb` | Does ablating top-K collapse refusal? Replicate on Phi-3. | H3 + H4 verdict |

---

## 4. Success criteria

Submission-grade if:

- E3 produces a refusal metric with ≥ 90% agreement vs human-judged refusals on a 50-prompt audit.
- E4 produces a (layer, head) heatmap where the top-10 components account for ≥ 50% of the total `|Δr|` mass (sparsity).
- E5 shows: zero-ablating top-10 drops refusal on held-out toxic prompts to ≤ 30%, while WikiText perplexity changes by ≤ 5%.
- At least the *qualitative* picture (depth band + attn/MLP split) replicates on Phi-3.

Failure modes that are still publishable findings:

- Refusal is **not** sparse (diffuse across many heads) → counter-evidence to the "safety switch" narrative, still a paper-worthy result.
- Refusal **transfers between models** at the depth level but not the index level → suggests a universal motif.

---

## 5. Timeline (12 weeks, indicative)

| Week | Milestone |
|---|---|
| 1 | Skeleton + Docker + Colab setup green. Smoke tests pass. |
| 2 | Data pipeline locked. Matched pairs released as `data/processed/pairs.jsonl`. |
| 3 | Refusal metric calibrated on TinyLlama. E3 figure. |
| 4–5 | E4 patching heatmap on TinyLlama. First "candidate safety heads" list. |
| 6 | Mean-ablation + perplexity controls. E5 results on TinyLlama. |
| 7–8 | Phi-3 replication. Comparison figure. |
| 9 | HarmBench (jailbreak) stress test on identified circuits — do they still fire? |
| 10 | Writeup draft. |
| 11 | Iterate, address gaps. |
| 12 | Final report + presentation. |

---

## 6. Out of scope

- Editing weights / steering vectors / SAEs (would be a natural follow-up; explicitly future work).
- Multi-axis safety (deception, bias, PII) — proposal commits to **toxic language** axis only.
- Models > 4B parameters — compute budget rules out.
- Training new probes — only forward-pass interventions.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| TinyLlama barely refuses → no signal to trace | Switch positive trigger set to AdvBench + HarmBench (stronger jailbreaks); validated during E3. If TinyLlama is truly under-aligned, swap to Qwen2.5-1.5B-Instruct. |
| Phi-3 not natively supported in TransformerLens | Load via custom `HookedTransformer.from_pretrained_no_processing` + manual state-dict mapping; documented in `src/safety_circuits/models.py`. |
| Colab session timeouts during long patching sweeps | Stream activation caches to Drive; resume from last (layer, head) checkpoint. |
| Refusal metric is noisy | Two-metric agreement gate (logit + regex); 50-prompt human audit. |

---

## 8. Deliverables

- This repo, reproducible end-to-end on a Colab Pro A100.
- A 4-page lab report following the course format.
- The mechanistic map figure (the centrepiece slide of the final talk).
- An `artifacts/safety_heads.json` listing the top-K causal components per model.
