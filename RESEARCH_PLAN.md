# Research plan — Mechanistic Interpretability of AI Safety Guardrails

> Pranav Yadav · MA-INF 4330 · University of Bonn
> Working title: *Refusal circuits are concentrated but not modular.*
>
> **Updated 2026-06-07 to reflect the executed study** (models, method, verdicts). The original a-priori
> plan — TinyLlama/Phi-3, Colab A100, 4-page report — is preserved in git history. Results live in
> `FINDINGS.md`; figures in `paper/figures/`; draft in `paper/paper.md`.

---

## 1. Research question

**Are there a small number of attention heads in a safety-tuned small LM that causally produce its
refusal of harmful prompts — and if so, can we identify them by activation patching, confirm their role
by ablation, and characterise how the circuit varies across model families and generations?**

Hypotheses & **verdicts** (9 instruct models: Qwen 1.5/2/2.5/3, Gemma 1/2/3, Llama-3.2 1B/3B):

- **H1 (sparse):** ✅ **Confirmed.** A dominant head + short tail carries most of the refusal-logit margin in all 9 (e.g. Qwen3 L0H3 = 8.87 ± 2.13).
- **H2 (causal):** ✅ **Confirmed.** Patching a single head harmless→harmful flips the refusal logit in all 9.
- **H3 (ablation removes refusal):** 🟡 **Partial** — top-10 zero-ablation drops refusal to ≤30% in 5/9 (Qwen2/2.5/3, Gemma-3, Qwen1.5).
- **H3b (…while preserving capability, ΔPPL ≤ 5%):** ❌ **Falsified — and this is the headline.** Every model that *fully* removes refusal also suffers catastrophic perplexity blow-up (Qwen ×128–×61,000; output is gibberish, not compliance). **Refusal is causally concentrated but not modular** — the heads that gate it are load-bearing for general generation.
- **H4 (cross-model structure):** 🟡 **Richer than predicted.** Sparsity + causality are universal, but circuit *location* is not fixed — it **migrates across generations** (Gemma L0→L13→L24; Qwen mid→L0), and **modularity scales with depth** (late-layer circuits, e.g. Gemma-3 L24, are the most cleanly removable).

The lab's central claim — that "safety" is mechanistic — holds, but the sharper, more useful result is
that it is *entangled*: you cannot excise refusal without collateral capability damage, and how cleanly
you can depends on where in the network it sits.

---

## 2. Method

Frozen models. Forward passes + hooks. No gradient updates anywhere.

### 2.1 Setup

- **Models (9, three families × generations/sizes), all in TransformerLens `HookedTransformer`:**
  Qwen1.5-1.8B-Chat · Qwen2-1.5B · Qwen2.5-1.5B · Qwen3-1.7B · Gemma-1-2B · Gemma-2-2B · Gemma-3-1B ·
  Llama-3.2-1B · Llama-3.2-3B. *(The proposal's TinyLlama/Phi-3 were dropped — TinyLlama under-refuses and
  neither is in the pinned TL `OFFICIAL_MODEL_NAMES`; Falcon3/OLMo-2 likewise unsupported. Documented
  exclusions in `FINDINGS.md`.)*
- **Compute:** single **Kaggle T4** GPU; one model per session (RAM-bound), orchestrated by `kaggle/run_experiment.py`.
- **Determinism:** temperature 0, fixed seed (0), greedy decode for the first generated token. **N = 50** matched pairs per model.

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

For the top-K (=10) components by |Δr|:
- **Zero-ablate**: replace activation with zeros.
- **Mean-ablate**: replace with the mean activation over a benign batch (Wang et al. — controls for the "removed all signal" confound).
- Measure (a) refusal rate on a held-out toxic split, (b) **WikiText-2 perplexity** (capability control), and (c) a **K-sweep** (K=5…40). **The perplexity control is essential** — it is what reveals that refusal removal coincides with model breakage (H3b).

### 2.6 Cross-model & generational analysis

Run 2.4–2.5 on all 9 models and compare **within families across generations**:
- **Relative layer depth** of the top head (normalised layer ÷ depth) — tracks the generational *migration* (Finding C).
- **Removability vs capability cost** (Δrefusal vs ΔPPL) — the coupling that falsifies H3b (Finding A).
- Extra probes: **HarmBench jailbreak** stress test (does the circuit still fire / margins flip?), **last-token** vs position-agnostic patching, **attention-pattern** vs `z` patching, and an **RTP continuation-toxicity** probe (does ablation transfer to toxic generation?).

---

## 3. Experiments

The notebooks `01–06` carry the executable thread (setup, data pipeline, refusal metric, patching,
ablation, metric audit); the production runs are driven by the multi-model orchestrator
`kaggle/run_experiment.py`, which executes the full per-model suite (patch sweep + heatmap, zero & mean
ablation + perplexity, K-sweep, last-token & attention-pattern sweeps, HarmBench jailbreak, RTP probe).
Outcome per model: §2's metrics + artifacts in `results/kaggle_neo/<model>/`.

---

## 4. Success criteria — outcome

- **Sparsity (H1):** ✅ a single head dominates in all 9 models (top head ≫ tail).
- **Causality (H2):** ✅ patching flips the refusal logit in all 9.
- **Ablation (H3):** 🟡 refusal ≤30% under top-10 zero-ablation in 5/9.
- **Capability (H3b):** ❌ the ≤5% perplexity criterion is **not met by any model that removes refusal** — full removal ⇒ PPL ×100–×60,000 (gibberish). **This is the central finding, not a failure:** the pre-registered "publishable failure mode" (refusal is *not* a clean removable module) is exactly what we observe.
- **Cross-model (H4):** circuit *location* migrates across generations (Gemma L0→L13→L24; Qwen mid→L0); modularity scales with depth.
- **Metric validation:** dual logit+regex metric; the optional 50-prompt human audit (`06_metric_audit.ipynb`) remains available to report ≥90% agreement.

Pre-registered failure modes (from the original plan) that materialised as the contribution:
- Refusal is sparse/causal **but not modular** → counter-evidence to a clean "safety switch."
- Refusal structure **transfers at the depth level** within families but *drifts* across generations → a moving target.

---

## 5. Status & timeline

The experimental programme (skeleton → data pipeline → refusal metric → patching → ablation +
perplexity → jailbreak/RTP → 9-model cross-model/generational sweep → figures) is **complete** as of
2026-06-07. Remaining work is the paper and presentations. The live, dated forward plan is in
**`PROJECT_PLAN.md` (Part C/D)**: midterm slides (19/06) → finish paper → final presentation (24/07) →
submit (31/08).

---

## 6. Out of scope

- Editing weights / steering vectors / SAEs (would be a natural follow-up; explicitly future work).
- Multi-axis safety (deception, bias, PII) — proposal commits to **toxic language** axis only.
- Models > 4B parameters — compute budget rules out.
- Training new probes — only forward-pass interventions.

---

## 7. Risks & mitigations — resolved

| Risk | Outcome |
|---|---|
| TinyLlama barely refuses | ✅ Swapped to the Qwen/Gemma/Llama instruct families (under-refusal confirmed). |
| Phi-3 not natively supported in TransformerLens | ✅ HF-port produces garbage logits (combined-QKV mis-map); **excluded** with a coherence-check diagnostic. |
| Kaggle session limits on long sweeps | ✅ One model per session; per-model results flushed + zipped; resumable via `SC_MODELS`/`SC_SKIP_EXISTING`. |
| Refusal metric is noisy | ✅ Dual logit+regex metric; per-prompt continuations saved (`*_examples.jsonl`); optional 50-prompt human audit harness. |
| System-RAM OOM loading 2B fp32 / dual-copy ports | ✅ HF-port + `low_cpu_mem_usage`; fp16 where VRAM-bound (Gemma-2). |

---

## 8. Deliverables

- This repo, reproducible on a single Kaggle T4 (one model per session). ✅
- A ~8-page conference/workshop-style paper (`paper/paper.md` — Results drafted). 🟡 in progress.
- The mechanistic-map figures (`paper/figures/`: per-head heatmaps, removal-vs-ΔPPL coupling, generational migration, jailbreak slope). ✅
- `<model>_safety_heads.json` (top-K causal heads) per model in `results/kaggle_neo/`. ✅
- Midterm (19/06) + final (24/07) presentations. ◻ pending.
