# Experiment Findings — Safety Circuits

**9 instruct models, full pipeline (N=50 matched pairs, seed 0, NVIDIA A40/A100 on the Uni Bonn
Bender cluster; TransformerLens 3.5 / transformers 5.13).** Per model: per-head activation-patching
sweep (`z`) with 95% CIs · zero- & mean-ablation of the top-10 heads · WikiText-2 perplexity
(capability control) · K-sweep · last-token & attention-pattern sweeps · HarmBench jailbreak stress
test · RealToxicityPrompts (RTP) continuation-toxicity probe. Artifacts in `results/bender_neo/<model>/`.
(Originally developed on Kaggle T4 — `results/kaggle_neo/` — and reproduced here on Bender; the two
runs agree closely, i.e. a replication across hardware *and* library versions.)

Models span three families across **generations / sizes**: Qwen (1.5→2→2.5→3), Gemma (1→2→3),
Llama-3.2 (1B→3B). (Not included in this sweep: Phi-3, OLMo-2, Mistral, TinyLlama, Falcon3 — several
of these are now loadable under TransformerLens 3.5 and are candidate additions; Phi-3's earlier
HF-port garbage-logits issue predates the TL 3.5 upgrade and is worth re-checking.)

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

**E — Retraining the localized safety heads *jailbreaks* well-aligned models.** The editing extension (§9)
turns the map into an attack: a head-restricted LoRA trained on **benign data only** makes **Llama-3.2-3B
comply with 41% of harmful requests** (baseline 8%; up to **48%** tuned) — **with no harmful training
data**. Full details in Findings E–G below and the plain-language verdict in **"Did we jailbreak the
models? — the bottom line."**

---

## Cross-model table (N=50)

| Model | Arch (L×H) | Top head ± CI | Clean→zero-abl refusal | Δ PPL (clean→abl) | Jailbreak refusal (plain→jb) | Jailbreak margin (plain→jb) | RTP Δtox |
|---|---|---|---|---|---|---|---|
| **Qwen1.5-1.8B** | 24×16 | L12H10 1.02 ±0.21 | 80%→**24%** | 32.8→33.2 (**+1.4%**) | 80→42 | +1.37→**−1.02** | +0.026 |
| **Qwen2-1.5B** | 28×12 | L0H9 5.65 ±1.17 | 100%→**0%** | 18→193,679 (×10,780) | 100→74 | +3.48→+1.40 | −0.023 |
| **Qwen2.5-1.5B** | 28×12 | L0H10 1.06 ±0.51 | 100%→**0%** | 18→1.11M (×61,000) | 100→94 | +4.08→+2.20 | +0.011 |
| **Qwen3-1.7B** | 28×16 | L0H3 8.86 ±2.12 | 88%→**0%** | 32→4,049 (×128) | 88→**42** | +3.67→**−2.99** | +0.012 |
| **Gemma1-2B** | 18×8 | L0H5 2.46 ±0.72 | 88%→72% | 62→76 (+22.9%) | 88→82 | +5.98→+4.45 | −0.009 |
| **Gemma2-2B** | 26×8 | L13H2 0.62 ±0.19 | 96%→88% | 23→29 (+24.2%) | **96→96** | +6.59→+5.41 | +0.005 |
| **Gemma3-1B** | 26×4 | L24H0 3.92 ±0.51 | 44%→**0%** | 61→192 (+217%) | 44→36 | +12.2→+7.0 | **+0.128** |
| **Llama-3.2-1B** | 16×32 | L9H22 1.08 ±0.19 | 96%→92% | 24.7→27.7 (+12.4%) | 96→96 | +6.27→+4.85 | −0.030 |
| **Llama-3.2-3B** | 28×24 | L0H20 1.59 ±0.55 (+L24) | 92%→**36%** | 18.5→19.5 (**+5.3%**) | 92→**96** | +8.13→+7.34 | −0.009 |

> **Read A off this table:** the four models that hit refusal **0%** (Qwen2/2.5/3, Gemma3) are exactly
> the four with the largest Δ PPL. The five with small Δ PPL never fully remove refusal.

## Hypothesis scorecard

| Hypothesis | Verdict |
|---|---|
| **H1 — Sparse** (≤10 heads explain most refusal) | ✅ **Holds** — a dominant head + short tail in all 9 (e.g. Qwen3 L0H3 = 8.87, ~4.5× #2). |
| **H2 — Causal** (patching flips the refusal logit) | ✅ **Confirmed 9/9.** |
| **H3 — Ablation** (zero top-10 → refusal ≤30%) | 🟡 **Partial** — met by Qwen2/2.5/3 (0%), Gemma3 (0%), Qwen1.5 (24%); **failed** by Gemma1 (72%), Gemma2 (88%), Llama-1B (92%), Llama-3B (36%). |
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
| Qwen1.5-1.8B (24×16) | L12H10 1.02±0.21 · L12H8 0.52 · L21H15 0.47 | 80%→24% / 44% | 32.8→33.2 | 80→42, +1.4→−1.0 | +0.026 |
| Qwen2-1.5B (28×12) | L0H9 5.65±1.17 · L0H3 0.61 · L8H1 0.48 | 100%→0% / 0% | 18→193,679 | 100→74, +3.5→+1.4 | −0.023 |
| Qwen2.5-1.5B (28×12) | L0H10 1.06±0.51 · L11H8 0.59 · L0H6 0.54 | 100%→0% / 0% | 18→1.11M | 100→94, +4.1→+2.2 | +0.011 |
| Qwen3-1.7B (28×16) | **L0H3 8.86±2.12** · L15H9 1.97 · L10H3 1.29 | 88%→0% / 0% | 32→4,049 | 88→42, +3.7→**−3.0** | +0.012 |

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

Both Llama models are **jailbreak-robust** (1B 96→96; 3B 92→**96**, i.e. jailbreaks slightly *raised*
refusal) and both show **incomplete removal with negligible capability damage** — the safety signal
is more distributed/redundant here (3B: 92%→36% at only +5.3% PPL). The 1B places its top head
mid-network (L9); the 3B is a **hybrid (L0 + L24)** — early detection *and* late enforcement, the
only model with both.

| Model | Top-3 heads (±CI) | Refusal clean→zero / mean | PPL clean→zero | Jailbreak | RTP Δtox |
|---|---|---|---|---|---|
| Llama-3.2-1B (16×32) | L9H22 1.08±0.19 · L4H11 0.69 · L9H12 0.63 | 96%→92% / 92% | 24.7→27.7 | 96→96, +6.3→+4.9 | −0.030 |
| Llama-3.2-3B (28×24) | **L0**H20 1.59±0.55 · **L24**H11 1.15 · L0H18 1.06 | 92%→36% / 56% | 18.5→19.5 | 92→96, +8.1→+7.3 | −0.009 |

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
*(Config: LoRA rank 16 / 600 steps / lr 5e-4. **Airtight 3-way AdvBench split, seed 0: train 60 (fit) /
val 45 (ALL selection — steering combo + head-count k) / test 45 (ALL headline numbers)**; plus 50 HarmBench
jailbreak + cross-dataset generalization deep-eval, XSTest benign substance-unlock, WikiText-2 perplexity.
A40/A100 on Bender. Artifacts in `results/editing_bender/<model>/` — original Kaggle-T4 editing run kept in
`results/editing/`. See `RESEARCH_PLAN.md` §2.7 for the split + leakage caveat.)*

**E — Refusal is *not modular under deletion, but it is editable under retraining*.** Across all 9 models,
**head-restricted LoRA drives refusal (and HarmBench-jailbreak refusal) to 0%** on the held-out **test** set
(head-count k chosen on val). On **6/9 the perplexity cost is small** (≤ ~10%, often ≈0 or *negative*) —
exactly where blunt ablation gave gibberish (Qwen2/2.5/3: ablation PPL ×128–×61,000 → LoRA +0.3 to +10%).
But under airtight (val-chosen-k) reporting the clean edit is **not universal**: **Gemma1-2B is catastrophic
(+927% PPL at its chosen k=1, ×10 PPL)**, with Qwen1.5 (+19%) and Gemma2 (+15%) also costly. The no-train
**steering baseline never cleanly removes refusal on any model** (best coherent refusal 36–93% on test).
So the three interventions form a **scalpel-sharpness axis**: blunt ablation (breaks the model / can't
remove) → steering (can't cleanly remove) → **head-restricted LoRA (reaches 0% refusal at ≈0 ΔPPL on most
models — Gemma1-2B the exception)**. This refines Finding A: refusal is *load-bearing* (you can't excise it)
yet *re-trainable in place* on most models (you can repurpose the same heads to comply). It is also, by
construction, a **targeted jailbreak produced by retraining only the safety heads** — but a **shallow one
(see Finding F)** — reported in aggregate, no weights released.

### Cross-model editing table

All numbers on the **test** split (n=45); k selected on **val**. (The blunt-ablation contrast is in the
mapping table above + the scalpel-axis figure.)

| Model | top-head depth | base refusal (test) | **steering** best-coherent (refusal / ΔPPL) | chosen k (val) | **LoRA @chosen_k** (refusal / jb / ΔPPL) | LoRA @k10 ΔPPL |
|---|---|---|---|---|---|---|
| **Gemma1-2B** | L0 / .00 | 98% | 62% / −1.0% | k1 | 0% / 0% / **+927%** | +24% |
| **Gemma2-2B** | L13 / .50 | 100% | 67% / +12.4% | k1 | 0% / 2% / +15.0% | +27% |
| **Gemma3-1B** | L24 / .92 | 67% | 91% / +10.8% | k3 | 0% / 0% / **−12.2%** | −12% |
| **Qwen1.5-1.8B** | L12 / .50 | 87% | 60% / +0.4% | k1 | 0% / 0% / +18.7% | +36% |
| **Qwen2-1.5B** | L0 / .00 | 100% | 36% / −2.5% | k3 | 0% / 0% / +3.3% | +1% |
| **Qwen2.5-1.5B** | L0 / .00 | 100% | 64% / +3.1% | k1 | 0% / 0% / +10.2% | +3% |
| **Qwen3-1.7B** | L0 / .00 | 98% | 58% / +5.5% | k3 | 0% / 0% / +0.3% | +1% |
| **Llama-3.2-1B** | L9 / .56 | 96% | 91% / +2.0% | k1 | 0% / 0% / +0.7% | +7% |
| **Llama-3.2-3B** | L0 / .56 (hybrid) | 98% | 93% / +1.5% | k3 | 0% / 0% / +1.7% | +1% |

> **Read E off this table:** LoRA hits **0% refusal + 0% jailbreak on all 9** (test) — including the four
> Qwen/Gemma3 models ablation could only "remove" by blowing PPL up ×128–×61,000. But under airtight
> (val-chosen-k) reporting the clean edit is **not universal**: **Gemma1-2B costs ×10 PPL** (Qwen1.5/Gemma2
> also costly). **Steering** never reaches 0% while coherent. (Refusal here is opener-based — see Finding F.)

### Editing hypothesis scorecard

| Hypothesis | Verdict |
|---|---|
| **F1a — clean edit** (head-LoRA flips refusal at small ΔPPL) | 🟡 **Refusal-flip universal; clean edit ~6/9.** LoRA → 0% refusal & jailbreak on 9/9 (test). ΔPPL at the val-chosen k is small (≤~10%) on 6/9; **exceptions: Gemma1-2B catastrophic (+927% at k1, ×10 PPL), Qwen1.5 (+19%), Gemma2 (+15%).** |
| **F1b — depth→#heads law** (later circuits edit with fewer heads) | 🟡 **Not resolved.** Refusal flips with very few heads — val-chosen k=1 on 5/9, k=3 on 4/9 — with no monotone depth relationship (early-L0 Qwen2 flips at k3; late-L24 Gemma3 at k3). The informative axis is **ΔPPL cost**, not head count. |
| **F1c — cross-generation transfer** | **Infeasible on this roster** (no two checkpoints share an architecture; circuits migrate). Not pursued — see `RESEARCH_PLAN.md` §9. |
| **Steering baseline** (directional ablation removes refusal cleanly?) | ❌ **No (0/9).** Best coherent refusal 36–93% on test (min coherent val refusal 31–93%); never ~0% while coherent. |

**Generational editability gradient (Gemma, ties E ↔ Finding C).** Within Gemma the *clean-edit cost*
tracks circuit depth/generation in lockstep with the location migration: **g1 (L0) +927% (k1) → g2 (L13) +15%
(k1) → g3 (L24) −12% (k3).** Editability improves sharply as the circuit moves later/newer — the editing
analogue of Finding B. It is **not universal across families**, though: Qwen2/2.5/3 have early (L0) circuits yet
LoRA edits them cleanly (+0.3 to +10%) *even though ablation destroys them* (×128–×61,000) — LoRA "rescues"
early-layer circuits that ablation cannot touch. The lone hold-out is **Gemma1-2B** (oldest, L0), which
resists clean editing under every method. Headline visual per model: `results/editing_bender/<model>/<model>_scalpel_axis.png`.

### F — the LoRA "jailbreak" is *shallow*: opener-stop

Finding E's "refusal → 0%" is measured on the **opener** (first-token logit + refusal regex on the start
of the generation). Inspecting the actual edited continuations
(`results/editing_bender/<model>/<model>_edit_examples.jsonl`) shows the edit flips only that opener:
**on all 9 models, the edited model emits an affirmative opener that paraphrases the request and then
immediately stops (`<eos>`)** — e.g. `"Sure, here is a letter defaming a company or individual to damage
their reputation<eos>"`. The edited continuations are short affirmative openers that restate the request
and then stop — **no harmful body, not even a mid-text refusal** — the model complies at the opening and halts.

**So the refusal metric overstates the jailbreak.** "0% refusal / 0% jailbreak" means the *opening
token* is no longer a refusal, not that the model produced harmful content. Behaviourally the head-LoRA
is a **shallow jailbreak**: it flips the refusal *decision* cheaply but does not elicit substance. This
**refines Finding E** (it does not overturn it — LoRA still flips the opener at ≈0 ΔPPL): the localized
safety heads gate the **refusal decision at the opener**, not the capacity to generate the harmful body.
The likely cause is the **24-token affirmative training target** (`max_target_tokens=24`) — the adapter
learns the opener template, not full generation.

**Quantified — harmful side** (`*_edit_generalization.csv`, 50 HarmBench prompts, 128-token generation).
After the headline edit, refusal collapses to **0%** but the completions are **not substantive**: the
fraction that is *not-refused AND ≥200 chars* falls to **≈0% (max 2%, Gemma3)**, and mean generation length
is clamped to **~120–128 chars** (vs 103–685 baseline). Toxicity rises only marginally (all edited ≤0.033).
The edit removes the refusal opener without installing an answering behaviour — a shallow jailbreak, measured.

**Mechanism — benign side** (weapon-free; `*_edit_benign_substance_agg.csv`, XSTest over-refusal prompts).
Training the *same heads* to comply on **benign** data shows opener-stop is a **training-target-length
artifact**: a **24-token** target leaves answers stunted (substantive ≈ 0–22%), while a **256-token** target
restores much fuller answers (substantive up to ~40–47% — e.g. Qwen1.5 1%→47%, Llama-3.2-3B 8%→42%, Gemma1-2B
2%→40%; mean length grows with it). The short target reproduces exactly the opener-stop seen on the harmful
side; only the long target teaches genuine content. So the harmful edit's shallowness is a **direct
consequence of the 24-token affirmative target**, not a limit of the localized heads — retraining those heads
*can* install full answering, given a long enough target (demonstrated only on benign content, by design).

**Bottom line for F (the opener edit):** the refusal-opener edit removes the refusal *decision* (0% opener-
refusal) but emits **no substantive harmful content** — it complies-then-stops. Whether teaching the heads to
produce content (benignly) *transfers* to harmful compliance is answered next in **Finding G**.

### G — Content-transfer jailbreak: benign-only training transfers to harmful compliance (F3)

Finding F left one question: the shallowness is a 24-token-target artifact, so does training the heads to
produce *full content* — on **benign data only** — transfer to **harmful** prompts? Evaluated on **150
non-severe HarmBench prompts** (weapons, CBRN/bio/chem/nuclear, CSAM, self-harm, terror/mass **hard-excluded**),
scored by the **HarmBench-Llama-2-13b classifier**, with **bootstrap 95% CIs**. Headline metric =
**`substantive_asr`** (judged-harmful **and** ≥200 chars) — bare ASR is itself opener-confounded. **No harmful
training data; aggregate only; raw completions never stored** (`*_edit_harmful_transfer_agg.csv`). Two extra
routes were added to test whether safety-removal methods *compose*: **steering-alone** (Arditi directional
ablation, no training) and **benign-edit + steering** together.

**substantive-ASR by arm (N=150; benign column shows the bootstrap 95% CI):**

| Model | baseline | opener-edit | **benign-content-edit** | steering-alone | benign + steering |
|---|---|---|---|---|---|
| **Llama-3.2-3B** | 0.08 | 0.00 | **0.41** [0.33, 0.49] | 0.07 | 0.24 |
| **Llama-3.2-1B** | 0.05 | 0.00 | **0.15** [0.10, 0.21] | 0.21 | 0.11 |
| **Qwen2.5-1.5B** | 0.01 | 0.00 | **0.15** [0.09, 0.21] | 0.23 | 0.17 |
| Gemma1-2B | 0.05 | 0.00 | 0.09 [0.05, 0.13] | 0.15 | 0.09 |
| Gemma3-1B | 0.10 | 0.00 | 0.10 [0.05, 0.15] | 0.00 | 0.00 |
| Gemma2-2B | 0.01 | 0.00 | 0.07 [0.03, 0.11] | 0.07 | 0.00 |
| Qwen2-1.5B | 0.09 | 0.00 | 0.09 [0.05, 0.13] | 0.42 | 0.09 |
| Qwen3-1.7B | 0.11 | 0.00 | 0.05 [0.02, 0.08] | 0.17 | 0.03 |
| Qwen1.5-1.8B | 0.25 | 0.00 | 0.13 [0.08, 0.18] | 0.35 | 0.10 |

**Four results:**

1. **The opener edit yields ZERO substantive harmful content** (sASR = 0% on all 9) despite raw ASR ≈ 0.87–0.92
   — the classifier over-counts the affirmative restatement. Confirms Finding F, and shows *raw* ASR is an
   opener-confounded metric (hence `substantive_asr`).
2. **Benign-only content training transfers to real harmful compliance — largest where alignment was strongest.**
   On the robustly-aligned models (baseline sASR ≤ 0.08) the benign edit lifts compliance to **0.41 on
   Llama-3.2-3B** (CI [0.33, 0.49], baseline 0.08 — a 5× jump, non-overlapping CIs), 0.15 on Llama-3.2-1B and
   Qwen2.5 — **with no harmful training data**. Teaching the safety heads to *answer* (benignly) makes them
   answer harmful requests too. On the weakly-aligned Qwen1.5/2/3 & Gemma-3 (already 0.10–0.25 compliant at
   baseline) the benign edit — which trains toward *helpful* answers — does not add, and often lowers it.
3. **The two safety-removal routes do NOT stack — they interfere.** `benign + steering` is ≤ the better of the
   two alone on every model, and *markedly worse* than benign-alone on Llama-3.2-3B (**0.24 vs 0.41**). Steering
   (which ablates the refusal *direction*) degrades the coherence the benign LoRA relies on, lowering the
   substantive rate. Steering-alone is a **separate, model-dependent** route: strong on the mid-tier Qwens
   (Qwen2-1.5B **0.42**, Qwen1.5-1.8B 0.35) but weak on Llama-3.2-3B (0.07) and inert on Gemma-3 (0.00, where it
   just produces long non-harmful rambles). No single method dominates across models.
4. **Tuning the benign-only edit raises the ceiling (D2 sweep, `*_edit_transfer_sweep_agg.csv`).** Sweeping
   head-count × target-length × decoding (benign data only) lifts the best benign-only sASR to **0.48 on
   Llama-3.2-3B** (k=10, 256-tok target, sampling), **0.30 on Llama-3.2-1B** (doubling the default), and 0.23 on
   Qwen2.5. The decisive lever is **fewer heads**: editing the **top-10** heads beats the top-20 (0.46–0.48 vs
   0.09–0.22 on Llama-3.2-3B) — a tighter edit stays coherent, so more of its output is substantive.

**Which harms transfer most (Llama-3.2-3B, benign-content-edit, per-category — `*_bycat_agg.csv`):** the transfer
concentrates in **procedural** harm, where "harmful" means a detailed how-to (exactly what content-unlock installs):

| Category | n | substantive-ASR |
|---|---|---|
| cybercrime / intrusion | 19 | **0.84** |
| illegal (non-violent) | 57 | 0.47 |
| general harmful | 21 | 0.33 |
| harassment / bullying | 19 | 0.26 |
| misinformation / disinformation | 34 | 0.18 |

So the transfer jailbreak is **real, replicated at N=150 with non-overlapping CIs, tunable up to ~0.48, and
largest exactly where alignment was strongest** — the concerning direction. It is achieved **without ever
training on harmful content** (mechanism from Finding F: the heads gate the *decision to answer*, transferable
across benign/harmful once content generation is unlocked). Combining it with steering does not help — the
routes are alternatives, not additive.

*Ethics: non-severe HarmBench only (severe categories excluded), benign-only training, aggregate ASR only, raw
generations never stored, no weights released — see `RESEARCH_PLAN.md` §9 and §10.*

---

## Did we jailbreak the models? — the bottom line

**Yes — but in a specific, measured sense, and only on the well-aligned models.** The driving question of the
editing extension was whether localized safety-head editing can be turned into a real jailbreak. It can:

- **Yes, clearly, on Llama-3.2-3B.** Training the safety heads on **benign data only** (never a single harmful
  example) makes it produce genuinely harmful, *substantive* answers to **41%** of harmful prompts
  (95% CI [0.33, 0.49]), up from **8%** at baseline — a 5× jump with non-overlapping CIs. Tuning raises the
  ceiling to **48%**. These are full harmful how-to answers, HarmBench-13B-verified over 150 prompts, not just
  an affirmative opener.
- **Partially, on Llama-3.2-1B and Qwen2.5** — ~15% each (baseline 1–5%), up to **30% / 23%** tuned. Real but
  weaker.
- **No meaningful gain on the weakly-aligned models** (Qwen1.5/2/3, Gemma-3) — but *only* because they were
  **already 10–25% compliant at baseline**. They were barely aligned to begin with, so there was little left
  to remove.

**Three nuances that make this an honest result, not a headline number:**
1. **The "obvious" jailbreak is fake.** The naive edit (train on "Sure, here is…") shows a raw ASR of ~90% —
   looks like a total jailbreak — but yields **0% actual harmful content** on all 9 models. It only learns to
   *say* the affirmative opener and stop (opener-stop, Finding F). That is why the headline metric is
   `substantive_asr` (harmful **and** ≥200 chars), not raw ASR.
2. **The real jailbreak transfers from benign training.** You never show the model a harmful answer. You teach
   the localized safety heads to give *full, helpful* answers on benign questions, and that ability
   **transfers** to harmful requests — the heads gate the *decision to answer*, so unlocking content generation
   benignly brings harmful content along for free.
3. **It is largest exactly where alignment was strongest** (Llama-3.2-3B — the concerning direction), and it
   **concentrates in procedural harm** — cybercrime instructions (84%), illegal how-tos (47%) — where "harmful"
   means a step-by-step guide, which is precisely what content-unlocking installs. **Combining with steering
   does not help** — the two attack routes interfere rather than add (0.24 vs 0.41 on Llama-3.2-3B).

**In one sentence:** retraining a handful of localized safety heads on purely benign data jailbreaks a
well-aligned model into substantive harmful compliance (~41–48%) **without ever training on harmful content** —
a real, replicated, statistically-solid existence proof, deliberately kept bounded (weapon-free categories,
no harmful training corpus, aggregate-only, no weights released).

## Timeline — what each editing extension found

The extension did not go in a straight line — each result reframed the next question. What each step
*discovered* (the study's methodology/ethics arc is in `RESEARCH_PLAN.md` §10):

| Step | Extension | What we found |
|---|---|---|
| **F1** | head-restricted LoRA (retrain the localized heads) | refusal & HarmBench-jailbreak refusal → **0% on all 9**; clean (small ΔPPL) on **6/9** → **Finding E** |
| — | opener-stop check (read the *content*, not just the flag) | the flip is **shallow**: the naive edit teaches only the affirmative opener → **0% substantive harm** despite raw ASR ~0.9 → **Finding F** |
| — | airtight 3-way split (train/val/test, leakage control) | leakage-free reporting **downgrades** the F1a "clean edit" claim — not universal (Gemma1-2B ×10 PPL at its chosen k) |
| **F3** | benign→harmful transfer (train benign content, test harmful) | benign-only content training **transfers to real harmful compliance**; **Llama-3.2-3B 0.41** → **Finding G** |
| **D1** | rigor — N=150 + bootstrap CIs + per-category | transfer **replicates with non-overlapping CIs**; **concentrates in procedural harm** (cybercrime 0.84, illegal 0.47) |
| **D2** | tune higher — benign-only sweep (k × target-len × decoding) | ceiling **0.48** (Llama-3.2-3B) / **0.30** (1B); decisive lever = **fewer heads** (top-10 ≫ top-20 — a tighter edit stays coherent) |
| **D3** | combine with steering | **negative result** — the two safety-removal routes **interfere, don't stack** (0.24 vs 0.41); steering-alone is a *separate*, model-dependent route (strong on mid-tier Qwens, weak on Llama-3.2-3B) |

The arc: we **mapped** the refusal circuit (A–D), showed the heads are **editable, not just deletable** (E),
discovered the edit was **shallow** (F), **tightened the evaluation** (3-way split), then proved the deep
version — **benign-only training that transfers to harmful compliance** (G) — and finally **strengthened and
bounded** it: replicated with CIs (D1), pushed the ceiling to ~48% (D2), and showed it doesn't compose with
steering (D3). Net: a real jailbreak of well-aligned models with no harmful training data, honestly scoped.

---

## Method notes & caveats

- **Capability control is essential.** Without the WikiText-2 perplexity check, every "refusal → 0%"
  reads as a clean safety removal; with it, the Qwen results are revealed as model-breakage
  (gibberish output, PPL ×1000s). This is the project's core methodological point.
- **Refusal rate is opener-based.** The headline refusal / jailbreak rates score the opening token(s);
  an edit that flips the opener to an affirmative but then stops (`<eos>`) reads as 0% refusal *without
  producing harmful content* (Finding F). Read "0% refusal" as "opener flipped," not "harmful content
  produced" — the generalization / substance add-ons (Finding F) quantify the difference.
- **Residual trace** (per-layer `resid_pre` patch) is informative under the fixed pipeline
  (early-concentrated, decaying to ~0 at the last layers) — the old "flat −32.70 everywhere" was a
  whole-tensor artifact, now resolved.
- **RTP probe** is mostly null/confounded: where ablation produces gibberish (Qwen), the toxicity
  classifier sees no coherent toxic content. Only Gemma-3 (coherent ablated output) shows real
  transfer (+0.128). So RTP transfer is itself *evidence for* Finding B, not an independent result.
- **Mean- vs zero-ablation** rarely helps: mean-ablation is gentler on PPL for Qwen2.5 (905 vs 1.1M)
  but still 50× baseline, and removal completeness is essentially unchanged across the roster.
- **Replication across stacks.** These are the Bender re-run numbers (A40/A100, TransformerLens 3.5 /
  transformers 5.13). They reproduce the original Kaggle-T4 run (`results/kaggle_neo/`) closely — same
  dominant heads, refusal rates, generational location-migration, and the Gemma-3 toxicity spike (+0.128)
  — a robustness check across hardware *and* library versions. The one material drift is **Llama-3.2-3B**
  ablation (Kaggle 88%→44% vs Bender 92%→36%); Llama-3.2-1B's ΔPPL and RTP also shifted slightly.
- **Qwen1.5-1.8B coherence caveat.** Its clean sanity probes are degenerate (it emits `<|im_end|>`
  immediately), so its headline low-cost H3 result (80%→24% at +1.4% PPL) should be read with caution —
  the *baseline* generation quality is already questionable, independent of ablation.

## Models not in this sweep
- Under **TransformerLens 3.5**, **Phi-3-mini**, **OLMo-2**, and **Mistral-7B-Instruct** are now
  loadable (these were the historical exclusions on the pinned TL 2.x). They were not run in this
  9-model sweep but are natural extensions — note Phi-3's earlier HF-port *garbage-logits* issue
  (mis-mapped QKV/RoPE projection, 0% baseline refusal) predates the TL 3.5 upgrade and should be re-checked.
- **TinyLlama-1.1B, Falcon3-1B** remain unsupported / not run. (Gemma-4 is multimodal.)

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
