# Talking points — Final presentation

**Format:** **10-min talk (strictly enforced) + 5-min Q&A** · **12 main slides** + an
**appendix** of 8 backup slides · target ~9:15 to leave a safety margin.
These are **cue bullets, not a script** — one idea + the number to land per slide.
Weighting: Act 1 (mapping) is fast setup; **Act 2 (editing/jailbreak) is the payoff**.

*The one thread to keep pulling:* **"you can't delete refusal without breaking the model —
but you can retrain the same heads into a jailbreak, using no harmful data."**

*Delivery reminders (instructor guidance):* less is more — don't read the slides; keep a
less-familiar audience in mind; **stand, move, keep eye contact**; memorise the first two
sentences for a confident start.

---

## Slide 1 — Title (~10s)
- Name, course, one line: *"mechanistic interpretability of refusal — and what happens when
  you try to edit it."*

## Slide 2 — The story in two acts (~30s)
- Set the frame: **Act 1 = map the refusal circuit; Act 2 = edit it.**
- The bridge is *"concentrated but not modular."* Act 2 is the new work — spend the time there.

## Slide 3 — Why look inside? + RQ (~45s)
- Refusal is a black box; jailbreaks still get through.
- Mech-interp reads the **causal** wiring, not correlation.
- Three questions: **point to it / remove it / same across families?**
- *(Hypotheses H1–H4 are in the appendix — don't read them here.)*

## Slide 4 — Method (one conceptual slide) (~50s)
- Walk the pipeline: matched pairs → patching → top-10 → ablate → **refusal rate + perplexity**.
- *Land the crux:* **the perplexity control is the whole ballgame** — without it, "0% refusal"
  is a broken model, not a win.
- One line on the roster: 9 models, 3 families; *why these* is in the appendix.

## Slide 5 — Act 1: concentrated, not modular (~55s)
- Sparse & causal: one dominant head, **9/9** (Qwen3 L0H3 = 8.87).
- *Point at the scatter:* the **4 models at 0% refusal = the 4 with the PPL blow-up.**
- *Land:* Qwen2.5 **×61,000** = gibberish → **H3b falsified.** Concentrated, not modular.

## Slide 6 — Act 2: delete → edit (~35s)
- Ablation could only *delete* — and deletion breaks the model.
- Introduce the **scalpel axis:** blunt ablation → steering → **head-restricted LoRA.**
- Flag: this is the **one deliberate exception** to "no training."

## Slide 7 — Finding E: editable under retraining (~55s)
- LoRA → **0% refusal on all 9**; small cost on **6/9** — *where ablation gave ×128–×61,000.*
- Steering never cleanly removes (0/9).
- *Honesty beat:* not universal — Gemma-1 costs ×10 PPL. Say it.

## Slide 8 — Finding F: the obvious jailbreak is FAKE (~60s)
- Naive edit *looks* like a win: **raw ASR ~0.90.**
- *Point at the bubble:* "Sure, here is …" then **`<eos>`** — it stops.
- *Land:* **substantive-ASR = 0** on all 9. Report `substantive_asr`, not raw ASR.

## Slide 9 — Finding G: the real, benign-data jailbreak (~65s)  ← the peak
- Teach the heads to answer **benign** questions → transfers to harmful.
- *Land slowly:* **Llama-3.2-3B: 0.08 → 0.41**, 5×, non-overlapping CIs, **no harmful data.**
- Largest **where alignment was strongest** — the concerning direction.

## Slide 10 — Finding G: concentration & tuning (~55s)
- **Where:** procedural harm — cybercrime **0.84**, illegal **0.47**.
- **How far:** tunes to **0.48**; lever = **fewer** heads (top-10 ≫ top-20).
- **Negative result:** routes interfere — benign+steering 0.24 < 0.41.

## Slide 11 — Did we jailbreak them? (~50s)
- Say it plainly: **yes — on the well-aligned models, measured.**
- Three honesty nuances: fake opener / **no harmful data** / largest where alignment was strongest.

## Slide 12 — Conclusion (~50s)
- Three takeaways: concentrated-not-modular / depth+migration / **editable → benign jailbreak.**
- **Methodological lesson:** capability control + read the content, not the opener.
- Ethics in one breath (benign-only, severe excluded, no weights). Future: **distribute refusal.**
- "Thank you — questions welcome."

**Running total ≈ 9:20** — leaves ~40s buffer under the strict 10:00.

---

# Using the appendix (backup slides) in Q&A

You have 8 backup slides after the "Appendix" divider. Jump to them if asked:
- **"Why not GPT-4 / newer models?"** → *Appendix: why these nine models* (TransformerLens
  support + T4 compute; Phi-3 garbage logits, TinyLlama under-refuses).
- **"How sparse, really?"** → *Appendix: refusal is sparse* (fig_sparsity, ±95% CI).
- **"Does the circuit move / what's Finding C?"** → *Appendix: depth & migration.*
- **"How robust to jailbreaks?"** → *Appendix: jailbreak robustness (Finding D).*
- **"Full hypothesis verdicts?"** → *Appendix: scorecard* (H1–H4 + E/F/G/steering).
- **"All nine models' numbers?"** → *Appendix: cross-model table (N=50).*
- **"How did you tune it / leakage?"** → *Appendix: editing detail* (head count, target
  length, the airtight 3-way split).

---

# Q&A prep (verbal answers)

- **Why these 9 / why not newest?** Two gates: TransformerLens `HookedTransformer` support
  (needed to hook activations) and single-T4 compute (≤3B, sequential). Phi-3 → garbage
  logits, TinyLlama under-refuses, Falcon3/OLMo-2/Gemma-4 unsupported. Within-family sweeps
  were the point — they show the circuit migrate. *(→ appendix slide.)*
- **Why is benign-only training the scary part?** No harmful example exists to screen out.
  The heads gate the *decision to answer*; unlock answering benignly and harmful comes along.
- **Why substantive-ASR, not ASR?** Raw ASR counts the affirmative opener as success — reads
  ~0.90 with zero harmful content. Substantive-ASR (harmful AND ≥200 chars) is honest.
- **Why the perplexity control?** It's the difference between "safety removed" and "model
  broken." Four models hit 0% refusal only by blowing PPL up ×100s–×1000s.
- **Gemma generational confound?** Yes — Gemma-3 is 1B vs 2B (no 2B tier). The depth
  migration is robust; stated as a caveat.
- **Is this responsible?** Small open-weight academic models; no harmful training data;
  severe categories excluded; aggregate ASR only, never stored; no weights released. Point is
  a *defense* direction — distribute refusal so it can't be head-edited out.
- **Does it scale past 3B?** Unknown — explicit future work, now feasible on Bender.
- **Steering vs LoRA?** Different points on the scalpel axis; steering never cleanly removes
  (0/9), and combining the two *interferes* rather than stacks.
