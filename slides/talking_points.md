# Talking points — Midterm presentation

**Finding the Safety Switches — Inside Small Language Models**
Pranav Yadav · MA-INF 4330 · University of Bonn

> Format: **10-minute talk + 5-minute Q&A.** Read the script below — one section per slide, timed to
> hit 10 minutes across **16 slides**. Deliver the pipeline, heatmaps, full-table and scorecard slides
> briskly (~30 s) to hold time. The **Q&A-prep** section at the end is for the 5-minute discussion —
> there are no backup slides, so the answers live here. *Italic bracketed notes are delivery cues.*

---

## Slide 1 — Title  *(~20 s)*

Good [morning/afternoon] everyone. My project is called *"Finding the Safety Switches — Inside Small
Language Models."* The one-line version of the result: in small instruct-tuned language models, the
circuit that makes the model refuse harmful requests is **concentrated** — you can almost point to a
single attention head — but it is **not modular**: you can't cleanly remove it. I'll show you how we
found that, across nine models, using only forward passes — no training at all.

---

## Slide 2 — Agenda  *(~20 s)*

Here's the plan. First, why safety is worth opening up, the research question, and the method —
activation patching and ablation. Then the four findings, A through D; the headline is the second one,
**concentrated but not modular**. And I'll close with where this work goes next.

---

## Slide 3 — Why look inside a safety guardrail?  *(~55 s)*

Instruct models are aligned to **refuse** harmful requests. But we mostly treat that refusal as a black
box — and jailbreaks clearly still get through, so the box is leaky.

Classic explainability — saliency maps, attention heatmaps — only tells you what *correlates* with the
output. **Mechanistic interpretability** is different: it reads the actual wiring and asks which
components *causally* produce the behaviour.

So if "safety" is a concrete circuit, three questions become answerable. Can we **point to it** — a few
heads, or smeared across the model? Can we **remove it** — and what breaks if we do? And does it **look
the same** across families and generations?

One constraint: this is entirely **read-only**. Forward passes and hooks, using TransformerLens. No
training, no fine-tuning, no gradients.

---

## Slide 4 — Research question & hypotheses  *(~50 s)*

The formal research question, in the box: are there a **small number of attention heads** that
**causally** produce a model's refusal — and how does that circuit vary across families and generations?

We pre-registered five hypotheses. **H1**, sparse: a handful of heads explain most of the refusal. **H2**,
causal: patching those heads *flips* the refusal signal. **H3**, ablation: zeroing the top ten heads
drives refusal below thirty percent. **H3b** — the crucial one — that we can do that *while preserving
capability*, less than five percent change in perplexity. And **H4**, that the same structure recurs
across models. Keep H3b in mind — that's where the interesting result lives.

---

## Slide 5 — Method  *(~55 s)*

Here's the pipeline in words, run identically on every model. Nine instruct models, fifty matched pairs
each, one Kaggle T4, fixed seed.

**Matched pairs** — a harmful instruction from AdvBench and a benign one from HH-RLHF, matched on length
and topic, so the only real difference is the harmful content. A **refusal metric** — the logit margin
on refusal tokens at the first generated token, with a regex backstop. Then the core: **activation
patching** — for every head, splice its benign output into the harmful run and measure how the refusal
signal moves. Then **ablation** of the top ten heads. And the step that mattered most: a **capability
control**, WikiText-2 perplexity, to check the model can still write coherent text.

Studying generations *within* a family is what lets us watch the circuit move.

---

## Slide 6 — The pipeline at a glance  *(~30 s)*

Same thing as a picture — left to right. We start from two prompt sets, **harmful** and **benign**, and
form **matched pairs**. We run them through the frozen model and **cache every activation** with hooks.
Then **per-head patching** gives us a heatmap of which heads matter, and we take the **top ten**. We
**ablate** those — and crucially we read out *two* things: did **refusal** drop, *and* what happened to
**perplexity**. Putting those two together is what gives us **Findings A through D**. The whole thing is
forward passes and hooks — the model is frozen, never trained.

---

## Slide 7 — Refusal is sparse and causal  *(~40 s)*

First two hypotheses, and they both hold cleanly. This bar chart is the single most important head in
each model, with ninety-five-percent confidence intervals.

In **every** model, one head dominates and stands well clear of its error bar — the extreme case is
Qwen3, almost nine. And patching just that one head from the benign run into the harmful run flips the
refusal logit, in all nine models. So refusal is **sparse** and **causal** — H1 and H2, nine of nine.

---

## Slide 8 — Where the safety heads sit  *(~30 s)*

Here's the same result as a map. Each panel is one model; axes are **layer by head**, colour is causal
importance. You can see it directly: in every model just **one or two bright cells** light up — that's
the sparsity. But their *position* differs from model to model — that difference is the seed of
Finding C.

---

## Slide 9 — Finding A: concentrated, but not modular  *(~70 s, headline)*

This is the headline slide, so let me slow down.

We've established refusal is *causally* concentrated. The natural next step is: let's remove it. This
plot tests that. On the x-axis, how much refusal the ablation removes. On the y-axis — note it's a **log
scale** — how much perplexity blows up; the dotted line near the bottom is our five-percent capability
budget.

The two are locked together. The four models that drop to **zero percent refusal**, top-right, are
*exactly* the four whose perplexity explodes. Qwen2.5's goes up by a factor of **sixty-one thousand**.
And the output isn't compliance — it's **gibberish**. The model is broken, not jailbroken.

So **no model** removes refusal *and* keeps capability. H3b is **falsified** — and that's the
contribution, not a failure. Refusal is entangled with the very heads the model needs to generate
coherent text. "Ten heads, refusal to zero" sounds like a clean safety switch; the capability control
shows it isn't.

---

## Slide 10 — Finding B: modularity scales with depth  *(~50 s)*

So *why* are some models so much worse to ablate? It tracks **where** the head sits.

When the dominant head is **early — layer zero**, like the Qwen models, zeroing it is catastrophic:
perplexity up a hundred to sixty-thousand-fold. A layer-zero head does general work for the whole
network; refusal is just one of its jobs.

When the head is **late — layer twenty-four**, like Gemma-3, it's far gentler: refusal goes to zero at
only plus-two-hundred-percent perplexity — and it's the *only* model where ablation produces genuinely
**toxic** output, not gibberish. So the most switch-like safety circuit we found is the **late-layer**
one. Early-layer "safety" heads are really load-bearing generation heads.

---

## Slide 11 — Finding C: location migrates across generations  *(~45 s)*

This is my favourite result. If you track the depth of that dominant head *across* a family's
generations, it **moves** — consistently per family.

**Gemma** pushes safety steadily **deeper** every generation: layer zero, thirteen, twenty-four — a
clean march to the output. **Qwen** goes the *opposite* way: mid-network in 1.5, then pinned at layer
zero from Qwen2 on. And **Llama-3B** is a hybrid — both an early detection head and a late enforcement
head. So *how* a family implements refusal isn't fixed; it drifts across releases.

---

## Slide 12 — Finding D: jailbreaks weaken refusal  *(~45 s)*

Last finding — jailbreak robustness, and it's **non-monotonic**. This is the refusal margin on plain
harmful prompts, left, versus adversarial HarmBench jailbreaks, right.

Most robust is **Gemma-2** — ninety-six percent, unchanged; Llama-3B actually refuses jailbreaks
slightly *more*. But the brittle ones — **Qwen3 and Qwen1.5** — their margin **crosses below zero**,
meaning under jailbreak the model is, on average, inclined to *comply*. And **Qwen3 is more jailbreakable
than Qwen2.5** — the newest model is the least robust. Newer is not safer.

---

## Slide 13 — Hypothesis scorecard  *(~30 s)*

To summarize against what we set out to test. H1, sparse — holds, nine of nine. H2, causal — holds, nine
of nine. H3, ablation removes refusal — partial, five of nine. H3b, *and keeps capability* —
**falsified**, no model manages both. And H4 — richer than predicted: the structure recurs, but its
location migrates. The failure mode we pre-registered as a possibility *became* the main result.

---

## Slide 14 — Cross-model results at a glance  *(~25 s)*

For completeness, every model on one page. The columns are: the top head and where it sits, how far
refusal drops under ablation, the perplexity cost, and the jailbreak numbers. The one thing to read off
it: the four models at **zero-percent refusal** are exactly the four with runaway perplexity — that's
Finding A in a single column.

---

## Slide 15 — Where this goes next  *(~40 s)*

Briefly, the paths forward. To actually make refusal modular, the next step is moving from blunt ablation
to **steering vectors** and **sparse autoencoders**, to disentangle those layer-zero heads into separable
features — and to test the design lever directly: can a late-layer circuit be cut cleanly? On rigor:
causal scrubbing, a human audit of the metric, a higher-K sweep on Llama. And to broaden it: scale past
four billion parameters, more families, and beyond toxicity to deception, bias, and other languages.

---

## Slide 16 — Conclusion  *(~40 s)*

To wrap up. Three things. One: refusal is **sparse and causal** — one dominant head, in all nine models.
Two — the headline — it's **concentrated but not modular**: you can't remove it without paying in
capability, so refusal-rate alone *overstates* a clean safety switch. Three: how cleanly you can remove
it scales with **depth**, the location **migrates** across generations, and newer isn't safer.

And the methodological lesson, in the box: a **capability control is mandatory** in any ablation study.
Without that perplexity check, "refusal went to zero" reads as success when it's really model breakage.

Thank you — I'm happy to take questions.

---

# Q&A prep — likely questions and answers

*No backup slides — these are verbal answers for the 5-minute Q&A. You can flip back to slide 6 (the
pipeline), slide 14 (the full table) or slide 8 (the heatmaps) on screen if useful.*

- **"Why these nine models?"**
  The design is **within-family generational sweeps** — Qwen 1.5/2/2.5/3, Gemma 1/2/3, Llama-3.2 1B and
  3B — because that's exactly what lets us see the circuit *move* (Finding C). The hard constraint is that
  every model must load into TransformerLens' `HookedTransformer` so we can hook activations cleanly.
  Excluded, with reasons: **TinyLlama** barely refuses, so there's no signal; **Phi-3** loads but emits
  garbage logits because its combined-QKV projection is mis-mapped in TransformerLens; **Falcon3, OLMo-2,
  Gemma-4** aren't in the pinned TransformerLens model list (Gemma-4 is also multimodal).

- **"Isn't a negative result — H3b failing — a weak outcome?"**
  No, it's the contribution. The clean "safety switch" story is the intuitive prediction, and we show,
  *with a capability control*, that it's wrong. That's more surprising and more useful than confirming it,
  and it directly informs safety editing and unlearning.

- **"How reliable is the refusal metric?"**
  Two metrics that cross-check. A **logit margin** for the inner loop — logit on refusal tokens ("I",
  "Sorry", "As"…) minus the best non-refusal continuation, at the first token; cheap and differentiable.
  And a **regex string-match rate** on a 30-token reply for human-readable confirmation. We save every
  reply, and there's a 50-prompt human-audit harness to report agreement.

- **"Why perplexity? Why not just report refusal rate?"**
  Refusal rate alone can't tell *compliance* from *breakage*. Perplexity is what reveals the zero-refusal
  models are emitting gibberish. And it isn't a zero-vs-mean-ablation artifact — mean-ablation doesn't
  rescue it; Qwen2.5 is still ~50× baseline.

- **"What's actually patched — attention pattern or value output?"**
  Primarily the head's output (`z`). We also ran last-token-only and attention-pattern-only variants as
  robustness checks; the dominant-head story holds across them.

- **"Isn't the Gemma generational comparison confounded by size?"**
  Yes, fair — Gemma-3 is 1B where Gemma-1 and -2 are 2B, because Google dropped the 2B tier. I flag it as
  a limitation. But the *direction* — deeper each generation — is clear, and Qwen shows the opposite drift
  at a constant size class, so it isn't just a size effect.

- **"Only models up to 3B — does this hold at scale?"**
  That's the honest limitation; a single T4 caps us, so 4B-plus is explicit future work. The cross-family
  consistency at this scale is what makes scaling up the natural next experiment.
