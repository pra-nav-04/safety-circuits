# Speaker script — Final Presentation
### *Finding the Safety Switches: from mapping refusal to editing it*
MA-INF 4330 · Lab Explainable AI and Applications · University of Bonn

**How to use this file:** this is a full word-for-word script for rehearsal. Read it aloud
to memorise the *flow*; on the day, glance at `talking_points_final.md` (the cue version),
not this — the instructor's advice is *don't read the slides, stand and speak freely, keep
eye contact.* `[Bracketed italics]` are stage directions — pause, point, click — not words
to say.

**Hard limit: 10 minutes, strictly enforced.** This script is paced to **~9:20**, leaving a
~40-second cushion. There are **12 main slides** plus an **appendix** of backup slides you
only show if a question calls for it.

**Pacing ribbon (cumulative clock — glance while practising):**

| # | Slide | Length | Clock |
|---|-------|--------|-------|
| 1 | Title | 0:10 | 0:10 |
| 2 | Two acts | 0:30 | 0:40 |
| 3 | Why look inside + RQ | 0:45 | 1:25 |
| 4 | Method | 0:50 | 2:15 |
| 5 | Act 1 — concentrated, not modular | 0:55 | 3:10 |
| 6 | Act 2 — delete → edit | 0:35 | 3:45 |
| 7 | Finding E — editable | 0:55 | 4:40 |
| 8 | Finding F — fake jailbreak | 1:00 | 5:40 |
| 9 | Finding G — real jailbreak | 1:05 | 6:45 |
| 10 | Finding G — concentration & tuning | 0:55 | 7:40 |
| 11 | Did we jailbreak them? | 0:50 | 8:30 |
| 12 | Conclusion | 0:50 | 9:20 |

> **Two acts, one thread.** Everything ladders to one sentence: *you cannot delete refusal
> without breaking the model — but you can retrain the very same heads into a jailbreak,
> using no harmful data at all.* If you forget everything else, land that.
>
> **Memorise these first two sentences** (confident start): *"Good morning — my project is
> about finding the circuit inside a language model that makes it refuse harmful requests.
> And then asking a harder question: what happens when you try to edit that circuit?"*

---

## Slide 1 — Title *(0:10)*

"Good morning — my project is about finding the circuit inside a language model that makes it
refuse harmful requests. And then asking a harder question: what happens when you try to edit
that circuit?"

*[Click to slide 2.]*

---

## Slide 2 — The story in two acts *(0:30)*

"The talk is in two acts. In **Act 1** we *map* the circuit — where does refusal live, and can
we simply delete it? That's the mapping study, and I'll go through it quickly. The finding
that comes out of it — *concentrated, but not modular* — is the bridge into **Act 2**, which is
all new: instead of deleting the circuit, we *retrain* it, and that turns our map into an
actual jailbreak built with no harmful data. Nine models; most of my time is in Act 2."

*[Click to slide 3.]*

---

## Slide 3 — Why look inside? + RQ *(0:45)*

"Why look *inside* a guardrail? Models are aligned to refuse harmful requests, but we treat
that refusal as a black box — and we know it's leaky, because jailbreaks get through. Classic
explainability only tells us what *correlates* with the output. Mechanistic interpretability
reads the actual wiring and asks which components *causally* produce the behaviour.

So three questions: can we **point to** the circuit, can we **remove it** — and what breaks if
we do — and does it **look the same** across model families? *[Gesture to the box.]* Formally:
are there a few attention heads that causally produce refusal, and how does that circuit change
across families and generations?"

*[Click to slide 4.]*

---

## Slide 4 — Method *(0:50)*

"Here's the method on one slide — all read-only, forward passes and hooks, no training in Act 1.

*[Trace left to right.]* For each model we build fifty matched harmful-versus-benign pairs, do
**activation patching** — swap one head's output from the benign run into the harmful run and
measure how much refusal moves — take the top ten heads, and ablate them.

And the part that matters most: alongside the refusal rate we measure **perplexity**, a
capability check. *[Pause.]* This is the crux. Without it, 'refusal dropped to zero' looks like
a win; with it, you find out whether you removed the safety or just broke the model. Quick note
on scope — nine instruct models across three families; *why* exactly these is a backup slide."

*[Click to slide 5.]*

---

## Slide 5 — Act 1: concentrated, not modular *(0:55)*

"Two things come out of mapping. First, refusal *is* sparse and causal — a single head
dominates in all nine models, and patching that one head flips the refusal. Nine out of nine.

But now the key plot. *[Point to the scatter.]* On the x-axis, how much refusal we removed;
on the y-axis, on a log scale, the damage to perplexity. The four models that drop to *zero
percent refusal* are **exactly** the four whose perplexity explodes — Qwen2.5 by
sixty-one-thousand-fold. That output isn't compliance, it's gibberish.

*[Slow down.]* Not one model removed refusal *and* kept its capability. So H3b — the clean
safety switch — is **falsified.** Refusal is concentrated, but it is *not* a modular part you
can cut out."

*[Click to slide 6.]*

---

## Slide 6 — Act 2: delete → edit *(0:35)*

"And that's the pivot. Everything so far could only ever *delete* these heads — and deletion
breaks the model. So Act 2 asks: if you can't delete refusal, can you *edit* it?

*[Point across the three boxes.]* Think of three tools on the same heads, blunt to sharp: crude
ablation, a no-train steering vector, and the sharp one — a **head-restricted LoRA** that
retrains *only* those localised safety heads. That LoRA is the one deliberate exception to my
'no training' rule."

*[Click to slide 7.]*

---

## Slide 7 — Finding E: editable under retraining *(0:55)*

"Finding E. *[Point to the 'clean corner' of the plot — low refusal, low cost.]* The LoRA
drives refusal to **zero on all nine models**, and on six of the nine it does that at almost no
capability cost — under ten percent. Remember, these are the *same heads* where crude ablation
blew perplexity up a hundred- to sixty-thousand-fold. Steering, by contrast, never cleanly
removes refusal on any model.

So the picture flips: refusal is *not* modular under deletion, but it *is* editable under
retraining. *[Honesty beat.]* One caveat I'll be straight about — it's not universal; on Gemma-1
the clean edit costs ten times the perplexity."

*[Click to slide 8.]*

---

## Slide 8 — Finding F: the obvious jailbreak is fake *(1:00)*

"Now, the honest version of the story. The naive way to weaponise this is to train the heads on
'Sure, here is…'. On paper it looks like a total jailbreak — raw attack-success around
**ninety percent.** *[Pause.]* But read what the model actually writes.

*[Point to the bubble.]* 'Sure, here is a letter defaming a company…' — and then it *stops.*
An affirmative opener, no harmful body. So we measure **substantive** success — harmful *and*
at least two hundred characters of real content — and by that honest metric the naive edit
scores **zero** on all nine. The ninety-percent number was the classifier fooled by the opening
line. The cause is mundane: a twenty-four-token training target teaches the opener, not the
answer. Lesson — report substantive success, never raw."

*[Click to slide 9.]*

---

## Slide 9 — Finding G: the real, benign-data jailbreak *(1:05)*

"Which sets up the real result — Finding G. If shallowness was just a short target, what if we
teach the heads to give *full* answers, but only ever on **benign** questions? Does that transfer
to harmful ones?

*[Point to the highlighted bar.]* It does. On Llama-3.2-3B, substantive harmful compliance goes
from **eight percent to forty-one** — confidence intervals that don't overlap, a five-fold jump —
and, I want to stress, **with no harmful training data anywhere.** We only ever showed it benign
answers.

*[Beat — the counter-intuitive part.]* And notice *where* it's largest: on the models that were
*best* aligned to begin with. The weakly-aligned ones barely move — they were already compliant.
So the stronger the alignment, the more there is to strip. This is at a hundred and fifty prompts,
judged by a thirteen-billion classifier, with bootstrap confidence intervals."

*[Click to slide 10.]*

---

## Slide 10 — Finding G: concentration & tuning *(0:55)*

"Three things sharpen it. *[Point to the bars.]* **Where** does it concentrate? In procedural
harm — cybercrime how-tos hit eighty-four percent. That makes sense: teaching the model to give
detailed answers is a how-to generator.

**How far?** Tuning pushes the ceiling to about **forty-eight percent**, and the decisive lever
is counter-intuitive — *fewer* heads, because a tighter edit stays coherent, so more of the
output is substantive. And one **negative result** worth reporting: combining the LoRA with
steering doesn't stack — it's *worse* — because steering degrades the coherence the LoRA needs."

*[Click to slide 11.]*

---

## Slide 11 — Did we jailbreak the models? *(0:50)*

"So — did we jailbreak the models? *[Answer plainly.]* Yes, but in a measured sense, and only
on the well-aligned ones. Retraining a handful of localised safety heads on **purely benign
data** turns a well-aligned model into one that gives substantive harmful answers to roughly
forty to forty-eight percent of requests, with no harmful content in training.

Three things keep this honest. *[Count them off.]* One — the *obvious* jailbreak is fake: it
looks like ninety percent but produces zero real content. Two — the *real* one transfers from
benign training; you never show it a harmful answer. Three — it's largest where alignment was
strongest, concentrates in procedural harm, and doesn't compose with steering."

*[Click to slide 12.]*

---

## Slide 12 — Conclusion *(0:50)*

"To wrap up. Refusal is sparse and causal, but *concentrated, not modular* — you can't delete
it without breaking the model. Its modularity scales with depth, its location migrates across
generations, and newer is not safer. And the same heads that resist deletion are **editable** —
retraining them on benign data alone transfers to forty-to-forty-eight-percent harmful
compliance on the best-aligned model.

*[Land the lesson.]* The method lesson runs through the whole talk: a capability control, and
reading the *content* rather than just the opening token, are what turn an apparent success into
an honest result. This was done responsibly — no harmful training data, severe categories
excluded, no weights released — and the natural next step is defensive: train refusal to be
*distributed*, so this kind of head-editing can't remove it.

*[Beat.]* Thank you — I'm happy to take questions."

*[Stop. Leave the conclusion slide up; jump to an appendix slide if a question calls for it.]*

---

### Rehearsal notes
- **If you're running long:** compress slide 10 to the two headline numbers (0.84, 0.48) and
  cut slide 6 to one sentence — that buys ~30s. The scorecard and full table are already in the
  appendix, so you're not tempted to over-talk them.
- **If you're running short:** add the Llama-1B / Qwen2.5 ~15% numbers on slide 9, or the Gemma
  editability gradient (+927% → +15% → −12%) on slide 7 — but don't dip into the appendix during
  the talk.
- **Three numbers to nail cold:** 61,000× (Qwen2.5 perplexity), 0.08 → 0.41 (the transfer), and
  0% substantive on the fake opener edit.
- **Hard pauses (let them sit a beat):** after "falsified" (slide 5), the pivot on slide 6, and
  after "zero" on slide 8.
- **Appendix map for Q&A:** why-these-models · sparsity · depth & migration · jailbreak
  robustness · scorecard · N=50 table · editing detail. Know their order so you can jump quickly.
