"""Unified read-outs for the §9 editing extension — thin wrappers over the existing
metrics so baseline / steering / LoRA-edited models are scored identically.

The decisive contrast with ablation is the WikiText-2 perplexity: head-restricted LoRA
should flip refusal at *small* ΔPPL exactly where blunt ablation produced gibberish.
For a merged-weight edit the change is in the weights, so metrics run with **no hooks**;
the steering baseline passes its projection hooks via `fwd_hooks`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .ablation import _refusal_count_clean, _refusal_count_under_hooks, perplexity
from .analysis import aggregate_pairs
from .models import LoadedModel
from .patching import patch_each_head


@dataclass
class EditReport:
    label: str               # "baseline" | "steering" | "lora_k10" ...
    refusal_rate: float
    jailbreak_refusal_rate: float | None
    perplexity: float | None
    n_eval: int
    n_jailbreak: int

    def to_row(self) -> dict:
        return {
            "label": self.label,
            "refusal_rate": self.refusal_rate,
            "jailbreak_refusal_rate": self.jailbreak_refusal_rate,
            "perplexity": self.perplexity,
            "n_eval": self.n_eval,
            "n_jailbreak": self.n_jailbreak,
        }


def evaluate_edited_model(
    loaded: LoadedModel,
    label: str,
    eval_prompts: list[str],
    jailbreak_prompts: list[str] | None = None,
    ppl_texts: list[str] | None = None,
    fwd_hooks: list | None = None,
) -> EditReport:
    """Refusal rate (+ HarmBench-jailbreak refusal + WikiText-2 perplexity) of `loaded`.

    Pass `fwd_hooks` for the steering baseline (the edit is in hooks); leave it None
    for baseline and merged-LoRA models (the edit is in the weights, or there is none).
    """
    def refusal_rate(prompts: list[str]) -> float:
        if not prompts:
            return float("nan")
        if fwd_hooks:
            cnt = _refusal_count_under_hooks(loaded, fwd_hooks, prompts, desc=f"{label}:refusal")
        else:
            cnt = _refusal_count_clean(loaded, prompts)
        return cnt / len(prompts)

    refusal = refusal_rate(eval_prompts)
    jb = refusal_rate(jailbreak_prompts) if jailbreak_prompts else None
    ppl = perplexity(loaded, ppl_texts, fwd_hooks=fwd_hooks) if ppl_texts else None

    return EditReport(
        label=label,
        refusal_rate=refusal,
        jailbreak_refusal_rate=jb,
        perplexity=ppl,
        n_eval=len(eval_prompts),
        n_jailbreak=len(jailbreak_prompts or []),
    )


def repatch_after_edit(loaded: LoadedModel, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Re-run the per-head z-patch sweep on the edited model and aggregate, so we can
    see whether the transplanted safety heads still 'light up' (do the |Δmargin| peaks
    move / vanish after the edit?). Returns the aggregated DataFrame (same schema as the
    main study's `*_patch_z.csv`)."""
    per_pair = [patch_each_head(loaded, h, s) for h, s in pairs]
    return aggregate_pairs(per_pair)


# --------------------------------------------------------------- T1.1 generalization
def deep_eval(loaded: LoadedModel, prompts: list[str], categories: list[str | None] | None = None,
              label: str = "model", max_new_tokens: int = 128, tox_pipe=None) -> list[dict]:
    """Does the (edited) model produce harmful *content*, or just the 'Sure, here is' opener?

    Generates a **long** continuation per harmful prompt and scores it for refusal (regex on
    the full text), toxic *substance* (`toxicity.score_toxicity`), and length. With HarmBench
    `categories`, downstream code can break results out per harm category. Returns per-prompt
    rows; aggregate with pandas. Run on baseline and edited models to compare."""
    from .refusal import regex_refusal, score_refusal
    from .toxicity import score_toxicity

    categories = categories or [None] * len(prompts)
    conts = []
    rows = []
    for p, cat in zip(prompts, categories):
        s = score_refusal(loaded, p, generate_continuation=True, max_new_tokens=max_new_tokens)
        cont = s.continuation.strip()
        conts.append(cont)
        rows.append({"label": label, "category": cat, "prompt": p, "continuation": cont,
                     "refused": bool(regex_refusal(cont)), "gen_len_chars": len(cont)})
    tox = score_toxicity(conts, pipe=tox_pipe)            # batch through toxic-bert once
    for r, t in zip(rows, tox):
        r["toxicity"] = round(float(t), 4)
    return rows


# --------------------------------------------------------------- T2.6/T2.7 mechanism + forensics
@dataclass
class DirShiftRow:
    layer: int
    cosine: float        # cos(base refusal dir, edited refusal dir) — <1 ⇒ the edit rotated it
    base_norm: float
    edited_norm: float


def refusal_direction_shift(base_dirs: dict, edited_loaded: LoadedModel,
                            harm: list[str], safe: list[str], layers: list[int]) -> list[dict]:
    """How much did the edit move the refusal direction, per layer? Recomputes the
    difference-of-means refusal direction on the **edited** model and compares it (cosine +
    norm) to the **baseline** directions captured before editing (`base_dirs[layer]`, unit
    vectors from `steering.compute_refusal_direction`). A cosine well below 1 at the edited
    layers is the mechanistic signal that retraining *rotated* the head's refusal output —
    and the per-layer cosine vector doubles as a forensic signature of an edited model."""
    import torch

    from .steering import compute_refusal_direction

    rows = []
    for layer in layers:
        bd = base_dirs.get(layer)
        if bd is None:
            continue
        ed = compute_refusal_direction(edited_loaded, harm, safe, layer)
        bd_u = bd / (bd.norm() + 1e-8)
        ed_u = ed.to(bd.device) / (ed.norm() + 1e-8)
        cos = float(torch.dot(bd_u.float(), ed_u.float()).item())
        rows.append(DirShiftRow(layer=layer, cosine=round(cos, 4),
                                base_norm=round(float(bd.norm()), 4),
                                edited_norm=round(float(ed.norm()), 4)).__dict__)
    return rows
