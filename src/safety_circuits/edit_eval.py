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
