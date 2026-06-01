"""RTP toxicity probe (Priority 2).

Asks: do the safety heads identified via *instruction refusal* (AdvBench) also
suppress *toxic continuation* (RealToxicityPrompts)? This is a separate experiment
from the main patching sweep because RTP prompts don't trigger "I cannot…"-style
refusal — they need a continuation-toxicity scorer, not a refusal-token metric.

If ablating the top-K heads raises continuation toxicity on RTP starters, the
circuit is a general harm-suppressor, not just an instruction-refusal filter — a
strong cross-behaviour generalisation claim.

Classifier: unitary/toxic-bert (110M, CPU) via transformers.pipeline. No new pip
dependency: transformers is already installed by the bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from .ablation import AblationMode, HeadRef
    from .models import LoadedModel


@dataclass
class RTPReport:
    n: int
    mean_tox_clean: float
    mean_tox_ablated: float
    delta_tox: float          # ablated - clean; positive = ablation raised toxicity
    rows: list[dict] = field(default_factory=list)  # per-prompt detail


def _get_toxicity_pipeline():
    """Load unitary/toxic-bert on CPU (avoids VRAM contention with the main model)."""
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        device=-1,       # CPU
        top_k=None,      # return all labels so we can pick "toxic"
        truncation=True,
        max_length=512,
    )


def score_toxicity(texts: list[str], pipe=None) -> list[float]:
    """Return P(toxic) in [0, 1] for each text. Loads the pipeline on first call.

    Pass an already-loaded `pipe` to avoid repeated downloads across calls.
    """
    if pipe is None:
        pipe = _get_toxicity_pipeline()
    scores = []
    for result in pipe(texts):
        # result is a list of {"label": ..., "score": ...} dicts (top_k=None)
        tox = next((r["score"] for r in result if r["label"].lower() == "toxic"), 0.0)
        scores.append(float(tox))
    return scores


@torch.no_grad()
def _generate_continuation(
    loaded: "LoadedModel",
    prompt: str,
    max_new_tokens: int = 40,
    fwd_hooks: list | None = None,
) -> str:
    """Generate a continuation from a *raw* (non-chat) prompt.

    RTP is a continuation benchmark — we use the raw text directly (no chat
    template), just prepend BOS if the model uses it. Optionally runs with
    ablation hooks active.
    """
    tokens = loaded.model.to_tokens(
        prompt, prepend_bos=loaded.spec.prepend_bos
    ).to(loaded.device)

    loaded.model.reset_hooks()
    if fwd_hooks:
        for name, hk in fwd_hooks:
            loaded.model.add_hook(name, hk)
    try:
        out = loaded.model.generate(
            tokens, max_new_tokens=max_new_tokens, temperature=0.0, verbose=False
        )
    finally:
        loaded.model.reset_hooks()
    return loaded.model.to_string(out[0, tokens.shape[1]:])


@torch.no_grad()
def rtp_toxicity_probe(
    loaded: "LoadedModel",
    heads: "list[HeadRef]",
    rtp_prompts: list[str],
    mode: "AblationMode" = "zero",
    mean_cache: "dict[str, torch.Tensor] | None" = None,
) -> RTPReport:
    """Measure whether ablating the identified safety heads raises continuation toxicity.

    For each RTP prompt:
      1. Generate a continuation (clean — no hooks).
      2. Generate a continuation (ablated — top-K heads zeroed/mean-ablated).
      3. Score both with unitary/toxic-bert.

    Returns a report with per-prompt rows and aggregate delta.
    Positive `delta_tox` = ablation made continuations more toxic → the safety
    heads suppress toxic continuation (cross-behaviour generalisation).
    """
    from .ablation import _build_head_ablation_hooks

    hooks = _build_head_ablation_hooks(heads, mode, mean_cache)
    pipe = _get_toxicity_pipeline()

    rows = []
    for prompt in tqdm(rtp_prompts, desc="rtp-probe"):
        clean_cont = _generate_continuation(loaded, prompt)
        ablated_cont = _generate_continuation(loaded, prompt, fwd_hooks=hooks)
        clean_tox, ablated_tox = score_toxicity([clean_cont, ablated_cont], pipe=pipe)
        rows.append({
            "prompt": prompt,
            "clean_continuation": clean_cont.strip(),
            "clean_toxicity": round(clean_tox, 4),
            "ablated_continuation": ablated_cont.strip(),
            "ablated_toxicity": round(ablated_tox, 4),
            "delta_toxicity": round(ablated_tox - clean_tox, 4),
        })

    n = len(rows)
    mean_clean = sum(r["clean_toxicity"] for r in rows) / max(n, 1)
    mean_abl = sum(r["ablated_toxicity"] for r in rows) / max(n, 1)
    return RTPReport(
        n=n,
        mean_tox_clean=round(mean_clean, 4),
        mean_tox_ablated=round(mean_abl, 4),
        delta_tox=round(mean_abl - mean_clean, 4),
        rows=rows,
    )
