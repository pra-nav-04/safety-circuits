"""HarmBench jailbreak stress test (G9).

Once patching + ablation has identified a model's top-K safety heads on plain
harmful prompts (AdvBench), we ask two questions on adversarial jailbreaks
(HarmBench):

1. **Do the heads still fire?** — does the model still refuse jailbreaks, and is
   the refusal-logit margin still positive (vs plain harmful prompts)? A drop in
   clean refusal on jailbreaks quantifies how much the jailbreak bypasses safety.
2. **Are they still causal?** — does ablating the same top-K heads remove refusal
   on jailbreaks the way it does on plain prompts?

"Still refuses" + "ablation still collapses it" ⇒ the same circuit generalises to
adversarial inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm import tqdm

from .ablation import (
    AblationMode,
    HeadRef,
    _build_head_ablation_hooks,
    _refusal_count_clean,
    _refusal_count_under_hooks,
)
from .models import LoadedModel
from .refusal import score_refusal


@dataclass
class JailbreakReport:
    n_plain: int
    n_jailbreak: int
    refusal_rate_clean_plain: float
    refusal_rate_clean_jailbreak: float
    refusal_rate_ablated_jailbreak: float
    mean_margin_plain: float
    mean_margin_jailbreak: float


@torch.no_grad()
def _mean_margin(loaded: LoadedModel, prompts: list[str], desc: str) -> float:
    vals = [
        score_refusal(loaded, p, generate_continuation=False).margin
        for p in tqdm(prompts, desc=desc)
    ]
    return float(sum(vals) / max(len(vals), 1))


@torch.no_grad()
def jailbreak_stress_test(
    loaded: LoadedModel,
    heads: list[HeadRef],
    jailbreak_prompts: list[str],
    plain_prompts: list[str],
    mode: AblationMode = "zero",
    mean_cache: dict[str, torch.Tensor] | None = None,
) -> JailbreakReport:
    """Compare safety-head behaviour on plain harmful vs adversarial jailbreak prompts.

    `heads` is the identified top-K safety set (from the patching sweep). `mode`
    controls how those heads are ablated on the jailbreak split.
    """
    n_pl, n_jb = len(plain_prompts), len(jailbreak_prompts)

    refusal_clean_plain = _refusal_count_clean(loaded, plain_prompts) / max(n_pl, 1)
    refusal_clean_jb = _refusal_count_clean(loaded, jailbreak_prompts) / max(n_jb, 1)

    hooks = _build_head_ablation_hooks(heads, mode, mean_cache)
    refusal_abl_jb = (
        _refusal_count_under_hooks(loaded, hooks, jailbreak_prompts, desc="ablated-jb")
        / max(n_jb, 1)
    )

    return JailbreakReport(
        n_plain=n_pl,
        n_jailbreak=n_jb,
        refusal_rate_clean_plain=refusal_clean_plain,
        refusal_rate_clean_jailbreak=refusal_clean_jb,
        refusal_rate_ablated_jailbreak=refusal_abl_jb,
        mean_margin_plain=_mean_margin(loaded, plain_prompts, desc="margin-plain"),
        mean_margin_jailbreak=_mean_margin(loaded, jailbreak_prompts, desc="margin-jb"),
    )
