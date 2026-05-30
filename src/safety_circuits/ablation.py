"""Zero / mean ablation of candidate components.

Once patching has identified candidates, ablation is the confirmatory test:
*remove* the head and check whether refusal rate collapses on a held-out set,
without breaking general perplexity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name

from .activations import run_with_cache
from .models import LoadedModel, apply_chat_template
from .refusal import score_refusal


AblationMode = Literal["zero", "mean"]


@dataclass
class HeadRef:
    layer: int
    head: int


@dataclass
class AblationReport:
    refusal_rate_clean: float
    refusal_rate_ablated: float
    perplexity_clean: float | None
    perplexity_ablated: float | None

    @property
    def perplexity_pct_change(self) -> float | None:
        """Δ perplexity as a percentage of clean (H3 target: ≤ 5%). None if unmeasured."""
        if self.perplexity_clean is None or self.perplexity_ablated is None:
            return None
        return 100.0 * (self.perplexity_ablated - self.perplexity_clean) / self.perplexity_clean


def _build_head_ablation_hooks(
    heads: list[HeadRef],
    mode: AblationMode,
    mean_cache: dict[str, torch.Tensor] | None,
):
    """Group hooks by layer so we register one per layer (TL needs that)."""
    by_layer: dict[int, list[HeadRef]] = {}
    for h in heads:
        by_layer.setdefault(h.layer, []).append(h)

    hooks = []
    for layer, head_refs in by_layer.items():
        hook_name = get_act_name("z", layer)
        head_idx = [hr.head for hr in head_refs]

        def make_hook(layer=layer, head_idx=head_idx, hook_name=hook_name):
            def hook(z: torch.Tensor, hook: HookPoint) -> torch.Tensor:
                # z: [batch, seq, head, d_head]
                if mode == "zero":
                    z[:, :, head_idx, :] = 0.0
                elif mode == "mean":
                    assert mean_cache is not None
                    mean = mean_cache[hook_name].to(z.device, dtype=z.dtype)  # [head, d_head]
                    # broadcast the constant per-head mean across batch & all positions
                    z[:, :, head_idx, :] = mean[head_idx, :]
                return z
            return hook

        hooks.append((hook_name, make_hook()))
    return hooks


@torch.no_grad()
def compute_mean_z_cache(
    loaded: LoadedModel, benign_prompts: list[str]
) -> dict[str, torch.Tensor]:
    """Mean head output per (head, d_head), averaged over all token positions and
    all benign prompts — used by mean-ablation.

    Mean-ablation replaces a head's output with this constant vector instead of
    zero (Wang et al.): it controls for the "removed all signal" confound by
    substituting the *typical benign* output rather than nothing. Returns a dict
    `{hook_name: [head, d_head]}`, which broadcasts cleanly across batch and
    sequence length at ablation time (no dependence on prompt length).
    """
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    for p in tqdm(benign_prompts, desc="mean-cache"):
        _, cache = run_with_cache(loaded, p, names_filter=["z"])
        for name, val in cache.items():
            v = val[0].detach()        # [seq, head, d_head]
            s = v.sum(dim=0)           # [head, d_head]
            if name not in sums:
                sums[name] = torch.zeros_like(s)
                counts[name] = 0
            sums[name] = sums[name] + s
            counts[name] += v.shape[0]  # number of positions accumulated
    return {k: sums[k] / counts[k] for k in sums}


@torch.no_grad()
def _refusal_count_clean(loaded: LoadedModel, eval_prompts: list[str]) -> int:
    """Number of `eval_prompts` the unmodified model refuses (regex on continuation)."""
    return sum(
        score_refusal(loaded, p, generate_continuation=True).refused_regex
        for p in tqdm(eval_prompts, desc="clean")
    )


@torch.no_grad()
def _refusal_count_under_hooks(
    loaded: LoadedModel,
    hooks: list,
    eval_prompts: list[str],
    desc: str = "ablated",
) -> int:
    """Number of `eval_prompts` refused while `hooks` (ablation) are active."""
    from .refusal import regex_refusal

    refused = 0
    for p in tqdm(eval_prompts, desc=desc):
        chat = apply_chat_template(loaded, p)
        tokens = loaded.model.to_tokens(chat, prepend_bos=True).to(loaded.device)
        loaded.model.reset_hooks()
        for name, hk in hooks:
            loaded.model.add_hook(name, hk)
        try:
            out = loaded.model.generate(tokens, max_new_tokens=30, temperature=0.0, verbose=False)
        finally:
            loaded.model.reset_hooks()
        continuation = loaded.model.to_string(out[0, tokens.shape[1]:])
        refused += int(regex_refusal(continuation))
    return refused


@torch.no_grad()
def evaluate_ablation(
    loaded: LoadedModel,
    heads: list[HeadRef],
    eval_prompts: list[str],
    mode: AblationMode = "zero",
    mean_cache: dict[str, torch.Tensor] | None = None,
    perplexity_texts: list[str] | None = None,
) -> AblationReport:
    """Refusal rate, clean vs ablated, on `eval_prompts`.

    If `perplexity_texts` is given (e.g. a WikiText-2 slice from
    `data.load_wikitext2`), also reports token-level perplexity clean vs ablated
    on those texts — the H3 capability-preservation control: refusal should
    collapse while general-language perplexity barely moves.
    """
    clean_refused = _refusal_count_clean(loaded, eval_prompts)

    hooks = _build_head_ablation_hooks(heads, mode, mean_cache)
    abl_refused = _refusal_count_under_hooks(loaded, hooks, eval_prompts)

    # capability-preservation control (optional)
    ppl_clean = ppl_ablated = None
    if perplexity_texts:
        ppl_clean = perplexity(loaded, perplexity_texts)
        ppl_ablated = perplexity(loaded, perplexity_texts, fwd_hooks=hooks)

    n = len(eval_prompts)
    return AblationReport(
        refusal_rate_clean=clean_refused / n,
        refusal_rate_ablated=abl_refused / n,
        perplexity_clean=ppl_clean,
        perplexity_ablated=ppl_ablated,
    )


@dataclass
class KSweepPoint:
    k: int
    refusal_rate_ablated: float
    perplexity_ablated: float | None = None


@torch.no_grad()
def ablation_k_sweep(
    loaded: LoadedModel,
    ranked_heads: list[HeadRef],
    eval_prompts: list[str],
    ks: list[int],
    mode: AblationMode = "zero",
    mean_cache: dict[str, torch.Tensor] | None = None,
    perplexity_texts: list[str] | None = None,
) -> tuple[float, list[KSweepPoint]]:
    """Refusal rate as a function of how many top heads are ablated.

    `ranked_heads` is the priority-ordered candidate list (strongest first). For
    each ``k`` in ``ks`` we ablate ``ranked_heads[:k]`` and measure the held-out
    refusal rate (and, if `perplexity_texts` is given, perplexity under that
    ablation). The clean baseline is computed once and returned separately.

    Resolves the Llama-3.2 question — does suppression need k>10? — without
    re-running the expensive patching sweep.

    Returns ``(refusal_rate_clean, [KSweepPoint, ...])``.
    """
    n = len(eval_prompts)
    clean_rate = _refusal_count_clean(loaded, eval_prompts) / n

    points: list[KSweepPoint] = []
    for k in ks:
        hooks = _build_head_ablation_hooks(ranked_heads[:k], mode, mean_cache)
        abl = _refusal_count_under_hooks(loaded, hooks, eval_prompts, desc=f"ablated k={k}")
        ppl = perplexity(loaded, perplexity_texts, fwd_hooks=hooks) if perplexity_texts else None
        points.append(KSweepPoint(k=k, refusal_rate_ablated=abl / n, perplexity_ablated=ppl))
    return clean_rate, points


@torch.no_grad()
def perplexity(
    loaded: LoadedModel,
    texts: Iterable[str],
    fwd_hooks: list | None = None,
) -> float:
    """Token-level perplexity on a stream of texts. Cheap proxy for capability loss.

    Pass `fwd_hooks` (the same `(name, hook)` list produced for ablation) to measure
    perplexity with those hooks active — i.e. perplexity *under ablation*.
    """
    nll, n = 0.0, 0
    for t in texts:
        tokens = loaded.model.to_tokens(t, prepend_bos=True).to(loaded.device)
        if fwd_hooks:
            logits = loaded.model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
        else:
            logits = loaded.model(tokens)
        log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
        target = tokens[0, 1:]
        nll += -log_probs.gather(-1, target.unsqueeze(-1)).sum().item()
        n += target.numel()
    return float(torch.exp(torch.tensor(nll / max(n, 1))).item())
