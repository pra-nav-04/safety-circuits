"""Activation patching (causal tracing).

Given matched prompts `p_harm` and `p_safe`, we ask: *for each (layer, head) — does
replacing the head's output during the harmful run with the value it had on the
safe run preserve refusal, or destroy it?*

If destroying it (`Δrefusal_margin` large and negative) means the head was
**carrying the refusal signal** — a "safety head" candidate.

This is the Wang-et-al / Meng-et-al methodology, scoped to a single token of
interest (the first generated token).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
from tqdm import tqdm
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name

from .activations import run_with_cache
from .models import LoadedModel, apply_chat_template
from .refusal import _refusal_token_ids


Component = Literal["z", "pattern", "attn_out", "mlp_out", "resid_pre", "resid_mid", "resid_post"]


@dataclass
class PatchResult:
    component: Component
    layer: int
    head: int | None        # None for MLP / residual patches
    delta_margin: float     # patched − clean


def _refusal_margin(logits_last: torch.Tensor, refusal_ids: list[int]) -> torch.Tensor:
    log_probs = torch.log_softmax(logits_last, dim=-1)
    refusal_mass = torch.logsumexp(log_probs[refusal_ids], dim=-1)
    mask = torch.ones_like(log_probs, dtype=torch.bool)
    mask[refusal_ids] = False
    other_mass = torch.logsumexp(log_probs[mask], dim=-1)
    return refusal_mass - other_mass


def _make_patch_hook(
    cache: ActivationCache,
    hook_name: str,
    head: int | None,
    position: int | None,
) -> Callable:
    """Return a hook that overwrites a component with cached values from another run."""
    source = cache[hook_name]

    def hook(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        # activation: [batch, seq, ...] or [batch, seq, head, d_head] for z
        src = source.to(activation.device, dtype=activation.dtype)
        # Ensure src has a batch dim (some TL versions cache without it).
        if src.dim() < activation.dim():
            src = src.unsqueeze(0)
        if head is None:
            if position is None:
                # Patch the common prefix; prompts may differ in token length.
                n = min(activation.shape[1], src.shape[1])
                activation[:, :n] = src[:, :n]
            else:
                activation[:, position] = src[:, position]
        else:
            # z hook: [batch, seq, head, d_head]
            if position is None:
                n = min(activation.shape[1], src.shape[1])
                activation[:, :n, head, :] = src[:, :n, head, :]
            else:
                activation[:, position, head, :] = src[:, position, head, :]
        return activation

    return hook


@torch.no_grad()
def patch_each_head(
    loaded: LoadedModel,
    harm_prompt: str,
    safe_prompt: str,
    position: int | None = None,
) -> list[PatchResult]:
    """Sweep every (layer, head): patch the head's `z` from safe run into harmful run.

    `position=None` patches every token position (coarser but cheaper). For a fine
    causal trace, set `position=-1` (last token only) and run a positional sweep
    separately.
    """
    refusal_ids = _refusal_token_ids(loaded)

    # 1. cache the safe run (the donor)
    _, safe_cache = run_with_cache(loaded, safe_prompt, names_filter=["z"])

    # 2. baseline: clean harmful run
    chat_harm = apply_chat_template(loaded, harm_prompt)
    harm_tokens = loaded.model.to_tokens(chat_harm, prepend_bos=True).to(loaded.device)
    clean_logits = loaded.model(harm_tokens)
    clean_margin = _refusal_margin(clean_logits[0, -1], refusal_ids).item()

    results: list[PatchResult] = []
    for layer in tqdm(range(loaded.n_layers), desc="patching heads"):
        hook_name = get_act_name("z", layer)
        for head in range(loaded.n_heads):
            hook = _make_patch_hook(safe_cache, hook_name, head=head, position=position)
            patched_logits = loaded.model.run_with_hooks(
                harm_tokens,
                fwd_hooks=[(hook_name, hook)],
                return_type="logits",
            )
            patched_margin = _refusal_margin(patched_logits[0, -1], refusal_ids).item()
            results.append(
                PatchResult(
                    component="z",
                    layer=layer,
                    head=head,
                    delta_margin=patched_margin - clean_margin,
                )
            )
    return results


@torch.no_grad()
def patch_each_head_pattern(
    loaded: LoadedModel,
    harm_prompt: str,
    safe_prompt: str,
) -> list[PatchResult]:
    """Sweep every (layer, head): patch the head's *attention pattern* safe→harm.

    Complements `patch_each_head`. `z`-patching replaces what a head *writes*
    (its value-weighted output); pattern-patching replaces *where it attends*
    (post-softmax weights, `[batch, head, query, key]`). A head that matters via
    `z` but not pattern is moving information without re-routing attention; one
    that matters via pattern is changing what the model looks at. Always
    position-agnostic (the whole query×key matrix is patched over the common prefix).
    """
    refusal_ids = _refusal_token_ids(loaded)
    _, safe_cache = run_with_cache(loaded, safe_prompt, names_filter=["pattern"])

    chat_harm = apply_chat_template(loaded, harm_prompt)
    harm_tokens = loaded.model.to_tokens(chat_harm, prepend_bos=True).to(loaded.device)
    clean_logits = loaded.model(harm_tokens)
    clean_margin = _refusal_margin(clean_logits[0, -1], refusal_ids).item()

    results: list[PatchResult] = []
    for layer in tqdm(range(loaded.n_layers), desc="patching patterns"):
        hook_name = get_act_name("pattern", layer)
        src = safe_cache[hook_name]  # [head, q, k] or [batch, head, q, k]

        for head in range(loaded.n_heads):
            def hook(pattern: torch.Tensor, hook: HookPoint, head=head, src=src) -> torch.Tensor:
                # pattern: [batch, head, query, key]
                s = src.to(pattern.device, dtype=pattern.dtype)
                if s.dim() < pattern.dim():
                    s = s.unsqueeze(0)
                n_q = min(pattern.shape[-2], s.shape[-2])
                n_k = min(pattern.shape[-1], s.shape[-1])
                pattern[:, head, :n_q, :n_k] = s[:, head, :n_q, :n_k]
                return pattern

            patched_logits = loaded.model.run_with_hooks(
                harm_tokens, fwd_hooks=[(hook_name, hook)], return_type="logits"
            )
            patched_margin = _refusal_margin(patched_logits[0, -1], refusal_ids).item()
            results.append(
                PatchResult(component="pattern", layer=layer, head=head,
                            delta_margin=patched_margin - clean_margin)
            )
    return results


@torch.no_grad()
def patch_each_mlp(
    loaded: LoadedModel,
    harm_prompt: str,
    safe_prompt: str,
    position: int | None = None,
) -> list[PatchResult]:
    refusal_ids = _refusal_token_ids(loaded)
    _, safe_cache = run_with_cache(loaded, safe_prompt, names_filter=["mlp_out"])

    chat_harm = apply_chat_template(loaded, harm_prompt)
    harm_tokens = loaded.model.to_tokens(chat_harm, prepend_bos=True).to(loaded.device)
    clean_logits = loaded.model(harm_tokens)
    clean_margin = _refusal_margin(clean_logits[0, -1], refusal_ids).item()

    results = []
    for layer in tqdm(range(loaded.n_layers), desc="patching MLPs"):
        hook_name = get_act_name("mlp_out", layer)
        hook = _make_patch_hook(safe_cache, hook_name, head=None, position=position)
        patched_logits = loaded.model.run_with_hooks(
            harm_tokens, fwd_hooks=[(hook_name, hook)], return_type="logits"
        )
        patched_margin = _refusal_margin(patched_logits[0, -1], refusal_ids).item()
        results.append(
            PatchResult(component="mlp_out", layer=layer, head=None, delta_margin=patched_margin - clean_margin)
        )
    return results


@torch.no_grad()
def patch_residual_stream(
    loaded: LoadedModel,
    harm_prompt: str,
    safe_prompt: str,
    point: Component = "resid_pre",
) -> list[PatchResult]:
    """Coarse trace — patch the whole residual stream at each layer.

    A useful first pass: it tells you *which layer band* matters before you spend
    n_layers × n_heads on the fine head sweep.
    """
    assert point in {"resid_pre", "resid_mid", "resid_post"}
    refusal_ids = _refusal_token_ids(loaded)
    _, safe_cache = run_with_cache(loaded, safe_prompt, names_filter=[point])

    chat_harm = apply_chat_template(loaded, harm_prompt)
    harm_tokens = loaded.model.to_tokens(chat_harm, prepend_bos=True).to(loaded.device)
    clean_logits = loaded.model(harm_tokens)
    clean_margin = _refusal_margin(clean_logits[0, -1], refusal_ids).item()

    results = []
    for layer in tqdm(range(loaded.n_layers), desc=f"patching {point}"):
        hook_name = get_act_name(point, layer)
        hook = _make_patch_hook(safe_cache, hook_name, head=None, position=None)
        patched_logits = loaded.model.run_with_hooks(
            harm_tokens, fwd_hooks=[(hook_name, hook)], return_type="logits"
        )
        patched_margin = _refusal_margin(patched_logits[0, -1], refusal_ids).item()
        results.append(
            PatchResult(component=point, layer=layer, head=None, delta_margin=patched_margin - clean_margin)
        )
    return results
