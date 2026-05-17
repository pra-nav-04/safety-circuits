"""Cache and inspect activations with TransformerLens hooks."""

from __future__ import annotations

import torch
from transformer_lens import ActivationCache

from .models import LoadedModel, apply_chat_template


@torch.no_grad()
def run_with_cache(
    loaded: LoadedModel,
    prompt: str,
    names_filter: list[str] | None = None,
) -> tuple[torch.Tensor, ActivationCache]:
    """Forward pass that captures activations at hook points.

    Pass `names_filter` (a list of hook-name substrings) to keep memory bounded —
    e.g., `["z", "mlp_out", "resid_pre"]` covers everything we patch.
    """
    chat = apply_chat_template(loaded, prompt)
    tokens = loaded.model.to_tokens(chat, prepend_bos=True).to(loaded.device)

    if names_filter is not None:
        filt = lambda name: any(s in name for s in names_filter)  # noqa: E731
    else:
        filt = None

    logits, cache = loaded.model.run_with_cache(tokens, names_filter=filt)
    return logits, cache


def first_pred_logits(logits: torch.Tensor) -> torch.Tensor:
    """Logits for the first generated token (post-prompt)."""
    return logits[0, -1]
